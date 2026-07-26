#!/usr/bin/env python3
"""Run the prescribed-schedule x metric-buffer configuration comparison.

This is intentionally labelled a configuration comparison: changing the
schedule changes window boundaries and dual-averaging restart cadence, while
the buffer-policy factor changes how metric evidence persists across windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from suite_common import (
    ImmutableJsonl,
    build_multi_chain_cell,
    finish_output,
    provenance,
    sample_single_chain,
)

import blackjax
import numpy as np
from blackjax.adaptation.low_rank_adaptation import (
    build_growing_window_schedule,
    window_adaptation_low_rank,
)
from blackjax.adaptation.staged_adaptation import build_schedule


SCHEDULES: dict[str, Callable] = {
    "stan_doubling": build_schedule,
    "proportional_growing": build_growing_window_schedule,
}
BUFFER_POLICIES = ("reset", "accumulating")


def _window_rows(
    *,
    model: str,
    seed: int,
    schedule_name: str,
    buffer_policy: str,
    schedule: Any,
    info: Any,
) -> list[dict[str, Any]]:
    schedule_array = np.asarray(schedule)
    nis = np.asarray(info.info.num_integration_steps)
    cumulative = np.cumsum(nis)
    step_sizes = np.asarray(info.adaptation_state.step_size)
    lam = np.asarray(info.adaptation_state.lam)
    boundaries: list[int] = []
    for index, (stage, is_window_end) in enumerate(schedule_array):
        is_last = index == schedule_array.shape[0] - 1
        stage_changes = (
            not is_last and int(schedule_array[index + 1, 0]) != int(stage)
        )
        if bool(is_window_end) or stage_changes or is_last:
            boundaries.append(index)
    rows: list[dict[str, Any]] = []
    start = 0
    for index, end in enumerate(boundaries):
        rows.append(
            {
                "record_type": "schedule_event",
                "model": model,
                "seed": seed,
                "schedule_family": schedule_name,
                "buffer_policy": buffer_policy,
                "window_index": index,
                "stage": "slow" if int(schedule_array[end, 0]) else "fast",
                "start_step": start,
                "end_step": int(end),
                "end_fraction": float((end + 1) / schedule_array.shape[0]),
                "cumulative_grads": int(cumulative[end]),
                "step_size_end": float(step_sizes[end]),
                "effective_rank_end": int(
                    np.sum(np.abs(lam[end] - 1.0) > 1e-6)
                ),
            }
        )
        start = end + 1
    return rows


def run_cell(
    model: str,
    seed: int,
    *,
    schedule_name: str,
    buffer_policy: str,
    num_warmup_steps: int,
    num_draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell = build_multi_chain_cell(model, seed)
    schedule_fn = SCHEDULES[schedule_name]
    warmup = window_adaptation_low_rank(
        blackjax.nuts,
        cell["logdensity_fn"],
        max_rank=10,
        schedule_fn=schedule_fn,
        buffer_policy=buffer_policy,
        recompute_every=1,
    )
    started = time.monotonic()
    result, info = warmup.run(
        cell["warmup_key"],
        np.asarray(cell["inits"])[0],
        num_warmup_steps,
    )
    wall_seconds = time.monotonic() - started
    # The wrapper returns the optimal translation in result.state.  The paper's
    # declared comparison starts instead from the actual warmup-end state.
    warmup_end_position = np.asarray(info.state.position)[-1]
    sampled = sample_single_chain(
        blackjax.nuts,
        cell["logdensity_fn"],
        result.parameters,
        warmup_end_position,
        cell["sample_key"],
        num_draws=num_draws,
    )
    warmup_grads = int(np.asarray(info.info.num_integration_steps).sum())
    total_grads = warmup_grads + sampled["sampling_grads"]
    row = {
        "record_type": "cell",
        "model": model,
        "seed": seed,
        "schedule_family": schedule_name,
        "buffer_policy": buffer_policy,
        "recompute_every": 1,
        "max_rank": 10,
        "num_warmup_steps": num_warmup_steps,
        "num_sampling_draws": num_draws,
        "start_policy": "actual_warmup_end",
        "warmup_grads": warmup_grads,
        "warmup_divergences": int(np.asarray(info.info.is_divergent).sum()),
        "sampling_grads": sampled["sampling_grads"],
        "sampling_divergences": sampled["sampling_divergences"],
        "min_bulk_ess": sampled["min_bulk_ess"],
        "ess_per_grad": sampled["min_bulk_ess"] / total_grads,
        "step_size": float(np.asarray(result.parameters["step_size"])),
        "warmup_wall_seconds": wall_seconds,
    }
    events = _window_rows(
        model=model,
        seed=seed,
        schedule_name=schedule_name,
        buffer_policy=buffer_policy,
        schedule=schedule_fn(num_warmup_steps),
        info=info,
    )
    return row, events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events-out", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    events_path = args.events_out or args.out.with_suffix(".events.jsonl")
    models = ["ill_cond_50"] if args.smoke else ["ill_cond_50", "radon"]
    seeds = [0] if args.smoke else [0, 1, 2]
    schedules = ["stan_doubling"] if args.smoke else list(SCHEDULES)
    policies = ["reset"] if args.smoke else list(BUFFER_POLICIES)
    num_warmup_steps = 100 if args.smoke else 1_000
    num_draws = 20 if args.smoke else 4_000
    expected = len(models) * len(seeds) * len(schedules) * len(policies)
    errors = 0
    started = time.monotonic()

    with ImmutableJsonl(args.out) as output, ImmutableJsonl(events_path) as events:
        prov = provenance(
            suite="schedule-configuration",
            argv=sys.argv,
            allow_dirty=args.allow_dirty,
            run_id=args.run_id,
        )
        output.emit(prov)
        events.emit({**prov, "suite": "schedule-configuration-events"})
        for model in models:
            for seed in seeds:
                for schedule_name in schedules:
                    for policy in policies:
                        label = (
                            f"{model}:seed={seed}:schedule={schedule_name}:"
                            f"buffer={policy}"
                        )
                        try:
                            row, event_rows = run_cell(
                                model,
                                seed,
                                schedule_name=schedule_name,
                                buffer_policy=policy,
                                num_warmup_steps=num_warmup_steps,
                                num_draws=num_draws,
                            )
                            row.update(
                                {
                                    "schema_version": prov["schema_version"],
                                    "run_id": args.run_id,
                                    "cell_id": label,
                                }
                            )
                            output.emit(row)
                            for event in event_rows:
                                event.update(
                                    {
                                        "schema_version": prov["schema_version"],
                                        "run_id": args.run_id,
                                        "cell_id": label,
                                    }
                                )
                                events.emit(event)
                            print(
                                f"{label} epg={row['ess_per_grad']:.6g}",
                                flush=True,
                            )
                        except Exception as exc:
                            errors += 1
                            output.emit(
                                {
                                    "record_type": "cell",
                                    "schema_version": prov["schema_version"],
                                    "run_id": args.run_id,
                                    "cell_id": label,
                                    "model": model,
                                    "seed": seed,
                                    "schedule_family": schedule_name,
                                    "buffer_policy": policy,
                                    "error": repr(exc),
                                    "traceback_tail": traceback.format_exc()[-2000:],
                                }
                            )
                            print(f"{label} ERROR {exc!r}", flush=True)
        output.emit(
            {
                "record_type": "block_summary",
                "run_id": args.run_id,
                "expected_cells": expected,
                "error_cells": errors,
                "wall_seconds": time.monotonic() - started,
            }
        )

    manifest = {
        "schema_version": prov["schema_version"],
        "run_id": args.run_id,
        "expected_cells": expected,
        "error_cells": errors,
        "results": finish_output(args.out),
        "events": finish_output(events_path),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
