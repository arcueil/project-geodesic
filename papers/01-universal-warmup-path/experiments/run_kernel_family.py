#!/usr/bin/env python3
"""Run the fixed-length HMC-family validation for Paper 1.

For each model/seed, a standard NUTS warmup calibrates one trajectory length.
That frozen length is then shared by HMC and multinomial HMC, and by their
automatic/manual metric arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from suite_common import (
    ImmutableJsonl,
    build_multi_chain_cell,
    finish_output,
    provenance,
    run_auto_multi_chain,
    run_manual_single_chain,
)

import blackjax
import numpy as np
from blackjax.adaptation.low_rank_adaptation import build_growing_window_schedule
from blackjax.adaptation.staged_adaptation import staged_adaptation


def calibrate_length(
    cell: dict[str, Any],
    *,
    num_warmup_steps: int,
) -> dict[str, Any]:
    def info_fn(_state: Any, info: Any, adaptation_state: Any) -> tuple:
        return (
            info.num_integration_steps,
            info.is_divergent,
            adaptation_state.step_size,
        )

    warmup = staged_adaptation(
        blackjax.nuts,
        cell["logdensity_fn"],
        metric="welford_diag",
        schedule_fn=build_growing_window_schedule,
        adaptation_info_fn=info_fn,
    )
    result, trace = warmup.run(
        cell["warmup_key"],
        np.asarray(cell["inits"])[0],
        num_warmup_steps,
    )
    integration_steps, divergences, step_sizes = trace
    integration_steps = np.asarray(integration_steps)
    tail_start = max(num_warmup_steps // 2, num_warmup_steps - 500)
    length = max(int(np.rint(np.median(integration_steps[tail_start:]))), 1)
    return {
        "record_type": "trajectory_calibration",
        "model": cell["model"],
        "seed": int(cell["seed"]),
        "warmup": "nuts_welford_diag_proportional_growing",
        "num_warmup_steps": num_warmup_steps,
        "tail_start": tail_start,
        "num_integration_steps": length,
        "tail_nis_median": float(np.median(integration_steps[tail_start:])),
        "tail_nis_min": int(np.min(integration_steps[tail_start:])),
        "tail_nis_max": int(np.max(integration_steps[tail_start:])),
        "warmup_divergences": int(np.asarray(divergences).sum()),
        "calibrated_step_size": float(np.asarray(result.parameters["step_size"])),
        "tail_log_step_size_amplitude": float(
            np.ptp(np.log(np.asarray(step_sizes)[tail_start:]))
        ),
    }


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
    seeds = [0] if args.smoke else [0, 1, 2]
    models = ["ill_cond_50"] if args.smoke else ["ill_cond_50", "german_credit"]
    algorithms = [("hmc", blackjax.hmc)] if args.smoke else [
        ("hmc", blackjax.hmc),
        ("multinomial_hmc", blackjax.mhmc),
    ]
    calibration_steps = 100 if args.smoke else 1_000
    budget = 4_000 if args.smoke else 50_000
    draws = 20 if args.smoke else 4_000
    expected_route_cells = len(models) * len(seeds) * len(algorithms)
    errors = 0
    started = time.monotonic()

    with ImmutableJsonl(args.out) as output, ImmutableJsonl(events_path) as events:
        prov = provenance(
            suite="kernel-family",
            argv=sys.argv,
            allow_dirty=args.allow_dirty,
            run_id=args.run_id,
        )
        output.emit(prov)
        events.emit({**prov, "suite": "kernel-family-schedule-events"})

        for model in models:
            rank = 25 if model == "ill_cond_50" else 10
            for seed in seeds:
                cell = build_multi_chain_cell(model, seed)
                try:
                    calibration = calibrate_length(
                        cell,
                        num_warmup_steps=calibration_steps,
                    )
                    calibration.update(
                        {
                            "run_id": args.run_id,
                            "schema_version": prov["schema_version"],
                        }
                    )
                    output.emit(calibration)
                    length = int(calibration["num_integration_steps"])
                except Exception as exc:
                    errors += len(algorithms)
                    output.emit(
                        {
                            "record_type": "trajectory_calibration",
                            "schema_version": prov["schema_version"],
                            "run_id": args.run_id,
                            "model": model,
                            "seed": seed,
                            "error": repr(exc),
                            "traceback_tail": traceback.format_exc()[-2000:],
                        }
                    )
                    continue

                for algorithm_name, algorithm in algorithms:
                    label = (
                        f"{model}:seed={seed}:algorithm={algorithm_name}:L={length}"
                    )
                    try:
                        auto, event_rows = run_auto_multi_chain(
                            cell,
                            max_grad_budget=budget,
                            num_draws=draws,
                            algorithm=algorithm,
                            algorithm_name=algorithm_name,
                            extra_parameters={"num_integration_steps": length},
                        )
                        auto.update(
                            {
                                "run_id": args.run_id,
                                "cell_id": label,
                                "arm": "automatic",
                                "trajectory_calibration": (
                                    f"{model}:seed={seed}:nuts"
                                ),
                            }
                        )
                        output.emit(auto)
                        for event in event_rows:
                            event.update(
                                {
                                    "run_id": args.run_id,
                                    "cell_id": label,
                                    "algorithm": algorithm_name,
                                }
                            )
                            events.emit(event)

                        # Equal one-chain share of the automatic M-chain budget.
                        manual_steps = int(auto["num_warmup_steps_per_chain"])
                        manual = run_manual_single_chain(
                            cell,
                            metric="fisher_low_rank",
                            max_rank=rank,
                            num_warmup_steps=manual_steps,
                            num_draws=draws,
                            algorithm=algorithm,
                            algorithm_name=algorithm_name,
                            extra_parameters={"num_integration_steps": length},
                        )
                        manual.update(
                            {
                                "run_id": args.run_id,
                                "cell_id": label,
                                "arm": "matched_manual",
                                "trajectory_calibration": (
                                    f"{model}:seed={seed}:nuts"
                                ),
                                "automatic_ess_per_grad_sc1": auto[
                                    "ess_per_grad_sc1"
                                ],
                                "automatic_to_manual_ratio": (
                                    auto["ess_per_grad_sc1"]
                                    / manual["ess_per_grad_sc1"]
                                ),
                            }
                        )
                        output.emit(manual)
                        print(
                            f"{label} route={auto['route']} "
                            f"ratio={manual['automatic_to_manual_ratio']:.3f} "
                            f"div={auto['sampling_divergences']}",
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
                                "algorithm": algorithm_name,
                                "error": repr(exc),
                                "traceback_tail": traceback.format_exc()[-2000:],
                            }
                        )
                        print(f"{label} ERROR {exc!r}", flush=True)

        output.emit(
            {
                "record_type": "block_summary",
                "run_id": args.run_id,
                "expected_route_cells": expected_route_cells,
                "error_cells": errors,
                "wall_seconds": time.monotonic() - started,
            }
        )

    manifest = {
        "schema_version": prov["schema_version"],
        "run_id": args.run_id,
        "expected_route_cells": expected_route_cells,
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
