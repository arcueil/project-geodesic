#!/usr/bin/env python3
"""Run the Paper 1 fixed routing and NUTS efficiency suite.

Each invocation writes one immutable block.  This makes the multi-hour corpus
resumable without appending to or overwriting an accepted raw file.
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
    build_isotropic_cell,
    build_multi_chain_cell,
    finish_output,
    provenance,
    run_auto_multi_chain,
    run_manual_population,
    run_manual_single_chain,
)


BLOCKS = (
    "smoke",
    "illcond",
    "german",
    "radon-50k",
    "radon-400k",
    "refusals",
    "controls",
    "clones",
    "manual-illcond",
    "manual-german",
    "manual-population-illcond",
    "manual-population-german",
)


def _auto_plan(block: str) -> list[dict[str, Any]]:
    if block == "smoke":
        return [
            {
                "cell": build_isotropic_cell(0, d=4),
                "budget": 4_000,
                "draws": 20,
                "sample_all_chains": True,
            }
        ]
    if block == "illcond":
        return [
            {
                "cell": build_multi_chain_cell("ill_cond_50", seed),
                "budget": 50_000,
                "draws": 4_000,
                "sample_all_chains": True,
            }
            for seed in range(7)
        ]
    if block == "german":
        return [
            {
                "cell": build_multi_chain_cell("german_credit", seed),
                "budget": 50_000,
                "draws": 4_000,
                "sample_all_chains": True,
            }
            for seed in range(5)
        ]
    if block == "radon-50k":
        return [
            {
                "cell": build_multi_chain_cell("radon", seed),
                "budget": 50_000,
                "draws": 4_000,
            }
            for seed in range(2)
        ]
    if block == "radon-400k":
        return [
            {
                "cell": build_multi_chain_cell("radon", seed),
                "budget": 400_000,
                "draws": 4_000,
            }
            for seed in range(2)
        ]
    if block == "refusals":
        cells = [
            *(build_multi_chain_cell("neals_funnel", seed) for seed in range(3)),
            *(build_multi_chain_cell("banana", seed) for seed in range(2)),
        ]
        return [{"cell": cell, "budget": 50_000, "draws": 4_000} for cell in cells]
    if block == "controls":
        cells = [
            *(build_multi_chain_cell("stoch_vol", seed) for seed in range(2)),
            *(build_multi_chain_cell("mvn_10", seed) for seed in range(2)),
            build_isotropic_cell(0),
        ]
        return [{"cell": cell, "budget": 50_000, "draws": 4_000} for cell in cells]
    if block == "clones":
        return [
            {
                "cell": build_multi_chain_cell(
                    "ill_cond_50", seed, clone_radius=0.01
                ),
                "budget": 50_000,
                "draws": 4_000,
            }
            for seed in range(3)
        ]
    return []


def _manual_plan(block: str) -> list[dict[str, Any]]:
    if block in ("manual-illcond", "manual-population-illcond"):
        model, seeds, rank = "ill_cond_50", range(7), 25
    elif block in ("manual-german", "manual-population-german"):
        model, seeds, rank = "german_credit", range(5), 10
    else:
        return []
    policy = (
        "equal_split_aggregate_budget_control"
        if block.startswith("manual-population-")
        else "historical_single_chain_nominal_B"
    )
    plan: list[dict[str, Any]] = []
    for seed in seeds:
        cell = build_multi_chain_cell(model, seed)
        plan.extend(
            [
                {
                    "cell": cell,
                    "metric": "fisher_low_rank",
                    "max_rank": rank,
                    "comparison_policy": policy,
                },
                {
                    "cell": cell,
                    "metric": "welford_diag",
                    "max_rank": None,
                    "comparison_policy": policy,
                },
            ]
        )
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=BLOCKS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events-out", type=Path)
    parser.add_argument("--arrays-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development smoke only; release runs must use clean dependencies.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    events_path = args.events_out or args.out.with_suffix(".events.jsonl")
    arrays_dir = args.arrays_dir or args.out.with_suffix(".arrays")
    auto_plan = _auto_plan(args.block)
    manual_plan = _manual_plan(args.block)
    expected = len(auto_plan) + len(manual_plan)
    errors = 0
    started = time.monotonic()
    array_artifacts: list[dict[str, Any]] = []
    array_index = 0

    with ImmutableJsonl(args.out) as output, ImmutableJsonl(events_path) as events:
        arrays_dir.mkdir(parents=True, exist_ok=False)
        base_provenance = provenance(
            suite="fixed-suite",
            argv=sys.argv,
            allow_dirty=args.allow_dirty,
            run_id=args.run_id,
        )
        base_provenance["block"] = args.block
        output.emit(base_provenance)
        events.emit({**base_provenance, "suite": "fixed-suite-schedule-events"})

        for spec in auto_plan:
            cell = spec["cell"]
            label = f"{cell['model']}:seed={cell['seed']}:B={spec['budget']}"
            cell_started = time.monotonic()
            artifact_path = arrays_dir / f"cell-{array_index:04d}.npz"
            array_index += 1
            try:
                row, event_rows = run_auto_multi_chain(
                    cell,
                    max_grad_budget=spec["budget"],
                    num_draws=spec["draws"],
                    sample_all_chains=spec.get("sample_all_chains", False),
                    draws_path=artifact_path,
                )
                row.update(
                    {
                        "block": args.block,
                        "run_id": args.run_id,
                        "cell_id": label,
                        "wall_seconds": time.monotonic() - cell_started,
                    }
                )
                output.emit(row)
                if row["draws_artifact"] is not None:
                    array_artifacts.append(
                        {
                            "cell_id": label,
                            **row["draws_artifact"],
                        }
                    )
                for event in event_rows:
                    event.update(
                        {
                            "block": args.block,
                            "run_id": args.run_id,
                            "cell_id": label,
                        }
                    )
                    events.emit(event)
                print(
                    f"{label} route={row['route']} rank={row['effective_rank']} "
                    f"handoff={row['handoff']} epg={row['ess_per_grad_sc1']:.6g}",
                    flush=True,
                )
            except Exception as exc:  # the validator turns any error row into failure
                errors += 1
                output.emit(
                    {
                        "record_type": "cell",
                        "schema_version": base_provenance["schema_version"],
                        "block": args.block,
                        "run_id": args.run_id,
                        "cell_id": label,
                        "model": cell["model"],
                        "seed": int(cell["seed"]),
                        "max_grad_budget": spec["budget"],
                        "error": repr(exc),
                        "traceback_tail": traceback.format_exc()[-2000:],
                    }
                )
                print(f"{label} ERROR {exc!r}", flush=True)

        for spec in manual_plan:
            cell = spec["cell"]
            label = (
                f"{cell['model']}:seed={cell['seed']}:metric={spec['metric']}"
            )
            cell_started = time.monotonic()
            artifact_path = arrays_dir / f"cell-{array_index:04d}.npz"
            array_index += 1
            try:
                if (
                    spec["comparison_policy"]
                    == "equal_split_aggregate_budget_control"
                ):
                    row = run_manual_population(
                        cell,
                        metric=spec["metric"],
                        max_rank=spec["max_rank"],
                        nominal_max_grad_budget=50_000,
                        num_warmup_steps_per_chain=312,
                        num_draws=4_000,
                        draws_path=artifact_path,
                    )
                else:
                    row = run_manual_single_chain(
                        cell,
                        metric=spec["metric"],
                        max_rank=spec["max_rank"],
                        num_warmup_steps=2_500,
                        num_draws=4_000,
                        draws_path=artifact_path,
                    )
                    row.update(
                        {
                            "comparison_policy": (
                                "historical_single_chain_nominal_B"
                            ),
                            "metric_analysis_role": (
                                "preregistered_primary_manual_metric"
                                if spec["metric"] == "fisher_low_rank"
                                else "preregistered_diagonal_control"
                            ),
                            "nominal_max_grad_budget": 50_000,
                            "budget_enforcement": (
                                "same_preregistered_nominal_allocation_"
                                "not_realized_cap"
                            ),
                            "warmup_step_conversion_assumed_"
                            "integration_steps": 20,
                            "num_warmup_steps_per_chain": 2_500,
                            "nominal_unused_gradient_remainder": 0,
                        }
                    )
                row.update(
                    {
                        "block": args.block,
                        "run_id": args.run_id,
                        "cell_id": label,
                        "wall_seconds": time.monotonic() - cell_started,
                    }
                )
                output.emit(row)
                if row["draws_artifact"] is not None:
                    array_artifacts.append(
                        {
                            "cell_id": label,
                            **row["draws_artifact"],
                        }
                    )
                epg = row.get(
                    "ess_per_grad_pooled_population",
                    row.get("ess_per_grad_sc1"),
                )
                print(
                    f"{label} policy={spec['comparison_policy']} "
                    f"epg={epg:.6g}",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                output.emit(
                    {
                        "record_type": "cell",
                        "schema_version": base_provenance["schema_version"],
                        "block": args.block,
                        "run_id": args.run_id,
                        "cell_id": label,
                        "model": cell["model"],
                        "seed": int(cell["seed"]),
                        "metric": spec["metric"],
                        "error": repr(exc),
                        "traceback_tail": traceback.format_exc()[-2000:],
                    }
                )
                print(f"{label} ERROR {exc!r}", flush=True)

        output.emit(
            {
                "record_type": "block_summary",
                "block": args.block,
                "run_id": args.run_id,
                "expected_cells": expected,
                "completed_cells": expected,
                "error_cells": errors,
                "wall_seconds": time.monotonic() - started,
            }
        )

    manifest = {
        "schema_version": base_provenance["schema_version"],
        "block": args.block,
        "run_id": args.run_id,
        "results": finish_output(args.out),
        "events": finish_output(events_path),
        "arrays_dir": arrays_dir.name,
        "array_artifacts": array_artifacts,
        "expected_cells": expected,
        "error_cells": errors,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
