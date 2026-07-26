#!/usr/bin/env python3
"""Run one side of the historical/current shipped-controller comparison.

Run this file twice in separate Python processes:

1. with BlackJAX 2f629218... importable and ``--revision historical``;
2. with BlackJAX 29d246885... importable and ``--revision current``.

The revisions differ in the shared-step-size mean-pooling correction and the
warmup-only NUTS depth cap (as well as later verdict semantics).  This is
therefore a shipped-bundle regression, not a pure one-line causal ablation.
The paired summarizer joins cells by model and seed.  Keeping revisions in
separate processes prevents Python module-cache contamination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

# A raw historical BlackJAX Git checkout does not contain setuptools-scm's
# generated ``blackjax/_version.py``.  Supply only that generated module in
# memory; the package source remains clean and its Git SHA remains authoritative.
for _entry in sys.path:
    _package_dir = Path(_entry) / "blackjax"
    if _package_dir.is_dir():
        if not (_package_dir / "_version.py").exists():
            _version_module = types.ModuleType("blackjax._version")
            _version_module.__version__ = "source-checkout"
            sys.modules["blackjax._version"] = _version_module
        break

from suite_common import (
    CURRENT_BLACKJAX_SHA,
    HISTORICAL_SEQUENTIAL_SHA,
    ImmutableJsonl,
    build_multi_chain_cell,
    finish_output,
    provenance,
    sample_single_chain,
)

import blackjax
import numpy as np
from blackjax.adaptation.meta._calibration import _ASSUMED_AVG_LEAPFROGS_PER_STEP
from blackjax.adaptation.staged_adaptation import staged_adaptation


def info_fn(_state: Any, info: Any, adaptation_state: Any) -> tuple:
    core = adaptation_state.imm_state
    return (
        info.num_integration_steps,
        info.is_divergent,
        info.acceptance_rate,
        adaptation_state.step_size,
        core.has_escalated,
        core.escalation_rank,
    )


def run_cell(
    model: str,
    seed: int,
    *,
    budget: int,
    draws: int,
    revision_label: str,
) -> dict[str, Any]:
    cell = build_multi_chain_cell(model, seed)
    n_chains = int(cell["n_chains"])
    num_steps = max(
        budget // (_ASSUMED_AVG_LEAPFROGS_PER_STEP * n_chains),
        1,
    )
    warmup = staged_adaptation(
        blackjax.nuts,
        cell["logdensity_fn"],
        metric="auto",
        max_grad_budget=budget,
        n_chains=n_chains,
        adaptation_info_fn=info_fn,
    )
    started = time.monotonic()
    result, trace = warmup.run(
        cell["warmup_key"],
        cell["inits"],
        num_steps,
    )
    wall_seconds = time.monotonic() - started
    nis, divergences, acceptance, step_sizes, escalated, nominal_rank = trace
    sampled = sample_single_chain(
        blackjax.nuts,
        cell["logdensity_fn"],
        result.parameters,
        np.asarray(result.state.position)[0],
        cell["sample_key"],
        num_draws=draws,
    )
    nis_array = np.asarray(nis)
    step_array = np.asarray(step_sizes)
    tail_start = max(num_steps // 2, num_steps - max(50, num_steps // 5))
    log_tail = np.log(step_array[tail_start:])
    equilibrium_step = float(np.exp(np.median(log_tail)))
    shipped_step = float(np.asarray(result.parameters["step_size"]))
    warmup_grads = int(nis_array.sum())
    total_grads = warmup_grads / n_chains + sampled["sampling_grads"]
    imm = result.parameters["inverse_mass_matrix"]
    lam = np.asarray(imm.lam)
    return {
        "record_type": "cell",
        "model": model,
        "seed": seed,
        "revision_arm": revision_label,
        "shipped_controller_bundle": (
            "historical_sequential_acceptance_uncapped"
            if revision_label == "historical"
            else "current_mean_pooled_acceptance_warmup_depth_capped"
        ),
        "comparison_scope": (
            "shipped_bundle_regression_mean_pooling_plus_warmup_depth_cap"
        ),
        "n_chains": n_chains,
        "max_grad_budget": budget,
        "num_warmup_steps_per_chain": num_steps,
        "num_sampling_draws": draws,
        "route": "low_rank" if bool(np.asarray(escalated).any()) else "diagonal",
        "nominal_rank": int(np.asarray(nominal_rank)[-1]),
        "effective_rank": int(np.sum(np.abs(lam - 1.0) > 1e-6)),
        "warmup_grads_all_chains": warmup_grads,
        "warmup_grads_sc1_charge": warmup_grads / n_chains,
        "warmup_divergences": int(np.asarray(divergences).sum()),
        "sampling_grads": sampled["sampling_grads"],
        "sampling_divergences": sampled["sampling_divergences"],
        "min_bulk_ess": sampled["min_bulk_ess"],
        "ess_per_grad_sc1": sampled["min_bulk_ess"] / total_grads,
        "shipped_step_size": shipped_step,
        "equilibrium_step_size_proxy": equilibrium_step,
        "shipped_to_equilibrium_ratio": shipped_step / equilibrium_step,
        "settled_log_amplitude_natural": float(np.ptp(log_tail)),
        "settled_log_amplitude_log10": float(np.ptp(log_tail) / np.log(10.0)),
        "mean_acceptance_tail": float(np.mean(np.asarray(acceptance)[tail_start:])),
        "warmup_wall_seconds": wall_seconds,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        choices=("historical", "current"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_sha = (
        HISTORICAL_SEQUENTIAL_SHA
        if args.revision == "historical"
        else CURRENT_BLACKJAX_SHA
    )
    plan = (
        [("ill_cond_50", 0)]
        if args.smoke
        else [
            *(("ill_cond_50", seed) for seed in range(7)),
            *(("german_credit", seed) for seed in range(5)),
        ]
    )
    budget = 4_000 if args.smoke else 50_000
    draws = 20 if args.smoke else 2_000
    errors = 0
    started = time.monotonic()

    with ImmutableJsonl(args.out) as output:
        prov = provenance(
            suite="shared-step-size",
            argv=sys.argv,
            expected_blackjax_sha=expected_sha,
            allow_dirty=args.allow_dirty,
            run_id=args.run_id,
        )
        prov["revision_arm"] = args.revision
        output.emit(prov)
        for model, seed in plan:
            label = f"{model}:seed={seed}:revision={args.revision}"
            try:
                row = run_cell(
                    model,
                    seed,
                    budget=budget,
                    draws=draws,
                    revision_label=args.revision,
                )
                row.update(
                    {
                        "schema_version": prov["schema_version"],
                        "run_id": args.run_id,
                        "cell_id": label,
                    }
                )
                output.emit(row)
                print(
                    f"{label} wg={row['warmup_grads_all_chains']} "
                    f"amp10={row['settled_log_amplitude_log10']:.2f} "
                    f"epg={row['ess_per_grad_sc1']:.6g}",
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
                        "revision_arm": args.revision,
                        "error": repr(exc),
                        "traceback_tail": traceback.format_exc()[-2000:],
                    }
                )
                print(f"{label} ERROR {exc!r}", flush=True)
        output.emit(
            {
                "record_type": "block_summary",
                "run_id": args.run_id,
                "revision_arm": args.revision,
                "expected_cells": len(plan),
                "error_cells": errors,
                "wall_seconds": time.monotonic() - started,
            }
        )

    manifest = {
        "schema_version": prov["schema_version"],
        "run_id": args.run_id,
        "revision_arm": args.revision,
        "expected_cells": len(plan),
        "error_cells": errors,
        "results": finish_output(args.out),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
