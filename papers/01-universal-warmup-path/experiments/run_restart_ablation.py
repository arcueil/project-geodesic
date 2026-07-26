#!/usr/bin/env python3
"""Compare per-window dual-averaging reseed with continuous dual averaging.

The current public engine always reseeds dual averaging after a slow window.
For this isolated experiment, a local host factory changes exactly that one
line while leaving the sampler, metric core, schedule, initial states, and
random keys unchanged.  The intervention is confined to engine construction;
BlackJAX source is never modified.
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
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np
import blackjax.adaptation.staged_adaptation as staged_module
from blackjax.adaptation.staged_adaptation import StagedAdaptationState
from blackjax.adaptation.step_size import dual_averaging_adaptation


def make_engine_with_restart_policy(
    metric_core: Any,
    *,
    target_acceptance_rate: float,
    n_da_updates: int = 1,
    reseed_at_window: bool,
) -> tuple[Callable, Callable, Callable]:
    """Copy of the public host engine with one explicit restart-policy factor."""

    da_init, da_update, da_final = dual_averaging_adaptation(
        target_acceptance_rate
    )

    def update_da(state: Any, acceptance: Any) -> Any:
        if n_da_updates == 1:
            return da_update(state, acceptance)
        return da_update(state, jnp.mean(acceptance))

    def init(position: Any, initial_step_size: float) -> StagedAdaptationState:
        n_dims = fu.ravel_pytree(position)[0].shape[0]
        metric_state = metric_core.init(n_dims)
        return StagedAdaptationState(
            da_init(initial_step_size),
            metric_state,
            initial_step_size,
            metric_state.inverse_mass_matrix,
        )

    def fast_update(
        _position: Any,
        _grad: Any,
        acceptance: Any,
        state: StagedAdaptationState,
    ) -> StagedAdaptationState:
        new_da = update_da(state.ss_state, acceptance)
        return state._replace(
            ss_state=new_da,
            step_size=jnp.exp(new_da.log_step_size),
        )

    def slow_update(
        position: Any,
        grad: Any,
        acceptance: Any,
        state: StagedAdaptationState,
    ) -> StagedAdaptationState:
        new_metric = metric_core.update(state.imm_state, position, grad)
        new_da = update_da(state.ss_state, acceptance)
        return StagedAdaptationState(
            new_da,
            new_metric,
            jnp.exp(new_da.log_step_size),
            new_metric.inverse_mass_matrix,
        )

    def slow_final(state: StagedAdaptationState) -> StagedAdaptationState:
        new_metric = metric_core.final(state.imm_state)
        if reseed_at_window:
            new_da = da_init(da_final(state.ss_state))
        else:
            new_da = state.ss_state
        return StagedAdaptationState(
            new_da,
            new_metric,
            jnp.exp(new_da.log_step_size),
            new_metric.inverse_mass_matrix,
        )

    def update(
        state: StagedAdaptationState,
        adaptation_stage: Any,
        position: Any,
        grad: Any,
        acceptance: Any,
    ) -> StagedAdaptationState:
        stage, is_window_end = adaptation_stage
        state = jax.lax.switch(
            stage,
            (fast_update, slow_update),
            position,
            grad,
            acceptance,
            state,
        )
        return jax.lax.cond(
            is_window_end,
            slow_final,
            lambda value: value,
            state,
        )

    def final(state: StagedAdaptationState) -> tuple[Any, Any]:
        return (
            jnp.exp(state.ss_state.log_step_size_avg),
            state.imm_state.inverse_mass_matrix,
        )

    return init, update, final


def build_warmup(
    logdensity_fn: Callable,
    *,
    budget: int,
    n_chains: int,
    reseed_at_window: bool,
    info_fn: Callable,
) -> Any:
    original = staged_module._make_engine

    def factory(
        metric_core: Any,
        *,
        target_acceptance_rate: float,
        n_da_updates: int = 1,
    ) -> tuple[Callable, Callable, Callable]:
        return make_engine_with_restart_policy(
            metric_core,
            target_acceptance_rate=target_acceptance_rate,
            n_da_updates=n_da_updates,
            reseed_at_window=reseed_at_window,
        )

    try:
        staged_module._make_engine = factory
        return staged_module.staged_adaptation(
            blackjax.nuts,
            logdensity_fn,
            metric="auto",
            max_grad_budget=budget,
            n_chains=n_chains,
            adaptation_info_fn=info_fn,
        )
    finally:
        staged_module._make_engine = original


def info_fn(_state: Any, info: Any, adaptation_state: Any) -> tuple:
    core = adaptation_state.imm_state
    return (
        info.num_integration_steps,
        info.is_divergent,
        adaptation_state.step_size,
        core.has_escalated,
        core.escalation_rank,
    )


def run_cell(
    model: str,
    seed: int,
    *,
    reseed_at_window: bool,
    budget: int,
    num_draws: int,
) -> dict[str, Any]:
    cell = build_multi_chain_cell(model, seed)
    n_chains = int(cell["n_chains"])
    num_warmup_steps = max(budget // (20 * n_chains), 1)
    warmup = build_warmup(
        cell["logdensity_fn"],
        budget=budget,
        n_chains=n_chains,
        reseed_at_window=reseed_at_window,
        info_fn=info_fn,
    )
    started = time.monotonic()
    result, trace = warmup.run(
        cell["warmup_key"],
        cell["inits"],
        num_warmup_steps,
    )
    wall_seconds = time.monotonic() - started
    nis, divergence_trace, step_size_trace, escalated, nominal_rank = trace
    sampled = sample_single_chain(
        blackjax.nuts,
        cell["logdensity_fn"],
        result.parameters,
        np.asarray(result.state.position)[0],
        cell["sample_key"],
        num_draws=num_draws,
    )
    warmup_grads = int(np.asarray(nis).sum())
    sc1_total = warmup_grads / n_chains + sampled["sampling_grads"]
    lam = np.asarray(result.parameters["inverse_mass_matrix"].lam)
    return {
        "record_type": "cell",
        "model": model,
        "seed": seed,
        "restart_policy": (
            "per_window_reseed" if reseed_at_window else "continuous"
        ),
        "n_chains": n_chains,
        "max_grad_budget": budget,
        "num_warmup_steps_per_chain": num_warmup_steps,
        "num_sampling_draws": num_draws,
        "route": "low_rank" if bool(np.asarray(escalated).any()) else "diagonal",
        "nominal_rank": int(np.asarray(nominal_rank)[-1]),
        "effective_rank": int(np.sum(np.abs(lam - 1.0) > 1e-6)),
        "warmup_grads_all_chains": warmup_grads,
        "warmup_grads_sc1_charge": warmup_grads / n_chains,
        "warmup_divergences": int(np.asarray(divergence_trace).sum()),
        "sampling_grads": sampled["sampling_grads"],
        "sampling_divergences": sampled["sampling_divergences"],
        "min_bulk_ess": sampled["min_bulk_ess"],
        "ess_per_grad_sc1": sampled["min_bulk_ess"] / sc1_total,
        "step_size": float(np.asarray(result.parameters["step_size"])),
        "settled_log_step_size_amplitude": float(
            np.ptp(np.log(np.asarray(step_size_trace)[-100:]))
        ),
        "warmup_wall_seconds": wall_seconds,
        "intervention": (
            "At each slow-window boundary, carry the existing dual-averaging "
            "state instead of da_init(da_final(state)); all other code is shared."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = ["ill_cond_50"] if args.smoke else ["ill_cond_50", "german_credit"]
    seeds = [0] if args.smoke else [0, 1, 2]
    policies = [True, False]
    budget = 4_000 if args.smoke else 50_000
    draws = 20 if args.smoke else 4_000
    expected = len(models) * len(seeds) * len(policies)
    errors = 0
    started = time.monotonic()

    with ImmutableJsonl(args.out) as output:
        prov = provenance(
            suite="restart-ablation",
            argv=sys.argv,
            allow_dirty=args.allow_dirty,
            run_id=args.run_id,
        )
        output.emit(prov)
        for model in models:
            for seed in seeds:
                for reseed in policies:
                    policy = "per_window_reseed" if reseed else "continuous"
                    label = f"{model}:seed={seed}:restart={policy}"
                    try:
                        row = run_cell(
                            model,
                            seed,
                            reseed_at_window=reseed,
                            budget=budget,
                            num_draws=draws,
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
                                "restart_policy": policy,
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
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
