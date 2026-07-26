"""Unit tests for the canonical GMM grid and strict artifact validator."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gmm_suite import (
    ValidationError,
    arm_specs,
    full_arm_specs,
    validate_arm_file,
)
from window_events import schedule_prescriptions


def test_full_grid_matches_the_public_gmm_scope() -> None:
    specs = full_arm_specs()

    assert sum(spec.expected_rows for spec in specs) == 239
    assert {spec.correlated_axes for spec in specs} == {2, 3}
    assert all("gmm25" not in spec.arm_id for spec in specs)
    assert len(arm_specs(smoke=True)) == len(specs)
    assert sum(spec.expected_rows for spec in arm_specs(smoke=True)) == 9


def _valid_event() -> dict:
    return {
        "event_schema_version": 1,
        "window_index": 0,
        "window_start_step": 1,
        "window_end_step": 10,
        "event": "scheduled_metric_window_end",
        "boundary_policy": "prescribed_static",
        "boundary_selected_from_evidence": False,
        "cumulative_gradients": 80,
        "cumulative_gradient_basis": "sum_num_integration_steps",
        "controller_budget_used": 80,
        "step_size": 0.1,
        "route": "low_rank",
        "effective_rank": 1,
        "metric_route_status": "historical_gate_pass",
        "metric_route_basis": "pooled_within",
        "metric_scope": "within_chain_conditional",
        "observed_ensemble_evidence": "no_disagreement_detected",
        "global_exploration": "not_established",
        "handoff": "none",
        "confidence": "high",
        "airm_stable": False,
        "converged_at_controller_budget": -1,
        "airm_velocity_previous": None,
        "airm_velocity_current": 0.2,
        "r2_latest": 0.8,
    }


def _valid_schedules() -> dict:
    window = {
        "window_index": 0,
        "window_start_step": 1,
        "window_end_step": 10,
    }
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    schedules["controller_actual"]["metric_windows"] = [window]
    return schedules


def _primary_row(spec) -> dict:
    return {
        "arm_id": spec.arm_id,
        "SR": spec.srs[0],
        "seed": spec.seeds[0],
        "init_kind": spec.init_kind,
        "budget": spec.budget,
        "correlated_axes": spec.correlated_axes,
        "M": spec.chains,
        "n_sample_draws": spec.draws,
        "route": "low_rank",
        "effective_rank": 1,
        "deferred_to_ensemble": False,
        "detection_branch": "pooled_within",
        "within_lam1": 1.5,
        "chain_consistency_psi": 0.8,
        "chain_collinearity": 0.9,
        "unimodality_gate": "pass",
        "metric_route_status": "historical_gate_pass",
        "metric_route_basis": "pooled_within",
        "metric_scope": "within_chain_conditional",
        "observed_ensemble_evidence": "no_disagreement_detected",
        "global_exploration": "not_established",
        "handoff": "none",
        "confidence_scope": "historical_route_selection_heuristic",
        "projected_within_variance": 20.0,
        "projected_between_variance": 30.0,
        "projected_total_variance": 21.0,
        "projected_target_variance": 22.0,
        "projected_partition_error": 0.0,
        "projected_total_reference_error": -1.0,
        "min_ess_per_grad": 0.01,
        "split_rhat": 1.001,
        "mode_weight_est": 0.7,
        "both_modes_visited_frac": 1.0,
        "num_divergences": 0,
        "window_events": [_valid_event()],
        "schedule_prescriptions": _valid_schedules(),
        "error": None,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validator_accepts_one_complete_primary_cell(tmp_path: Path) -> None:
    spec = replace(
        full_arm_specs()[0],
        srs=(1.5,),
        seeds=(42,),
        draws=20,
    )
    path = tmp_path / spec.filename
    _write_rows(path, [_primary_row(spec)])

    summary = validate_arm_file(path, spec)

    assert summary["status"] == "valid"
    assert summary["rows"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "error", "nonfinite", "missing_cell", "bad_invariant"],
)
def test_validator_rejects_corrupt_primary_output(
    mutation: str,
    tmp_path: Path,
) -> None:
    spec = replace(
        full_arm_specs()[0],
        srs=(1.5,),
        seeds=(42,),
        draws=20,
    )
    row = _primary_row(spec)
    rows = [row]
    if mutation == "duplicate":
        rows.append(dict(row))
    elif mutation == "error":
        row["error"] = "synthetic failure"
    elif mutation == "nonfinite":
        row["split_rhat"] = float("nan")
    elif mutation == "missing_cell":
        rows.clear()
    else:
        row["projected_partition_error"] = 1e-3
    path = tmp_path / spec.filename
    _write_rows(path, rows)

    with pytest.raises(ValidationError):
        validate_arm_file(path, spec)
