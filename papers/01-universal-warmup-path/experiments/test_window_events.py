"""Tests for the stored window-event schema and schedule-only figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_schedule_evidence import render
from window_events import (
    FISHER_HMC_SOURCE,
    build_metric_window_events,
    schedule_prescriptions,
    validate_window_events,
)


def test_schedule_prescriptions_keep_boundaries_static() -> None:
    schedules = schedule_prescriptions(312, n_chains=8, dimension=5)

    assert schedules["stan"]["metric_windows"]
    assert schedules["proportional_growing"]["metric_windows"]
    assert [
        (
            segment["stage"],
            segment["segment_start_step"],
            segment["segment_end_step"],
            segment["metric_adaptation"],
        )
        for segment in schedules["stan"]["stage_segments"]
    ] == [
        ("fast", 1, 75, False),
        ("slow", 76, 262, True),
        ("fast", 263, 312, False),
    ]
    fisher = schedules["seyboldt_fisher_hmc"]
    assert fisher["source"] == FISHER_HMC_SOURCE
    assert "not a generic schedule" in fisher["scope"]
    assert fisher["source_sections"] == ["3.2", "3.3"]
    assert (
        fisher["online_memory_window_rule"]
        == "a_n=max(0,L(floor(n/L)-1))"
    )
    assert (
        fisher["low_rank_update_rule"]
        == "update the metric at global multiples of L"
    )
    assert [
        (
            phase["phase_start_step"],
            phase["phase_end_step"],
            phase["refresh_or_memory_period"],
            phase["metric_adaptation"],
            phase["step_size_reinitialized_at_start"],
            phase["first_global_refresh_step"],
        )
        for phase in fisher["phases"]
    ] == [
        (1, 93, 10, True, False, 10),
        (94, 265, 80, True, True, 160),
        (266, 312, None, False, False, None),
    ]
    assert [
        (
            refresh["phase_index"],
            refresh["warmup_step"],
            refresh["period"],
        )
        for refresh in fisher["refresh_events"]
    ] == [
        *((1, step, 10) for step in range(10, 94, 10)),
        *((2, step, 80) for step in range(160, 266, 80)),
    ]
    assert all(
        refresh["source"] == FISHER_HMC_SOURCE
        for refresh in fisher["refresh_events"]
    )
    actual = schedules["controller_actual"]
    assert actual["metric_windows"]
    assert [
        (
            segment["stage"],
            segment["segment_start_step"],
            segment["segment_end_step"],
        )
        for segment in actual["stage_segments"]
    ] == [
        ("slow", 1, 265),
        ("fast", 266, 312),
    ]
    assert actual["boundary_policy"] == "prescribed_static"
    assert actual["evidence_evaluated_at_scheduled_boundaries"] is True
    assert actual["boundaries_selected_from_evidence"] is False


def _event(step: int, route: str, *, stable: bool = False) -> dict:
    return {
        "event_schema_version": 1,
        "window_index": step,
        "window_start_step": step * 10 + 1,
        "window_end_step": (step + 1) * 10,
        "event": "scheduled_metric_window_end",
        "boundary_policy": "prescribed_static",
        "boundary_selected_from_evidence": False,
        "cumulative_gradients": (step + 1) * 80,
        "cumulative_gradient_basis": "sum_num_integration_steps",
        "global_exploration": "not_established",
        "route": route,
        "handoff": "none",
        "airm_stable": stable,
    }


def test_event_validator_rejects_evidence_selected_boundary() -> None:
    events = [_event(0, "diagonal")]
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    events[0]["boundary_selected_from_evidence"] = True

    errors = validate_window_events(events, schedules)

    assert any("evidence-selected" in error for error in errors)


def test_event_validator_rejects_unattributed_fisher_schedule() -> None:
    events = [_event(0, "diagonal")]
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    schedules["seyboldt_fisher_hmc"]["source"] = "unknown"

    errors = validate_window_events(events, schedules)

    assert any("unknown source" in error for error in errors)


def test_event_validator_rejects_metric_window_in_fast_stage() -> None:
    events = [_event(0, "diagonal")]
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    schedules["stan"]["metric_windows"][0]["window_start_step"] = 1

    errors = validate_window_events(events, schedules)

    assert any("outside a slow stage" in error for error in errors)


def test_event_validator_rejects_incorrect_fisher_refresh_alignment() -> None:
    events = [_event(0, "diagonal")]
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    schedules["seyboldt_fisher_hmc"]["refresh_events"][0][
        "global_multiple_index"
    ] = 99

    errors = validate_window_events(events, schedules)

    assert any("invalid alignment metadata" in error for error in errors)


def test_short_support_aware_schedule_allows_no_metric_windows() -> None:
    schedules = schedule_prescriptions(3, n_chains=8, dimension=50)
    assert schedules["controller_actual"]["metric_windows"] == []

    events = build_metric_window_events(
        prescriptions=schedules,
        integration_steps=np.ones((3, 8), dtype=int),
        step_sizes=np.ones(3),
        controller_budget_trace=np.arange(1, 4),
        evidence_at_boundary=lambda *_args: (_ for _ in ()).throw(
            AssertionError("no boundary should be evaluated")
        ),
    )

    assert events == []


def test_schedule_figure_reads_only_stored_events(tmp_path: Path) -> None:
    schedules = schedule_prescriptions(100, n_chains=8, dimension=5)
    row = {
        "arm_id": "synthetic",
        "SR": 6.0,
        "seed": 42,
        "schedule_prescriptions": schedules,
        "window_events": [
            _event(0, "diagonal"),
            _event(1, "low_rank"),
            _event(2, "low_rank", stable=True),
        ],
    }
    output = tmp_path / "schedule.pdf"

    render(row, output)

    assert output.is_file()
    assert output.stat().st_size > 0
