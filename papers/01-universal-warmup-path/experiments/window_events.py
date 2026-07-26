"""Recoverable event schema for scheduled warmup-window decisions."""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import jax
import numpy as np


EVENT_SCHEMA_VERSION = 1
FISHER_HMC_SOURCE = "arXiv:2603.18845"


def _slow_windows(schedule: Any) -> list[dict[str, int]]:
    """Return one-based inclusive metric-window boundaries from a schedule."""
    array = np.asarray(schedule)
    windows: list[dict[str, int]] = []
    start: int | None = None
    for index, (stage, is_end) in enumerate(array):
        if int(stage) == 1 and start is None:
            start = index
        if int(stage) == 1 and bool(is_end):
            assert start is not None
            windows.append(
                {
                    "window_index": len(windows),
                    "window_start_step": start + 1,
                    "window_end_step": index + 1,
                }
            )
            start = index + 1
        elif int(stage) != 1:
            start = None
    return windows


def _stage_segments(schedule: Any) -> list[dict[str, Any]]:
    """Return one-based inclusive fast/slow runs from a schedule."""
    array = np.asarray(schedule)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 1:
        raise ValueError("schedule must be a nonempty two-dimensional array")

    segments: list[dict[str, Any]] = []
    start = 0
    stage = int(array[0, 0])
    if stage not in (0, 1):
        raise ValueError(f"unknown adaptation stage {stage}")
    for index in range(1, array.shape[0]):
        next_stage = int(array[index, 0])
        if next_stage not in (0, 1):
            raise ValueError(f"unknown adaptation stage {next_stage}")
        if next_stage == stage:
            continue
        segments.append(
            {
                "segment_index": len(segments),
                "stage": "slow" if stage == 1 else "fast",
                "segment_start_step": start + 1,
                "segment_end_step": index,
                "metric_adaptation": stage == 1,
                "step_size_adaptation": True,
            }
        )
        start = index
        stage = next_stage
    segments.append(
        {
            "segment_index": len(segments),
            "stage": "slow" if stage == 1 else "fast",
            "segment_start_step": start + 1,
            "segment_end_step": int(array.shape[0]),
            "metric_adaptation": stage == 1,
            "step_size_adaptation": True,
        }
    )
    return segments


def _fisher_hmc_prescription(num_steps: int) -> dict[str, Any]:
    """Discretize the Fisher-HMC warmup described by Seyboldt et al."""
    if num_steps < 3:
        raise ValueError("Fisher-HMC phase metadata requires at least three steps")

    phase_1_end = max(1, min(num_steps - 2, math.floor(0.30 * num_steps)))
    phase_2_end = max(
        phase_1_end + 1,
        min(num_steps - 1, math.floor(0.85 * num_steps)),
    )
    phases = [
        {
            "phase_index": 1,
            "name": "early_metric_and_step_size",
            "target_fraction": 0.30,
            "phase_start_step": 1,
            "phase_end_step": phase_1_end,
            "metric_adaptation": True,
            "refresh_or_memory_period": 10,
            "online_memory_wipe_period": 10,
            "low_rank_metric_update_period": 10,
            "step_size_adaptation": True,
            "step_size_initialized_at_start": True,
            "step_size_reinitialized_at_start": False,
            "acceptance_statistic": "metropolis_acceptance_probability",
            "source": FISHER_HMC_SOURCE,
        },
        {
            "phase_index": 2,
            "name": "main_metric_and_step_size",
            "target_fraction": 0.55,
            "phase_start_step": phase_1_end + 1,
            "phase_end_step": phase_2_end,
            "metric_adaptation": True,
            "refresh_or_memory_period": 80,
            "online_memory_wipe_period": 80,
            "low_rank_metric_update_period": 80,
            "step_size_adaptation": True,
            "step_size_initialized_at_start": False,
            "step_size_reinitialized_at_start": True,
            "acceptance_statistic": "metropolis_acceptance_probability",
            "source": FISHER_HMC_SOURCE,
        },
        {
            "phase_index": 3,
            "name": "final_step_size_only",
            "target_fraction": 0.15,
            "phase_start_step": phase_2_end + 1,
            "phase_end_step": num_steps,
            "metric_adaptation": False,
            "refresh_or_memory_period": None,
            "online_memory_wipe_period": None,
            "low_rank_metric_update_period": None,
            "step_size_adaptation": True,
            "step_size_initialized_at_start": False,
            "step_size_reinitialized_at_start": False,
            "acceptance_statistic": "symmetric_acceptance_statistic",
            "source": FISHER_HMC_SOURCE,
        },
    ]

    refresh_events: list[dict[str, Any]] = []
    for phase in phases:
        period_value = phase["refresh_or_memory_period"]
        if period_value is None:
            phase["first_global_refresh_step"] = None
            phase["global_refresh_alignment"] = None
            continue
        period = int(period_value)
        start = int(phase["phase_start_step"])
        end = int(phase["phase_end_step"])
        first_refresh = ((start + period - 1) // period) * period
        phase["first_global_refresh_step"] = (
            first_refresh if first_refresh <= end else None
        )
        phase["global_refresh_alignment"] = "warmup_step modulo L equals zero"
        for step in range(first_refresh, end + 1, period):
            refresh_events.append(
                {
                    "refresh_index": len(refresh_events),
                    "warmup_step": step,
                    "phase_index": int(phase["phase_index"]),
                    "period": period,
                    "diagonal_action": "wipe_online_estimator_memory",
                    "low_rank_action": "update_metric",
                    "global_multiple_index": step // period,
                    "alignment_rule": "warmup_step modulo L equals zero",
                    "source": FISHER_HMC_SOURCE,
                }
            )

    return {
        "name": "seyboldt_fisher_hmc_warmup",
        "label": "Seyboldt et al. Fisher-HMC warmup",
        "scope": (
            "warmup prescription reported for Fisher HMC; "
            "not a generic schedule for every nutpie version"
        ),
        "source": FISHER_HMC_SOURCE,
        "source_sections": ["3.2", "3.3"],
        "num_steps": num_steps,
        "boundary_policy": "prescribed_static",
        "phase_rounding": (
            "floor cumulative boundaries at 30% and 85%; "
            "assign the remainder to the final phase"
        ),
        "refresh_alignment": "one-based global multiples of the phase period",
        "online_memory_window_rule": "a_n=max(0,L(floor(n/L)-1))",
        "low_rank_update_rule": "update the metric at global multiples of L",
        "phases": phases,
        "refresh_events": refresh_events,
    }


def schedule_prescriptions(
    num_steps: int,
    *,
    n_chains: int,
    dimension: int,
) -> dict[str, Any]:
    """Materialize the schedule prescriptions used in comparison plots."""
    from blackjax.adaptation.low_rank_adaptation import (
        build_growing_window_schedule,
    )
    from blackjax.adaptation.staged_adaptation import build_schedule

    from blackjax.adaptation.meta._calibration import _MAX_RANK_CAP
    from blackjax.adaptation.meta._schedule import _build_mc_window_schedule

    actual_rank = min(_MAX_RANK_CAP, max(dimension // 2, 1))
    stan = build_schedule(num_steps)
    proportional = build_growing_window_schedule(num_steps)
    if n_chains > 1:
        controller = _build_mc_window_schedule(
            num_steps,
            n_chains,
            actual_rank,
        )
        controller_name = "pooled_proportional_growing"
    else:
        controller = proportional
        controller_name = "proportional_growing"

    def _entry(name: str, schedule: Any) -> dict[str, Any]:
        num_schedule_steps = int(np.asarray(schedule).shape[0])
        return {
            "name": name,
            "num_steps": num_schedule_steps,
            "boundary_policy": "prescribed_static",
            "stage_segments": _stage_segments(schedule),
            "metric_windows": _slow_windows(schedule),
        }

    return {
        "stan": _entry("stan_doubling", stan),
        "seyboldt_fisher_hmc": _fisher_hmc_prescription(num_steps),
        "proportional_growing": _entry(
            "proportional_growing",
            proportional,
        ),
        "controller_actual": {
            **_entry(controller_name, controller),
            "evidence_evaluated_at_scheduled_boundaries": True,
            "boundaries_selected_from_evidence": False,
        },
    }


def _json_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"expected scalar event value, got shape {array.shape}")
    scalar = array.item()
    if isinstance(scalar, float) and not math.isfinite(scalar):
        return None
    if isinstance(scalar, np.generic):
        return scalar.item()
    return scalar


def build_metric_window_events(
    *,
    prescriptions: Mapping[str, Any],
    integration_steps: Any,
    step_sizes: Any,
    controller_budget_trace: Any,
    evidence_at_boundary: Callable[[int, Mapping[str, int]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the shared event envelope at prescribed metric boundaries.

    The caller supplies diagnostic evidence evaluated at each boundary.  The
    schedule fields are protected here so evidence can never change, add, or
    relabel a controller boundary.
    """

    controller = prescriptions.get("controller_actual")
    if not isinstance(controller, Mapping):
        raise ValueError("controller_actual schedule prescription is missing")
    windows = controller.get("metric_windows")
    num_steps = controller.get("num_steps")
    if not isinstance(windows, list):
        raise ValueError("controller_actual metric windows are missing")
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("controller_actual step count is invalid")

    integration_array = np.asarray(integration_steps)
    step_size_array = np.asarray(step_sizes)
    budget_array = np.asarray(controller_budget_trace)
    if integration_array.shape[0] != num_steps:
        raise ValueError("integration-step trace does not match the schedule")
    if step_size_array.shape[0] != num_steps:
        raise ValueError("step-size trace does not match the schedule")
    if budget_array.shape[0] != num_steps:
        raise ValueError("controller-budget trace does not match the schedule")
    if not windows:
        return []

    per_step_gradients = integration_array.reshape(num_steps, -1).sum(axis=1)
    cumulative_gradients = np.cumsum(per_step_gradients)
    protected = {
        "event_schema_version",
        "window_index",
        "window_start_step",
        "window_end_step",
        "event",
        "boundary_policy",
        "boundary_selected_from_evidence",
        "cumulative_gradients",
        "cumulative_gradient_basis",
        "controller_budget_used",
        "step_size",
        "global_exploration",
    }

    events: list[dict[str, Any]] = []
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError("controller_actual contains a malformed window")
        trace_index = int(window["window_end_step"]) - 1
        evidence = dict(evidence_at_boundary(trace_index, window))
        conflicts = protected.intersection(evidence)
        if conflicts:
            raise ValueError(
                "evidence callback attempted to set protected event fields: "
                + ", ".join(sorted(conflicts))
            )
        events.append(
            {
                **evidence,
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "window_index": int(window["window_index"]),
                "window_start_step": int(window["window_start_step"]),
                "window_end_step": int(window["window_end_step"]),
                "event": "scheduled_metric_window_end",
                "boundary_policy": "prescribed_static",
                "boundary_selected_from_evidence": False,
                "cumulative_gradients": int(
                    cumulative_gradients[trace_index]
                ),
                "cumulative_gradient_basis": "sum_num_integration_steps",
                "controller_budget_used": int(
                    _json_scalar(budget_array[trace_index])
                ),
                "step_size": float(
                    _json_scalar(step_size_array[trace_index])
                ),
                "global_exploration": "not_established",
            }
        )
    return events


def extract_window_events(
    warmup_info: Any,
    *,
    max_grad_budget: int,
    n_chains: int,
    dimension: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract controller evidence at each prescribed metric-window boundary.

    The event stream is evaluation-only. The controller selects a route when
    it reaches a precomputed boundary; no evidence field in this stream moves
    that boundary.
    """
    from blackjax.adaptation.meta import (
        extract_meta_verdict,
        extract_multi_chain_verdict,
    )

    state_trace = warmup_info.adaptation_state.imm_state
    num_steps = int(np.asarray(state_trace.budget_used).shape[0])
    prescriptions = schedule_prescriptions(
        num_steps,
        n_chains=n_chains,
        dimension=dimension,
    )
    integration_steps = np.asarray(warmup_info.info.num_integration_steps)
    final_budget_used = int(np.asarray(state_trace.budget_used)[-1])

    def _evidence(
        trace_index: int,
        _window: Mapping[str, int],
    ) -> Mapping[str, Any]:
        state = jax.tree_util.tree_map(lambda value: value[trace_index], state_trace)
        if n_chains > 1:
            verdict = extract_multi_chain_verdict(
                state,
                max_grad_budget=max_grad_budget,
                num_warmup_steps=final_budget_used,
            )
        else:
            verdict = extract_meta_verdict(
                state,
                max_grad_budget=max_grad_budget,
                num_warmup_steps=final_budget_used,
            )
        flags = verdict.flags

        event: dict[str, Any] = {
            "route": verdict.route,
            "effective_rank": int(verdict.effective_rank),
            "metric_route_status": str(
                flags.get("metric_route_status", "unassessed")
            ),
            "metric_route_basis": str(
                flags.get("metric_route_basis", "none")
            ),
            "metric_scope": str(flags.get("metric_scope", "unassessed")),
            "observed_ensemble_evidence": str(
                flags.get("observed_ensemble_evidence", "unassessed")
            ),
            "handoff": str(flags.get("handoff", "none")),
            "confidence": verdict.confidence,
            "airm_stable": verdict.exit_reason == "airm_velocity_converged",
            "converged_at_controller_budget": int(
                _json_scalar(state.converged_at_step)
            ),
            "airm_velocity_previous": _json_scalar(state.airm_vel_prev),
            "airm_velocity_current": _json_scalar(state.airm_vel_curr),
            "r2_latest": _json_scalar(state.r2_latest),
        }
        if n_chains > 1:
            event.update(
                {
                    "within_lam1": _json_scalar(state.within_lam1),
                    "chain_consistency_psi": _json_scalar(
                        state.chain_consistency_psi
                    ),
                    "chain_collinearity": _json_scalar(
                        state.chain_collinearity
                    ),
                    "r1_top": _json_scalar(state.r1_top),
                    "unimodality_passed": bool(
                        _json_scalar(state.unimodality_passed)
                    ),
                    "unimodality_flag_count": int(
                        _json_scalar(state.unimodality_flag_count)
                    ),
                    "deferred_to_ensemble": bool(
                        _json_scalar(state.deferred_to_ensemble)
                    ),
                }
            )
        return event

    events = build_metric_window_events(
        prescriptions=prescriptions,
        integration_steps=integration_steps,
        step_sizes=warmup_info.adaptation_state.step_size,
        controller_budget_trace=state_trace.budget_used,
        evidence_at_boundary=_evidence,
    )

    return events, prescriptions


def validate_window_events(
    events: Any,
    prescriptions: Any,
) -> list[str]:
    """Return event-schema errors; an empty list means the stream is valid."""
    errors: list[str] = []
    if not isinstance(events, list) or not events:
        return ["window_events must be a nonempty list"]
    if not isinstance(prescriptions, dict):
        return ["schedule_prescriptions must be an object"]
    for key in (
        "stan",
        "seyboldt_fisher_hmc",
        "proportional_growing",
        "controller_actual",
    ):
        if key not in prescriptions:
            errors.append(f"schedule_prescriptions missing {key}")
    for key in ("stan", "proportional_growing", "controller_actual"):
        schedule = prescriptions.get(key, {})
        num_schedule_steps = schedule.get("num_steps")
        segments = schedule.get("stage_segments")
        if not isinstance(num_schedule_steps, int) or num_schedule_steps < 1:
            errors.append(f"{key} schedule has an invalid step count")
            continue
        if not isinstance(segments, list) or not segments:
            errors.append(f"{key} schedule lacks stage segments")
            continue
        previous_segment_end = 0
        for segment_index, segment in enumerate(segments):
            prefix = f"{key} stage_segments[{segment_index}]"
            stage = segment.get("stage")
            start = segment.get("segment_start_step")
            end = segment.get("segment_end_step")
            if segment.get("segment_index") != segment_index:
                errors.append(f"{prefix} has the wrong index")
            if stage not in ("fast", "slow"):
                errors.append(f"{prefix} has an unknown stage")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start != previous_segment_end + 1
                or end < start
            ):
                errors.append(f"{prefix} has invalid bounds")
            else:
                previous_segment_end = end
            if segment.get("metric_adaptation") is not (stage == "slow"):
                errors.append(f"{prefix} has an inconsistent metric state")
            if segment.get("step_size_adaptation") is not True:
                errors.append(f"{prefix} disables step-size adaptation")
        if previous_segment_end != num_schedule_steps:
            errors.append(f"{key} stage segments do not cover the schedule")
        metric_windows = schedule.get("metric_windows")
        if not isinstance(metric_windows, list):
            errors.append(f"{key} schedule lacks metric windows")
        else:
            slow_intervals = [
                (
                    segment.get("segment_start_step"),
                    segment.get("segment_end_step"),
                )
                for segment in segments
                if segment.get("stage") == "slow"
            ]
            for window_index, window in enumerate(metric_windows):
                start = window.get("window_start_step")
                end = window.get("window_end_step")
                if not any(
                    isinstance(start, int)
                    and isinstance(end, int)
                    and isinstance(slow_start, int)
                    and isinstance(slow_end, int)
                    and slow_start <= start <= end <= slow_end
                    for slow_start, slow_end in slow_intervals
                ):
                    errors.append(
                        f"{key} metric window {window_index} is outside a slow stage"
                    )
    fisher = prescriptions.get("seyboldt_fisher_hmc", {})
    if fisher.get("source") != FISHER_HMC_SOURCE:
        errors.append("Fisher-HMC schedule has an unknown source")
    if fisher.get("boundary_policy") != "prescribed_static":
        errors.append("Fisher-HMC schedule does not identify prescribed phases")
    if fisher.get("source_sections") != ["3.2", "3.3"]:
        errors.append("Fisher-HMC schedule has incorrect source sections")
    if (
        fisher.get("online_memory_window_rule")
        != "a_n=max(0,L(floor(n/L)-1))"
    ):
        errors.append("Fisher-HMC schedule has an incorrect memory-window rule")
    if (
        fisher.get("low_rank_update_rule")
        != "update the metric at global multiples of L"
    ):
        errors.append("Fisher-HMC schedule has an incorrect low-rank update rule")
    fisher_num_steps = fisher.get("num_steps")
    if not isinstance(fisher_num_steps, int) or fisher_num_steps < 3:
        errors.append("Fisher-HMC schedule has an invalid step count")
    phases = fisher.get("phases")
    expected_phase_fields = (
        (1, 0.30, 10, True, False),
        (2, 0.55, 80, True, True),
        (3, 0.15, None, False, False),
    )
    if not isinstance(phases, list) or len(phases) != 3:
        errors.append("Fisher-HMC schedule must contain three phases")
    else:
        previous_phase_end = 0
        for phase, expected in zip(phases, expected_phase_fields, strict=True):
            index, fraction, period, metric_on, reinitialized = expected
            if phase.get("phase_index") != index:
                errors.append(f"Fisher-HMC phase {index} has the wrong index")
            if phase.get("target_fraction") != fraction:
                errors.append(f"Fisher-HMC phase {index} has the wrong fraction")
            if phase.get("refresh_or_memory_period") != period:
                errors.append(f"Fisher-HMC phase {index} has the wrong cadence")
            if phase.get("metric_adaptation") is not metric_on:
                errors.append(f"Fisher-HMC phase {index} has the wrong metric state")
            if phase.get("step_size_reinitialized_at_start") is not reinitialized:
                errors.append(
                    f"Fisher-HMC phase {index} has the wrong step-size restart state"
                )
            if phase.get("source") != FISHER_HMC_SOURCE:
                errors.append(f"Fisher-HMC phase {index} has an unknown source")
            start = phase.get("phase_start_step")
            end = phase.get("phase_end_step")
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start != previous_phase_end + 1
                or end < start
            ):
                errors.append(f"Fisher-HMC phase {index} has invalid bounds")
            else:
                previous_phase_end = end
            if period is None:
                expected_first_refresh = None
                expected_alignment = None
            elif isinstance(start, int) and isinstance(end, int):
                candidate = ((start + period - 1) // period) * period
                expected_first_refresh = candidate if candidate <= end else None
                expected_alignment = "warmup_step modulo L equals zero"
            else:
                expected_first_refresh = None
                expected_alignment = "warmup_step modulo L equals zero"
            if phase.get("first_global_refresh_step") != expected_first_refresh:
                errors.append(
                    f"Fisher-HMC phase {index} has the wrong first refresh"
                )
            if phase.get("global_refresh_alignment") != expected_alignment:
                errors.append(
                    f"Fisher-HMC phase {index} has the wrong refresh alignment"
                )
        if (
            isinstance(fisher_num_steps, int)
            and previous_phase_end != fisher_num_steps
        ):
            errors.append("Fisher-HMC phases do not cover the declared step count")
    refresh_events = fisher.get("refresh_events")
    if not isinstance(refresh_events, list):
        errors.append("Fisher-HMC schedule lacks refresh events")
    elif isinstance(phases, list) and len(phases) == 3:
        expected_refreshes = []
        for phase in phases[:2]:
            period = phase.get("refresh_or_memory_period")
            start = phase.get("phase_start_step")
            end = phase.get("phase_end_step")
            if all(isinstance(value, int) for value in (period, start, end)):
                first = ((start + period - 1) // period) * period
                expected_refreshes.extend(
                    (phase["phase_index"], step, period)
                    for step in range(first, end + 1, period)
                )
        observed_refreshes = [
            (
                refresh.get("phase_index"),
                refresh.get("warmup_step"),
                refresh.get("period"),
            )
            for refresh in refresh_events
            if isinstance(refresh, dict)
        ]
        if observed_refreshes != expected_refreshes:
            errors.append("Fisher-HMC refresh events do not match phase cadences")
        if any(
            not isinstance(refresh, dict)
            or refresh.get("source") != FISHER_HMC_SOURCE
            for refresh in refresh_events
        ):
            errors.append("Fisher-HMC refresh events have an unknown source")
        if any(
            not isinstance(refresh, dict)
            or not isinstance(refresh.get("warmup_step"), int)
            or not isinstance(refresh.get("period"), int)
            or refresh["period"] <= 0
            or refresh.get("global_multiple_index")
            != refresh["warmup_step"] // refresh["period"]
            or refresh.get("alignment_rule")
            != "warmup_step modulo L equals zero"
            for refresh in refresh_events
        ):
            errors.append("Fisher-HMC refresh events have invalid alignment metadata")
    actual_windows = prescriptions.get("controller_actual", {}).get(
        "metric_windows",
        [],
    )
    actual_controller = prescriptions.get("controller_actual", {})
    if actual_controller.get("boundary_policy") != "prescribed_static":
        errors.append("controller schedule does not identify static boundaries")
    if (
        actual_controller.get("evidence_evaluated_at_scheduled_boundaries")
        is not True
    ):
        errors.append("controller schedule omits boundary-evaluation semantics")
    if actual_controller.get("boundaries_selected_from_evidence") is not False:
        errors.append("controller schedule claims evidence-selected boundaries")
    event_windows = [
        {
            "window_index": event.get("window_index"),
            "window_start_step": event.get("window_start_step"),
            "window_end_step": event.get("window_end_step"),
        }
        for event in events
    ]
    if event_windows != actual_windows:
        errors.append("window events do not match the prescribed controller windows")
    previous_end = 0
    previous_gradients = -1
    for index, event in enumerate(events):
        prefix = f"window_events[{index}]"
        if event.get("event_schema_version") != EVENT_SCHEMA_VERSION:
            errors.append(f"{prefix} has an unsupported schema version")
        if event.get("event") != "scheduled_metric_window_end":
            errors.append(f"{prefix} has an unknown event type")
        if event.get("window_index") != index:
            errors.append(f"{prefix} has the wrong window index")
        if event.get("boundary_policy") != "prescribed_static":
            errors.append(f"{prefix} does not identify a prescribed boundary")
        if event.get("boundary_selected_from_evidence") is not False:
            errors.append(f"{prefix} claims an evidence-selected boundary")
        start = event.get("window_start_step")
        end = event.get("window_end_step")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{prefix} has invalid window bounds")
        elif not (previous_end < start <= end):
            errors.append(f"{prefix} window bounds are not increasing")
        else:
            previous_end = end
        if not isinstance(event.get("cumulative_gradients"), int):
            errors.append(f"{prefix} lacks cumulative gradients")
        elif event["cumulative_gradients"] <= previous_gradients:
            errors.append(f"{prefix} cumulative gradients are not increasing")
        else:
            previous_gradients = event["cumulative_gradients"]
        if event.get("cumulative_gradient_basis") != "sum_num_integration_steps":
            errors.append(f"{prefix} has an unknown gradient-count basis")
        if not isinstance(event.get("controller_budget_used"), int):
            errors.append(f"{prefix} lacks the controller budget")
        if not isinstance(event.get("step_size"), (int, float)):
            errors.append(f"{prefix} lacks the boundary step size")
        if event.get("global_exploration") != "not_established":
            errors.append(f"{prefix} overstates global exploration")
    return errors
