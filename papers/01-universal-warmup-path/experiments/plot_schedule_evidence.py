#!/usr/bin/env python
"""Plot prescribed warmup boundaries and stored controller evidence events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(row)
    return rows


def _select_row(
    rows: list[dict[str, Any]],
    *,
    arm_id: str | None,
    sr: float,
    seed: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if float(row.get("SR", float("nan"))) == sr
        and int(row.get("seed", -1)) == seed
        and (arm_id is None or row.get("arm_id") == arm_id)
    ]
    if len(matches) != 1:
        identities = [
            (row.get("arm_id"), row.get("SR"), row.get("seed")) for row in matches
        ]
        raise ValueError(
            "selection must identify exactly one row; "
            f"matched {len(matches)} rows: {identities}"
        )
    return matches[0]


def _plot_schedule(
    axis: Any,
    prescription: dict[str, Any],
    *,
    y: float,
    color: str,
) -> None:
    segments = prescription.get("stage_segments")
    windows = prescription.get("metric_windows")
    if not isinstance(segments, list) or not segments:
        raise ValueError("schedule prescription must contain stage segments")
    if not isinstance(windows, list):
        raise ValueError("schedule prescription must contain metric windows")
    for segment in segments:
        start = int(segment["segment_start_step"])
        end = int(segment["segment_end_step"])
        stage = segment["stage"]
        if stage not in ("fast", "slow"):
            raise ValueError(f"unknown schedule stage {stage!r}")
        axis.plot(
            [start, end],
            [y, y],
            color=color,
            linewidth=3,
            alpha=0.45,
            linestyle="-" if stage == "slow" else "--",
        )
        axis.plot(
            [end, end],
            [y - 0.12, y + 0.12],
            color=color,
            linewidth=0.8,
            alpha=0.7,
        )
    for window in windows:
        end = int(window["window_end_step"])
        axis.plot(
            [end, end],
            [y - 0.12, y + 0.12],
            color=color,
            linewidth=1,
        )


def _plot_fisher_hmc(
    axis: Any,
    prescription: dict[str, Any],
    *,
    y: float,
    color: str,
) -> None:
    phases = prescription.get("phases")
    refresh_events = prescription.get("refresh_events")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValueError("Fisher-HMC prescription must contain three phases")
    if not isinstance(refresh_events, list):
        raise ValueError("Fisher-HMC prescription must contain refresh events")

    phase_labels = (
        "30%: metric + step size; L=10",
        "55%: metric + step size; L=80; step size reinit",
        "15%: step size only",
    )
    for phase, label in zip(phases, phase_labels, strict=True):
        start = int(phase["phase_start_step"])
        end = int(phase["phase_end_step"])
        metric_on = phase.get("metric_adaptation") is True
        axis.plot(
            [start, end],
            [y, y],
            color=color,
            linewidth=3,
            alpha=0.45,
            linestyle="-" if metric_on else "--",
        )
        axis.plot(
            [end, end],
            [y - 0.12, y + 0.12],
            color=color,
            linewidth=1,
        )
        axis.annotate(
            label,
            ((start + end) / 2, y),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=6.5,
            ha="center",
            color=color,
        )

    refresh_steps = [int(event["warmup_step"]) for event in refresh_events]
    if refresh_steps:
        axis.scatter(
            refresh_steps,
            [y] * len(refresh_steps),
            marker="|",
            color=color,
            s=13,
            alpha=0.65,
            linewidths=0.5,
            zorder=4,
        )


def _first_event(
    events: list[dict[str, Any]],
    predicate: Any,
) -> dict[str, Any] | None:
    return next((event for event in events if predicate(event)), None)


def render(
    row: dict[str, Any],
    output: Path,
    *,
    force: bool = False,
) -> None:
    """Render one stored experiment row without importing the controller."""
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass --force to replace the derived figure"
        )
    schedules = row.get("schedule_prescriptions")
    events = row.get("window_events")
    if not isinstance(schedules, dict) or not isinstance(events, list) or not events:
        raise ValueError("selected row has no recoverable schedule/event stream")

    fig, axis = plt.subplots(figsize=(8.0, 3.4))
    lanes = (
        ("stan", 3.0, "#666666", "Stan prescribed"),
        (
            "seyboldt_fisher_hmc",
            2.0,
            "#0072B2",
            "Fisher-HMC paper prescription",
        ),
        (
            "controller_actual",
            1.0,
            "#009E73",
            "universal-path scheduled",
        ),
    )
    for key, y, color, _label in lanes:
        if key == "seyboldt_fisher_hmc":
            _plot_fisher_hmc(axis, schedules[key], y=y, color=color)
        else:
            _plot_schedule(
                axis,
                schedules[key],
                y=y,
                color=color,
            )

    route_event = _first_event(events, lambda event: event.get("route") == "low_rank")
    handoff_event = _first_event(
        events,
        lambda event: event.get("handoff") not in (None, "none")
        or event.get("route") == "reparam_suggested",
    )
    stable_event = _first_event(events, lambda event: event.get("airm_stable") is True)
    observed = (
        (route_event, "^", "#D55E00", "first low-rank latch"),
        (handoff_event, "X", "#CC79A7", "first handoff/reparameterize"),
        (stable_event, "*", "#E69F00", "first AIRM-stability marker"),
    )
    for event, marker, color, label in observed:
        if event is None:
            continue
        step = int(event["window_end_step"])
        axis.scatter(
            [step],
            [1.0],
            marker=marker,
            color=color,
            s=75,
            zorder=5,
            label=label,
        )
        axis.annotate(
            f"{int(event['cumulative_gradients']):,} gradients",
            (step, 1.0),
            xytext=(3, -20),
            textcoords="offset points",
            fontsize=7,
            rotation=30,
        )

    labels = [label for _key, _y, _color, label in lanes]
    axis.set_yticks([3.0, 2.0, 1.0], labels)
    axis.set_xlabel("warmup step (window endpoints are one-based)")
    axis.set_ylim(0.55, 3.45)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axis.set_title(
        "Prescribed boundaries and evidence evaluated at those boundaries\n"
        "the current controller chooses a route; it does not move the boundary",
        fontsize=10,
    )
    axis.text(
        0.01,
        0.98,
        "solid: metric + step-size stage; dashed: step-size-only stage",
        transform=axis.transAxes,
        fontsize=6.5,
        color="#555555",
        va="top",
    )
    if any(event is not None for event, *_rest in observed):
        axis.legend(loc="upper right", frameon=False, fontsize=7)
    identity = (
        f"{row.get('arm_id', 'standalone')}, "
        f"SR={row.get('SR')}, seed={row.get('seed')}"
    )
    axis.text(
        0.01,
        0.02,
        identity,
        transform=axis.transAxes,
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm-id")
    parser.add_argument("--sr", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        row = _select_row(
            _load_rows(args.input),
            arm_id=args.arm_id,
            sr=args.sr,
            seed=args.seed,
        )
        render(row, args.out, force=args.force)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"schedule figure failed: {exc}\n")


if __name__ == "__main__":
    main()
