#!/usr/bin/env python3
"""Report Paper 1 numbers from the frozen summary and authenticated raw files.

The full-corpus validator is the authority that constructs ``frozen_summary``.
This script performs no sampling and writes nothing.  It verifies the source
execution manifest and the manuscript-facing raw files, then emits a
deterministic JSON report containing the direct summary values and the small
reporting transforms used by the paper (ratios, percentages, geometric means,
and the seed-clustered GMM interval).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "paper1-paper-number-report-v1"

TRUSTED_REFERENCE_HASHES = {
    "source_execution_manifest": (
        "79bcb82b84b83bc9406b0694e8912dff63862dbd7f9ff309fd63e45eaa5a9ffb"
    ),
    "execution_manifest_overlay": (
        "f86c1e02ba233ffabde97c61c0868e0be0991e48a506ad01608160c4ba5f8296"
    ),
    "frozen_summary": (
        "94136b0e528b883148540efdbe43cd24ebd39bcaa905b86bd73a8c8d4763f24a"
    ),
    "validation_attestation": (
        "85b60b2958cfc3395828c0236a77d1fb4ee7ded16a4249e499fff339cee60b02"
    ),
}

RAW_INPUTS: dict[str, tuple[str, ...]] = {
    "fixed_efficiency": (
        "fixed/illcond.jsonl",
        "fixed/german.jsonl",
        "fixed/manual_illcond.jsonl",
        "fixed/manual_german.jsonl",
        "fixed/manual_population_illcond.jsonl",
        "fixed/manual_population_german.jsonl",
    ),
    "shared_step_size": (
        "shared_step/current.jsonl",
        "shared_step/historical.jsonl",
    ),
    "controlled_gmm": (
        "gmm/gmm_k2_primary_60k.jsonl",
        "gmm/gmm_k2_budget_20k.jsonl",
        "gmm/gmm_k2_budget_120k.jsonl",
        "gmm/gmm_k2_single_chain_60k.jsonl",
        "gmm/gmm_k2_matched_diagonal_60k.jsonl",
        "gmm/gmm_k3_primary_60k.jsonl",
        "gmm/gmm_k3_matched_diagonal_60k.jsonl",
    ),
    "schedule_configuration": (
        "schedule/schedule_configuration.jsonl",
    ),
    "restart_ablation": (
        "restart/restart_ablation.jsonl",
    ),
    "kernel_family": (
        "kernel/kernel_family.jsonl",
    ),
    "figures": (
        "figures/figure_bbp.pdf",
        "figures/schedule_evidence.pdf",
        "figures/schedule_evidence.png",
        "gmm/gmm_k2_primary_60k.jsonl",
    ),
}

SUMMARY_KEYS: dict[str, tuple[str, ...]] = {
    "fixed_efficiency": ("efficiency",),
    "shared_step_size": ("shared_step_size",),
    "controlled_gmm": ("controlled_gmm",),
    "schedule_configuration": ("schedule_configuration",),
    "restart_ablation": ("restart_ablation",),
    "kernel_family": ("kernel_family",),
    "figures": ("figures", "controlled_gmm"),
}

SECTIONS = tuple(SUMMARY_KEYS)


class ReportError(ValueError):
    """Raised when the frozen inputs do not establish a valid report."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportError(f"{label} must be an array")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ReportError(f"{label} must be positive")
    return result


def _ratio(numerator: Any, denominator: Any, label: str) -> float:
    return _number(numerator, f"{label} numerator") / _positive(
        denominator,
        f"{label} denominator",
    )


def _range(values: Iterable[float], label: str) -> list[float]:
    materialized = list(values)
    if not materialized:
        raise ReportError(f"{label} has no values")
    return [min(materialized), max(materialized)]


def _geometric_mean(values: Iterable[float], label: str) -> float:
    materialized = [_positive(value, label) for value in values]
    if not materialized:
        raise ReportError(f"{label} has no values")
    return math.exp(statistics.fmean(math.log(value) for value in materialized))


def _require_valid_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("status") != "valid":
        raise ReportError("frozen summary status must be 'valid'")
    if summary.get("mode") != "full":
        raise ReportError("paper-number reporting requires a full-corpus summary")
    for section in SUMMARY_KEYS:
        for key in SUMMARY_KEYS[section]:
            if key not in summary:
                raise ReportError(f"frozen summary is missing {key!r}")


def _load_authenticated_reference(
    summary_path: Path,
    attestation_path: Path,
    result_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the one released summary only after checking trusted anchors."""

    summary_path = summary_path.resolve()
    attestation_path = attestation_path.resolve()
    observed_summary_hash = sha256_file(summary_path)
    if observed_summary_hash != TRUSTED_REFERENCE_HASHES["frozen_summary"]:
        raise ReportError(
            "frozen summary does not match the trusted release SHA-256"
        )
    observed_attestation_hash = sha256_file(attestation_path)
    if (
        observed_attestation_hash
        != TRUSTED_REFERENCE_HASHES["validation_attestation"]
    ):
        raise ReportError(
            "validation attestation does not match the trusted release SHA-256"
        )

    summary = _load_object(summary_path)
    attestation = _load_object(attestation_path)
    _require_valid_summary(summary)
    if attestation.get("schema_version") != "paper1-validation-attestation-v1":
        raise ReportError("validation attestation schema is not supported")
    if attestation.get("status") != "valid":
        raise ReportError("validation attestation status must be 'valid'")
    if attestation.get("source_run_id") != summary.get("run_id"):
        raise ReportError("validation attestation and summary run IDs differ")
    if attestation.get("source_run_directory_name") != result_dir.resolve().name:
        raise ReportError(
            "validation attestation names a different source result directory"
        )

    attestation_inputs = _mapping(
        attestation.get("inputs"),
        "validation_attestation.inputs",
    )
    repaired = _mapping(
        attestation.get("repaired_validation"),
        "validation_attestation.repaired_validation",
    )
    anchored_fields = (
        (
            attestation_inputs,
            "source_execution_manifest_sha256",
            "source_execution_manifest",
        ),
        (
            repaired,
            "execution_manifest_overlay_sha256",
            "execution_manifest_overlay",
        ),
        (repaired, "frozen_summary_sha256", "frozen_summary"),
    )
    for source, field, anchor in anchored_fields:
        if source.get(field) != TRUSTED_REFERENCE_HASHES[anchor]:
            raise ReportError(
                f"validation attestation has an untrusted {field}"
            )

    correction_scope = _mapping(
        attestation.get("correction_scope"),
        "validation_attestation.correction_scope",
    )
    if any(
        correction_scope.get(field) is not False
        for field in (
            "figures_changed",
            "producer_computation_rerun",
            "producer_sources_changed",
            "raw_jsonl_changed",
            "raw_npz_changed",
        )
    ):
        raise ReportError(
            "validation attestation does not establish a validation-only repair"
        )

    return (
        summary,
        attestation,
        {
            "frozen_summary": {
                "path": summary_path.name,
                "sha256": observed_summary_hash,
            },
            "validation_attestation": {
                "path": attestation_path.name,
                "sha256": observed_attestation_hash,
            },
            "trusted_sha256_anchors": dict(TRUSTED_REFERENCE_HASHES),
        },
    )


def _verify_inputs(
    summary: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """Authenticate the execution manifest and every mapped raw input."""

    result_dir = result_dir.resolve()
    if not result_dir.is_dir():
        raise ReportError(f"result directory does not exist: {result_dir}")
    execution_path = result_dir / "execution_manifest.json"
    execution = _load_object(execution_path)
    traceability = _mapping(summary["traceability"], "traceability")
    source_execution = _mapping(
        traceability["source_execution_manifest"],
        "traceability.source_execution_manifest",
    )
    expected_execution_hash = source_execution.get("sha256")
    observed_execution_hash = sha256_file(execution_path)
    if observed_execution_hash != expected_execution_hash:
        raise ReportError(
            "source execution manifest hash does not match frozen summary"
        )
    if execution.get("run_id") != summary.get("run_id"):
        raise ReportError("execution manifest and frozen summary run IDs differ")

    output_files = _mapping(
        execution.get("output_files"),
        "execution_manifest.output_files",
    )
    verified: dict[str, dict[str, Any]] = {}
    for relative in sorted({path for paths in RAW_INPUTS.values() for path in paths}):
        path = result_dir / relative
        if not path.is_file():
            raise ReportError(f"mapped raw input is missing: {relative}")
        recorded = _mapping(
            output_files.get(relative),
            f"execution_manifest.output_files[{relative!r}]",
        )
        observed_hash = sha256_file(path)
        if observed_hash != recorded.get("sha256"):
            raise ReportError(f"raw input hash mismatch: {relative}")
        if path.stat().st_size != recorded.get("bytes"):
            raise ReportError(f"raw input size mismatch: {relative}")
        verified[relative] = {
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }

    arms = _sequence(
        _mapping(summary["controlled_gmm"], "controlled_gmm")["arms"],
        "controlled_gmm.arms",
    )
    for arm_value in arms:
        arm = _mapping(arm_value, "controlled_gmm arm")
        arm_id = arm.get("arm_id")
        relative = f"gmm/{arm_id}.jsonl"
        if relative not in verified:
            raise ReportError(f"unmapped controlled-GMM arm: {arm_id!r}")
        if verified[relative]["sha256"] != arm.get("sha256"):
            raise ReportError(f"GMM arm hash differs from frozen summary: {arm_id}")

    figures = _mapping(summary["figures"], "figures")
    for name, value in figures.items():
        figure = _mapping(value, f"figures[{name!r}]")
        relative = figure.get("path")
        if relative not in verified:
            raise ReportError(f"unmapped frozen figure: {relative!r}")
        if verified[relative]["sha256"] != figure.get("sha256"):
            raise ReportError(f"figure hash differs from frozen summary: {name}")

    return {
        "source_execution_manifest": {
            "path": "execution_manifest.json",
            "sha256": observed_execution_hash,
        },
        "raw_inputs": verified,
    }


def _fixed_efficiency(summary: Mapping[str, Any]) -> dict[str, Any]:
    efficiency = _mapping(summary["efficiency"], "efficiency")
    cells = _sequence(efficiency["cells"], "efficiency.cells")
    reported_cells: list[dict[str, Any]] = []
    paper_grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    sensitivity_grouped: dict[
        str,
        dict[str, dict[str, list[float]]],
    ] = {}

    for cell_index, cell_value in enumerate(cells):
        cell = _mapping(cell_value, f"efficiency.cells[{cell_index}]")
        model = str(cell["model"])
        seed = int(cell["seed"])
        automatic = _mapping(
            cell["automatic"],
            f"efficiency.cells[{cell_index}].automatic",
        )
        comparator_values = _mapping(
            cell["comparators"],
            f"efficiency.cells[{cell_index}].comparators",
        )
        reported_comparators: dict[str, Any] = {}
        for comparator_name in ("fisher_low_rank", "welford_diag"):
            comparator = _mapping(
                comparator_values[comparator_name],
                f"{model}/{seed}/{comparator_name}",
            )
            historical = _mapping(
                comparator["historical_nominal_B_sensitivity"],
                f"{model}/{seed}/{comparator_name}/historical",
            )
            population = _mapping(
                comparator["equal_split_population"],
                f"{model}/{seed}/{comparator_name}/population",
            )
            automatic_marginal = _sequence(
                automatic["marginal_per_chain"],
                f"{model}/{seed}/automatic marginal per chain",
            )
            population_marginal = _sequence(
                population["marginal_per_chain"],
                f"{model}/{seed}/{comparator_name}/population marginal per chain",
            )
            if len(automatic_marginal) != len(population_marginal):
                raise ReportError(
                    f"marginal chain counts differ for "
                    f"{model}/{seed}/{comparator_name}"
                )
            marginal_ratios = [
                _ratio(
                    numerator,
                    denominator,
                    f"{model}/{seed}/{comparator_name}/marginal chain {chain}",
                )
                for chain, (numerator, denominator) in enumerate(
                    zip(
                        automatic_marginal,
                        population_marginal,
                        strict=True,
                    )
                )
            ]
            nominal_B_chain0_sensitivity = _ratio(
                automatic["marginal_amortized_chain0"],
                historical["marginal_amortized_chain0"],
                f"{model}/{seed}/{comparator_name}/nominal-B chain0 sensitivity",
            )
            one_output_ratio = _ratio(
                automatic["one_output_total"],
                historical["one_output_total"],
                f"{model}/{seed}/{comparator_name}/one-output",
            )
            pooled_ratio: float | None
            if comparator.get("pooled_ratio_reportable") is True:
                pooled_ratio = _ratio(
                    automatic["pooled_population_total"],
                    population["pooled_population_total"],
                    f"{model}/{seed}/{comparator_name}/pooled",
                )
                stored_ratio = _number(
                    comparator["automatic_to_manual_pooled_ratio"],
                    f"{model}/{seed}/{comparator_name}/stored pooled ratio",
                )
                if not math.isclose(
                    pooled_ratio,
                    stored_ratio,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ReportError(
                        f"stored pooled ratio is inconsistent for "
                        f"{model}/{seed}/{comparator_name}"
                    )
            else:
                pooled_ratio = None

            paper_ratios = {
                "pooled_population_total_over_equal_split_population": (
                    pooled_ratio
                ),
                "marginal_per_chain_over_equal_split_population": (
                    marginal_ratios
                ),
                "one_output_total_over_historical_single_chain": (
                    one_output_ratio
                ),
            }
            reported_comparators[comparator_name] = {
                "role": comparator["role"],
                "paper_ratio_components": paper_ratios,
                "nonpaper_nominal_B_sensitivity": {
                    "marginal_amortized_chain0_over_historical_nominal_B": (
                        nominal_B_chain0_sensitivity
                    )
                },
                "historical_nominal_B_sensitivity": historical,
                "equal_split_population": population,
            }
            paper_bucket = paper_grouped.setdefault(model, {}).setdefault(
                comparator_name,
                {
                    "pooled_population_total_over_equal_split_population": [],
                    "marginal_per_chain_over_equal_split_population": [],
                    "one_output_total_over_historical_single_chain": [],
                },
            )
            if pooled_ratio is not None:
                paper_bucket[
                    "pooled_population_total_over_equal_split_population"
                ].append(pooled_ratio)
            paper_bucket[
                "marginal_per_chain_over_equal_split_population"
            ].extend(marginal_ratios)
            paper_bucket[
                "one_output_total_over_historical_single_chain"
            ].append(one_output_ratio)
            sensitivity_grouped.setdefault(model, {}).setdefault(
                comparator_name,
                {
                    "marginal_amortized_chain0_over_historical_nominal_B": []
                },
            )[
                "marginal_amortized_chain0_over_historical_nominal_B"
            ].append(nominal_B_chain0_sensitivity)

        reported_cells.append(
            {
                "model": model,
                "seed": seed,
                "automatic": automatic,
                "comparators": reported_comparators,
            }
        )

    paper_ratios_by_model: dict[str, Any] = {}
    for model, comparator_values in sorted(paper_grouped.items()):
        paper_ratios_by_model[model] = {}
        for comparator_name, ratio_values in sorted(comparator_values.items()):
            paper_ratios_by_model[model][comparator_name] = {
                "geometric_mean": {
                    ratio_name: _geometric_mean(
                        values,
                        f"{model}/{comparator_name}/{ratio_name}",
                    )
                    for ratio_name, values in ratio_values.items()
                },
                "range": {
                    ratio_name: _range(
                        values,
                        f"{model}/{comparator_name}/{ratio_name}",
                    )
                    for ratio_name, values in ratio_values.items()
                },
                "component_count": {
                    ratio_name: len(values)
                    for ratio_name, values in ratio_values.items()
                },
            }
    nonpaper_sensitivity_ranges: dict[str, Any] = {}
    for model, comparator_values in sorted(sensitivity_grouped.items()):
        nonpaper_sensitivity_ranges[model] = {}
        for comparator_name, ratio_values in sorted(comparator_values.items()):
            nonpaper_sensitivity_ranges[model][comparator_name] = {
                ratio_name: _range(
                    values,
                    f"{model}/{comparator_name}/{ratio_name}",
                )
                for ratio_name, values in ratio_values.items()
            }

    return {
        "selection_policy": efficiency["selection_policy"],
        "estimand_definitions": efficiency["estimand_definitions"],
        "primary_reporting_gate": efficiency["primary_reporting_gate"],
        "cell_count": len(reported_cells),
        "paper_ratio_definitions": {
            "pooled_population_total_over_equal_split_population": (
                "geometric mean across seeds of automatic pooled-population "
                "ESS/gradient divided by equal-split-population ESS/gradient"
            ),
            "marginal_per_chain_over_equal_split_population": (
                "geometric mean across every seed-by-chain elementwise ratio "
                "of automatic to equal-split-population marginal ESS/gradient"
            ),
            "one_output_total_over_historical_single_chain": (
                "geometric mean across seeds of automatic one-output-total "
                "ESS/gradient divided by the historical single-chain "
                "one-output-total ESS/gradient"
            ),
        },
        "paper_ratios_by_model": paper_ratios_by_model,
        "nonpaper_nominal_B_sensitivity_ranges": nonpaper_sensitivity_ranges,
        "cells": reported_cells,
    }


def _shared_step_size(summary: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(summary["shared_step_size"], "shared_step_size")
    pairs = _sequence(section["pairs"], "shared_step_size.pairs")
    grouped: dict[str, list[float]] = {}
    for pair_value in pairs:
        pair = _mapping(pair_value, "shared_step_size pair")
        grouped.setdefault(str(pair["model"]), []).append(
            _number(
                pair["warmup_grad_ratio_historical_over_current"],
                "warmup gradient ratio",
            )
        )
    all_values = [value for values in grouped.values() for value in values]
    return {
        "comparison_scope": section["comparison_scope"],
        "warmup_grad_ratio_historical_over_current_range": _range(
            all_values,
            "shared-step-size warmup ratios",
        ),
        "range_by_model": {
            model: _range(values, f"{model} warmup ratios")
            for model, values in sorted(grouped.items())
        },
        "pairs": pairs,
    }


def _arm_by_id(
    controlled_gmm: Mapping[str, Any],
    arm_id: str,
) -> Mapping[str, Any]:
    arms = _sequence(controlled_gmm["arms"], "controlled_gmm.arms")
    matches = [
        _mapping(arm, "controlled_gmm arm")
        for arm in arms
        if isinstance(arm, Mapping) and arm.get("arm_id") == arm_id
    ]
    if len(matches) != 1:
        raise ReportError(f"expected exactly one GMM arm {arm_id!r}")
    return matches[0]


def _gmm_regime_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sr: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("error") is not None:
            raise ReportError("primary GMM regime table contains an error row")
        by_sr.setdefault(_number(row["SR"], "GMM SR"), []).append(row)

    table: list[dict[str, Any]] = []
    for sr, cells in sorted(by_sr.items()):
        within = [_number(cell["within_lam1"], "within_lam1") for cell in cells]
        split_rhat = [_number(cell["split_rhat"], "split_rhat") for cell in cells]
        route_counts: dict[str, int] = {}
        for cell in cells:
            route = str(cell["route"])
            route_counts[route] = route_counts.get(route, 0) + 1
        table.append(
            {
                "SR": sr,
                "route_counts": dict(sorted(route_counts.items())),
                "deferred_count": sum(
                    cell.get("deferred_to_ensemble") is True for cell in cells
                ),
                "persistent_disagreement_count": sum(
                    cell.get("observed_ensemble_evidence")
                    == "persistent_disagreement_signal"
                    for cell in cells
                ),
                "population_handoff_count": sum(
                    cell.get("handoff") == "population" for cell in cells
                ),
                "within_lam1_mean": statistics.fmean(within),
                "within_lam1_range": [min(within), max(within)],
                "split_rhat_mean": statistics.fmean(split_rhat),
                "split_rhat_range": [min(split_rhat), max(split_rhat)],
            }
        )
    return table


def _first_majority_defer(by_sr: list[Any]) -> float | None:
    for value in by_sr:
        row = _mapping(value, "controlled_gmm.by_sr row")
        route_counts = _mapping(row["route_counts"], "GMM route counts")
        n_cells = sum(int(count) for count in route_counts.values())
        if n_cells < 1:
            raise ReportError("GMM route-count row is empty")
        if int(row["deferred_count"]) >= math.ceil(n_cells / 2):
            return _number(row["SR"], "GMM SR")
    return None


def _matched_diagonal_report(arm: Mapping[str, Any]) -> dict[str, Any]:
    projection = _mapping(
        arm["projection_bulk_ess_per_grad_ratio"],
        "matched-diagonal projection ratios",
    )
    cells = _sequence(projection["cells"], "matched-diagonal cells")
    transformed: list[dict[str, Any]] = []
    for value in cells:
        cell = _mapping(value, "matched-diagonal cell")
        ratio = _positive(cell["ratio"], "matched-diagonal ratio")
        transformed.append(
            {
                **cell,
                "percent_change_low_rank_over_diagonal": 100.0 * (ratio - 1.0),
            }
        )
    return {
        "rows": arm["rows"],
        "median": projection["median"],
        "range": projection["range"],
        "cells": transformed,
    }


def _k3_cluster_report(arm: Mapping[str, Any]) -> dict[str, Any]:
    projection = _mapping(
        arm["projection_bulk_ess_per_grad_ratio"],
        "k3 projection ratios",
    )
    cells = [
        _mapping(cell, "k3 matched-diagonal cell")
        for cell in _sequence(projection["cells"], "k3 matched-diagonal cells")
    ]
    log_ratios = [
        math.log(_positive(cell["ratio"], "k3 projection ratio"))
        for cell in cells
    ]
    seeds = sorted({int(cell["seed"]) for cell in cells})
    srs = sorted({_number(cell["SR"], "k3 SR") for cell in cells})
    cluster_logs: list[float] = []
    for seed in seeds:
        values = [
            math.log(_positive(cell["ratio"], "k3 projection ratio"))
            for cell in cells
            if int(cell["seed"]) == seed
        ]
        if len(values) != len(srs):
            raise ReportError(f"k3 seed {seed} does not have one row per SR")
        cluster_logs.append(statistics.fmean(values))
    if len(cluster_logs) < 2:
        raise ReportError("k3 clustered interval requires at least two seeds")

    try:
        from scipy.stats import t
    except ImportError as exc:  # pragma: no cover - locked environment includes scipy
        raise ReportError("scipy is required for the clustered t interval") from exc

    mean_log = statistics.fmean(cluster_logs)
    standard_error = statistics.stdev(cluster_logs) / math.sqrt(len(cluster_logs))
    critical = float(t.ppf(0.975, len(cluster_logs) - 1))
    interval = [
        math.exp(mean_log - critical * standard_error),
        math.exp(mean_log + critical * standard_error),
    ]
    per_sr = {
        str(sr): math.exp(
            statistics.fmean(
                math.log(_positive(cell["ratio"], "k3 projection ratio"))
                for cell in cells
                if _number(cell["SR"], "k3 SR") == sr
            )
        )
        for sr in srs
    }
    return {
        "estimand": (
            "geometric mean of low-rank/matched-diagonal projection bulk "
            "ESS per gradient ratios"
        ),
        "overall_geometric_mean": math.exp(statistics.fmean(log_ratios)),
        "geometric_mean_by_SR": per_sr,
        "seed_clustered_log_t_interval_95": interval,
        "cluster_count": len(cluster_logs),
        "degrees_of_freedom": len(cluster_logs) - 1,
        "log_scale_standard_error": standard_error,
    }


def _k3_matched_raw_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("error") is None]
    if len(valid) != len(rows):
        raise ReportError("k3 matched-diagonal output contains an error row")
    by_sr: dict[float, list[Mapping[str, Any]]] = {}
    for row in valid:
        by_sr.setdefault(_number(row["SR"], "k3 matched SR"), []).append(row)
    valid_profiles_by_sr = {
        str(sr): sum(
            row.get("auto_route") == "low_rank"
            and row.get("auto_imm_kind") == "low_rank"
            for row in cells
        )
        for sr, cells in sorted(by_sr.items())
    }
    ratios = [
        _positive(
            row["projection_bulk_ess_per_grad_ratio"],
            "k3 projection ratio",
        )
        for row in valid
    ]
    split_rhats = [
        _number(row[key], f"k3 {key}")
        for row in valid
        for key in ("lr_projection_split_rhat", "diag_projection_split_rhat")
    ]
    return {
        "design": {
            "correlated_axes": sorted(
                {int(row["correlated_axes"]) for row in valid}
            ),
            "chains": sorted({int(row["M"]) for row in valid}),
            "warmup_gradient_budget": sorted(
                {int(row["budget"]) for row in valid}
            ),
            "post_warmup_draws_per_chain": sorted(
                {int(row["n_sample_draws"]) for row in valid}
            ),
            "separation_ratios": sorted(by_sr),
            "seeds": sorted({int(row["seed"]) for row in valid}),
        },
        "completed": len(valid),
        "total": len(rows),
        "valid_low_rank_profiles_by_SR": valid_profiles_by_sr,
        "paired_ratio_above_one": sum(value > 1.0 for value in ratios),
        "paired_ratio_below_one": sum(value < 1.0 for value in ratios),
        "paired_ratio_equal_one": sum(value == 1.0 for value in ratios),
        "low_rank_divergences_total": sum(
            int(row["lr_num_divergences"]) for row in valid
        ),
        "matched_diagonal_divergences_total": sum(
            int(row["diag_num_divergences"]) for row in valid
        ),
        "maximum_projection_split_rhat": max(split_rhats),
    }


def _persistent_suffix_report(
    table: list[dict[str, Any]],
) -> dict[str, Any]:
    first_index: int | None = None
    for index, row in enumerate(table):
        n_cells = sum(int(count) for count in row["route_counts"].values())
        if all(
            later["persistent_disagreement_count"]
            == sum(int(count) for count in later["route_counts"].values())
            for later in table[index:]
        ):
            first_index = index
            break
    if first_index is None:
        return {
            "first_SR_with_all_seeds_persistent": None,
            "population_handoff_counts_from_that_SR": [],
        }
    suffix = table[first_index:]
    return {
        "first_SR_with_all_seeds_persistent": suffix[0]["SR"],
        "population_handoff_counts_from_that_SR": [
            {
                "SR": row["SR"],
                "population_handoff_count": row["population_handoff_count"],
                "total": sum(
                    int(count) for count in row["route_counts"].values()
                ),
            }
            for row in suffix
        ],
    }


def _controlled_gmm(
    summary: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    section = _mapping(summary["controlled_gmm"], "controlled_gmm")
    arms = _sequence(section["arms"], "controlled_gmm.arms")
    k2_primary = _arm_by_id(section, "gmm_k2_primary_60k")
    k2_matched = _arm_by_id(section, "gmm_k2_matched_diagonal_60k")
    k3_matched = _arm_by_id(section, "gmm_k3_matched_diagonal_60k")
    primary_rows = _load_jsonl(
        result_dir / "gmm" / "gmm_k2_primary_60k.jsonl"
    )
    k3_primary_rows = _load_jsonl(
        result_dir / "gmm" / "gmm_k3_primary_60k.jsonl"
    )
    k3_matched_rows = _load_jsonl(
        result_dir / "gmm" / "gmm_k3_matched_diagonal_60k.jsonl"
    )
    k2_regime_table = _gmm_regime_table(primary_rows)
    k3_regime_table = _gmm_regime_table(k3_primary_rows)

    majority_defer_onset: dict[str, float | None] = {}
    for arm_value in arms:
        arm = _mapping(arm_value, "controlled_gmm arm")
        if arm.get("kind") == "primary":
            majority_defer_onset[str(arm["arm_id"])] = _first_majority_defer(
                _sequence(arm["by_sr"], f"{arm['arm_id']}.by_sr")
            )

    return {
        "global_exploration_policy": section["global_exploration_policy"],
        "validation": section["validation"],
        "first_SR_with_at_least_half_seeds_deferred": majority_defer_onset,
        "k2_primary_60k_regime_table": k2_regime_table,
        "k2_single_chain_60k": _arm_by_id(
            section,
            "gmm_k2_single_chain_60k",
        ),
        "k2_matched_diagonal_60k": _matched_diagonal_report(k2_matched),
        "k3_matched_diagonal_60k": {
            "stored_summary": _matched_diagonal_report(k3_matched),
            "clustered_report": _k3_cluster_report(k3_matched),
            "raw_completion_report": _k3_matched_raw_report(k3_matched_rows),
        },
        "k3_primary_60k_regime_table": k3_regime_table,
        "k3_persistent_suffix": _persistent_suffix_report(k3_regime_table),
        "arms": arms,
        "k2_primary_hash": k2_primary["sha256"],
    }


def _ratio_contrasts(
    rows: list[Any],
    *,
    log_key: str,
    ratio_name: str,
    percent_name: str,
) -> list[dict[str, Any]]:
    reported: list[dict[str, Any]] = []
    for value in rows:
        row = _mapping(value, "log-ratio contrast")
        log_ratio = _number(row[log_key], log_key)
        reported.append(
            {
                **row,
                ratio_name: math.exp(log_ratio),
                percent_name: 100.0 * math.expm1(log_ratio),
            }
        )
    return reported


def _schedule_configuration(summary: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(
        summary["schedule_configuration"],
        "schedule_configuration",
    )
    contrasts = _ratio_contrasts(
        _sequence(
            section["predeclared_contrasts"],
            "schedule_configuration.predeclared_contrasts",
        ),
        log_key="log_ess_per_grad_ratio",
        ratio_name="ess_per_grad_ratio",
        percent_name="ess_per_grad_percent_change",
    )
    return {
        "interpretation": section["interpretation"],
        "stored_event_rows": section["stored_event_rows"],
        "predeclared_contrasts": contrasts,
        "cells": section["cells"],
    }


def _restart_ablation(summary: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(summary["restart_ablation"], "restart_ablation")
    pairs = _ratio_contrasts(
        _sequence(section["pairs"], "restart_ablation.pairs"),
        log_key="log_ess_per_grad_ratio_continuous_over_reseed",
        ratio_name="ess_per_grad_ratio_continuous_over_reseed",
        percent_name="ess_per_grad_percent_change_continuous_over_reseed",
    )
    return {
        "claim_policy": section["claim_policy"],
        "pairs": pairs,
    }


def _kernel_family(summary: Mapping[str, Any]) -> dict[str, Any]:
    section = _mapping(summary["kernel_family"], "kernel_family")
    cells = _sequence(section["cells"], "kernel_family.cells")
    ratios_by_algorithm: dict[str, list[float]] = {}
    route_counts: dict[str, int] = {}
    automatic_divergences = 0
    manual_divergences = 0
    for value in cells:
        cell = _mapping(value, "kernel-family cell")
        algorithm = str(cell["algorithm"])
        route = str(cell["route"])
        ratios_by_algorithm.setdefault(algorithm, []).append(
            _positive(
                cell["automatic_to_manual_ess_per_grad_ratio"],
                "kernel-family ratio",
            )
        )
        route_counts[route] = route_counts.get(route, 0) + 1
        automatic_divergences += int(cell["automatic_divergences"])
        manual_divergences += int(cell["manual_divergences"])
    return {
        "cell_count": len(cells),
        "route_counts": dict(sorted(route_counts.items())),
        "automatic_divergences_total": automatic_divergences,
        "manual_divergences_total": manual_divergences,
        "automatic_to_manual_ratio_range_by_algorithm": {
            algorithm: _range(values, f"{algorithm} ratios")
            for algorithm, values in sorted(ratios_by_algorithm.items())
        },
        "cells": cells,
    }


def _figures(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frozen_summary_artifacts": summary["figures"],
        "gmm_money_panel": {
            "summary_keys": [
                "controlled_gmm.validation",
                "controlled_gmm.arms[arm_id=gmm_k2_primary_60k]",
            ],
            "raw_input": "gmm/gmm_k2_primary_60k.jsonl",
            "canonical_output": "../figures/gmm_money_panel.png",
        },
    }


def _fixed_manuscript_headline(
    fixed_efficiency: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    automatic_paths = (
        result_dir / "fixed" / "illcond.jsonl",
        result_dir / "fixed" / "german.jsonl",
    )
    population_paths = (
        result_dir / "fixed" / "manual_population_illcond.jsonl",
        result_dir / "fixed" / "manual_population_german.jsonl",
    )
    historical_paths = (
        result_dir / "fixed" / "manual_illcond.jsonl",
        result_dir / "fixed" / "manual_german.jsonl",
    )
    automatic = [
        row
        for path in automatic_paths
        for row in _load_jsonl(path)
        if row.get("record_type") == "cell"
    ]
    population = [
        row
        for path in population_paths
        for row in _load_jsonl(path)
        if row.get("record_type") == "cell"
    ]
    historical = [
        row
        for path in historical_paths
        for row in _load_jsonl(path)
        if row.get("record_type") == "cell"
    ]
    declared_population = [*automatic, *population]
    passed = sum(
        row.get("population_quality_pass") is True
        for row in declared_population
    )
    warmup_divergences = sum(
        int(
            row.get(
                "warmup_divergences_all_chains",
                row.get("warmup_divergences", 0),
            )
        )
        for row in declared_population
    )
    post_warmup_divergences = sum(
        int(
            row.get(
                "sampling_divergences_all_chains",
                row.get("sampling_divergences", 0),
            )
        )
        for row in declared_population
    )
    ratio_source = _mapping(
        fixed_efficiency["paper_ratios_by_model"],
        "paper_ratios_by_model",
    )
    ratios: dict[str, Any] = {}
    for model, comparator_values in ratio_source.items():
        ratios[str(model)] = {}
        for comparator, report_value in _mapping(
            comparator_values,
            f"paper ratios for {model}",
        ).items():
            report = _mapping(
                report_value,
                f"paper ratios for {model}/{comparator}",
            )
            ratios[str(model)][str(comparator)] = report["geometric_mean"]
    return {
        "automatic_low_rank_routes": {
            "count": sum(row.get("route") == "low_rank" for row in automatic),
            "total": len(automatic),
        },
        "declared_population_quality_gates": {
            "passed": passed,
            "total": len(declared_population),
        },
        "warmup_divergences": warmup_divergences,
        "post_warmup_divergences": post_warmup_divergences,
        "comparator_warmup_design": {
            "equal_split_population_steps_per_chain": sorted(
                {
                    int(row["num_warmup_steps_per_chain"])
                    for row in population
                }
            ),
            "historical_single_chain_steps": sorted(
                {int(row["num_warmup_steps_per_chain"]) for row in historical}
            ),
        },
        "geometric_mean_automatic_over_comparator_ratios": ratios,
    }


def _shared_step_manuscript_headline(
    shared_step_size: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "warmup_gradient_ratio_historical_over_current_by_model": (
            shared_step_size["range_by_model"]
        )
    }


def _gmm_manuscript_headline(
    controlled_gmm: Mapping[str, Any],
) -> dict[str, Any]:
    onset = _mapping(
        controlled_gmm[
            "first_SR_with_at_least_half_seeds_deferred"
        ],
        "GMM majority onset",
    )
    single = _mapping(
        controlled_gmm["k2_single_chain_60k"],
        "k2 single-chain report",
    )
    single_cells = _sequence(single["cells"], "k2 single-chain cells")
    k2_matched = _mapping(
        controlled_gmm["k2_matched_diagonal_60k"],
        "k2 matched-diagonal report",
    )
    k3_matched = _mapping(
        controlled_gmm["k3_matched_diagonal_60k"],
        "k3 matched-diagonal report",
    )
    return {
        "k2": {
            "primary_regime_table": controlled_gmm[
                "k2_primary_60k_regime_table"
            ],
            "majority_population_handoff_onset_SR_by_budget": {
                "20k": onset["gmm_k2_budget_20k"],
                "60k": onset["gmm_k2_primary_60k"],
                "120k": onset["gmm_k2_budget_120k"],
            },
            "single_chain": [
                {
                    "SR": cell["SR"],
                    "route": cell["route"],
                    "mode_weight_est": cell["mode_weight_est"],
                }
                for cell in single_cells
            ],
            "matched_diagonal_low_rank_over_diagonal_ratios": [
                {
                    "SR": cell["SR"],
                    "ratio": cell["ratio"],
                }
                for cell in _sequence(
                    k2_matched["cells"],
                    "k2 matched-diagonal cells",
                )
            ],
        },
        "k3": {
            "primary_persistent_suffix": controlled_gmm[
                "k3_persistent_suffix"
            ],
            "matched_diagonal": {
                "clustered_report": k3_matched["clustered_report"],
                "completion_report": k3_matched["raw_completion_report"],
            },
        },
    }


def _restart_manuscript_headline(
    restart_ablation: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = [
        _mapping(value, "restart pair")
        for value in _sequence(restart_ablation["pairs"], "restart pairs")
    ]
    ratios = [
        _positive(
            pair["ess_per_grad_ratio_continuous_over_reseed"],
            "restart ratio",
        )
        for pair in pairs
    ]
    by_model: dict[str, list[float]] = {}
    for pair, ratio in zip(pairs, ratios, strict=True):
        by_model.setdefault(str(pair["model"]), []).append(ratio)
    return {
        "continuous_wins": sum(ratio > 1.0 for ratio in ratios),
        "total_pairs": len(ratios),
        "continuous_over_reseed_ratio_range": _range(
            ratios,
            "restart ratios",
        ),
        "continuous_over_reseed_geometric_mean_by_model": {
            model: _geometric_mean(values, f"{model} restart ratios")
            for model, values in sorted(by_model.items())
        },
    }


def _kernel_manuscript_headline(
    kernel_family: Mapping[str, Any],
) -> dict[str, Any]:
    cells = [
        _mapping(value, "kernel cell")
        for value in _sequence(kernel_family["cells"], "kernel cells")
    ]
    multinomial = [
        cell for cell in cells if cell["algorithm"] == "multinomial_hmc"
    ]
    return {
        "low_rank_routes": sum(cell["route"] == "low_rank" for cell in cells),
        "total_cells": len(cells),
        "automatic_post_warmup_divergences": sum(
            int(cell["automatic_divergences"]) for cell in cells
        ),
        "multinomial_losses_to_comparator": sum(
            _positive(
                cell["automatic_to_manual_ess_per_grad_ratio"],
                "multinomial ratio",
            )
            < 1.0
            for cell in multinomial
        ),
        "multinomial_cells": len(multinomial),
    }


def _schedule_manuscript_headline(
    schedule_configuration: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    cells = [
        _mapping(value, "schedule cell")
        for value in _sequence(
            schedule_configuration["cells"],
            "schedule cells",
        )
    ]
    failing_cells = [
        cell
        for cell in cells
        if cell["model"] == "radon"
        and cell["schedule_family"] == "proportional_growing"
        and cell["buffer_policy"] == "accumulating"
        and int(cell["recompute_every"]) == 1
    ]
    contrasts = [
        _mapping(value, "schedule contrast")
        for value in _sequence(
            schedule_configuration["predeclared_contrasts"],
            "schedule contrasts",
        )
    ]
    relevant_contrasts = [
        contrast
        for contrast in contrasts
        if contrast["model"] == "radon"
        and (
            (
                contrast["contrast"]
                == "proportional_growing_over_stan_doubling"
                and contrast.get("buffer_policy") == "accumulating"
            )
            or (
                contrast["contrast"] == "accumulating_over_reset"
                and contrast.get("schedule_family")
                == "proportional_growing"
            )
        )
    ]
    gmm_rows = _load_jsonl(
        result_dir / "gmm" / "gmm_k2_primary_60k.jsonl"
    )
    if not gmm_rows:
        raise ReportError("primary GMM file is empty")
    prescriptions = _mapping(
        gmm_rows[0]["schedule_prescriptions"],
        "GMM schedule prescriptions",
    )
    fisher = _mapping(
        prescriptions["seyboldt_fisher_hmc"],
        "Fisher-HMC schedule prescription",
    )
    fisher_phases = [
        _mapping(value, "Fisher-HMC phase")
        for value in _sequence(fisher["phases"], "Fisher-HMC phases")
    ]
    return {
        "fisher_hmc_reference_phase_fractions": [
            _number(phase["target_fraction"], "Fisher-HMC target fraction")
            for phase in fisher_phases
        ],
        "radon_failing_configuration": {
            "schedule_family": "proportional_growing",
            "buffer_policy": "accumulating",
            "recompute_every": 1,
            "cells": [
                {
                    "seed": cell["seed"],
                    "ess_per_grad": cell["ess_per_grad"],
                    "warmup_grads": cell["warmup_grads"],
                    "warmup_divergences": cell["warmup_divergences"],
                    "sampling_divergences": cell["sampling_divergences"],
                }
                for cell in failing_cells
            ],
            "predeclared_contrasts": [
                {
                    key: contrast[key]
                    for key in (
                        "contrast",
                        "seed",
                        "ess_per_grad_ratio",
                        "ess_per_grad_percent_change",
                        "warmup_grad_ratio",
                        "divergence_delta",
                    )
                }
                for contrast in relevant_contrasts
            ],
        }
    }


def _manuscript_headlines(
    *,
    fixed_efficiency: Mapping[str, Any],
    shared_step_size: Mapping[str, Any],
    controlled_gmm: Mapping[str, Any],
    schedule_configuration: Mapping[str, Any],
    restart_ablation: Mapping[str, Any],
    kernel_family: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    return {
        "fixed_efficiency": _fixed_manuscript_headline(
            fixed_efficiency,
            result_dir,
        ),
        "shared_step_size": _shared_step_manuscript_headline(
            shared_step_size
        ),
        "controlled_gmm": _gmm_manuscript_headline(controlled_gmm),
        "schedule_configuration": _schedule_manuscript_headline(
            schedule_configuration,
            result_dir,
        ),
        "restart_ablation": _restart_manuscript_headline(restart_ablation),
        "kernel_family": _kernel_manuscript_headline(kernel_family),
    }


def build_report(
    summary_path: Path,
    result_dir: Path,
    attestation_path: Path | None = None,
) -> dict[str, Any]:
    """Build and authenticate the complete deterministic report."""

    summary_path = summary_path.resolve()
    if attestation_path is None:
        attestation_path = summary_path.with_name("validation_attestation.json")
    summary, _, trusted_reference = _load_authenticated_reference(
        summary_path,
        attestation_path,
        result_dir,
    )
    verification = _verify_inputs(summary, result_dir)
    verification["trusted_reference"] = trusted_reference
    fixed_efficiency = _fixed_efficiency(summary)
    shared_step_size = _shared_step_size(summary)
    controlled_gmm = _controlled_gmm(summary, result_dir)
    schedule_configuration = _schedule_configuration(summary)
    restart_ablation = _restart_ablation(summary)
    kernel_family = _kernel_family(summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "source_run_id": summary["run_id"],
        "number_source_policy": _mapping(
            summary["traceability"],
            "traceability",
        )["number_source_policy"],
        "source_map": {
            section: {
                "frozen_summary_keys": list(SUMMARY_KEYS[section]),
                "raw_paths": list(RAW_INPUTS[section]),
            }
            for section in SECTIONS
        },
        "verification": verification,
        "manuscript_headlines": _manuscript_headlines(
            fixed_efficiency=fixed_efficiency,
            shared_step_size=shared_step_size,
            controlled_gmm=controlled_gmm,
            schedule_configuration=schedule_configuration,
            restart_ablation=restart_ablation,
            kernel_family=kernel_family,
            result_dir=result_dir,
        ),
        "fixed_efficiency": fixed_efficiency,
        "shared_step_size": shared_step_size,
        "controlled_gmm": controlled_gmm,
        "schedule_configuration": schedule_configuration,
        "restart_ablation": restart_ablation,
        "kernel_family": kernel_family,
        "figures": _figures(summary),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--attestation",
        type=Path,
        help=(
            "Validation attestation to authenticate. Defaults to "
            "validation_attestation.json beside --summary."
        ),
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--section",
        choices=(
            "source_map",
            "verification",
            "manuscript_headlines",
            *SECTIONS,
        ),
        help="Print one report section instead of the complete report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.summary,
            args.result_dir.resolve(),
            args.attestation,
        )
    except (OSError, json.JSONDecodeError, ReportError) as exc:
        print(f"paper-number report failed: {exc}", file=sys.stderr)
        return 1
    value: Any = report if args.section is None else report[args.section]
    json.dump(value, sys.stdout, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
