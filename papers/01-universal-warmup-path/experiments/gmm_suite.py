"""Canonical cell definitions and validation for the controlled-GMM artifact."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from gmm_boundary import analytic_diagnostic, verify_invariance


EXPECTED_BLACKJAX_SHA = "29d2468857be4de1644ca4470c2a4aa7f8137656"
PRIMARY_SRS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    9.5,
    10.0,
)
PRIMARY_SEEDS = (42, 123, 456)


@dataclass(frozen=True)
class ArmSpec:
    """One independently persisted experiment arm."""

    arm_id: str
    kind: Literal["primary", "single_chain", "matched_diagonal"]
    filename: str
    correlated_axes: int
    budget: int
    draws: int
    srs: tuple[float, ...]
    seeds: tuple[int, ...]
    init_kind: str = "broad"
    chains: int = 8

    @property
    def expected_rows(self) -> int:
        return len(self.srs) * len(self.seeds)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def full_arm_specs() -> tuple[ArmSpec, ...]:
    """Return the frozen cell grid used by the manuscript's GMM claims."""
    return (
        ArmSpec(
            "gmm_k2_primary_60k",
            "primary",
            "gmm_k2_primary_60k.jsonl",
            2,
            60_000,
            1_000,
            PRIMARY_SRS,
            PRIMARY_SEEDS,
        ),
        ArmSpec(
            "gmm_k2_budget_20k",
            "primary",
            "gmm_k2_budget_20k.jsonl",
            2,
            20_000,
            1_000,
            PRIMARY_SRS,
            PRIMARY_SEEDS,
        ),
        ArmSpec(
            "gmm_k2_budget_120k",
            "primary",
            "gmm_k2_budget_120k.jsonl",
            2,
            120_000,
            1_000,
            PRIMARY_SRS,
            PRIMARY_SEEDS,
        ),
        ArmSpec(
            "gmm_k2_single_chain_60k",
            "single_chain",
            "gmm_k2_single_chain_60k.jsonl",
            2,
            60_000,
            200,
            (1.5, 5.0, 10.0),
            (42,),
            chains=1,
        ),
        ArmSpec(
            "gmm_k2_matched_diagonal_60k",
            "matched_diagonal",
            "gmm_k2_matched_diagonal_60k.jsonl",
            2,
            60_000,
            1_000,
            (1.5, 4.0, 5.0, 9.0),
            (42,),
        ),
        ArmSpec(
            "gmm_k3_primary_60k",
            "primary",
            "gmm_k3_primary_60k.jsonl",
            3,
            60_000,
            1_000,
            PRIMARY_SRS,
            PRIMARY_SEEDS,
        ),
        ArmSpec(
            "gmm_k3_matched_diagonal_60k",
            "matched_diagonal",
            "gmm_k3_matched_diagonal_60k.jsonl",
            3,
            60_000,
            4_000,
            (3.0, 3.5, 4.0, 4.5),
            tuple(range(1001, 1017)),
        ),
    )


def arm_specs(smoke: bool = False) -> tuple[ArmSpec, ...]:
    """Return full specs, or a downscaled grid exercising every suite arm."""
    specs = full_arm_specs()
    if not smoke:
        return specs

    smoke_specs: list[ArmSpec] = []
    for spec in specs:
        if spec.kind == "primary" and "budget" not in spec.arm_id:
            smoke_specs.append(
                replace(spec, srs=(1.5, 9.0), seeds=(spec.seeds[0],), draws=20)
            )
        elif spec.kind == "primary":
            smoke_specs.append(
                replace(spec, srs=(6.0,), seeds=(spec.seeds[0],), draws=20)
            )
        elif spec.kind == "single_chain":
            smoke_specs.append(
                replace(spec, srs=(10.0,), seeds=(spec.seeds[0],), draws=20)
            )
        else:
            smoke_specs.append(
                replace(
                    spec,
                    srs=(spec.srs[0],),
                    seeds=(spec.seeds[0],),
                    draws=20,
                )
            )
    return tuple(smoke_specs)


class ValidationError(RuntimeError):
    """Raised when raw experiment output is incomplete or malformed."""


_SEMANTIC_FIELDS = {
    "metric_route_status",
    "metric_route_basis",
    "metric_scope",
    "observed_ensemble_evidence",
    "global_exploration",
    "handoff",
    "confidence_scope",
}
_EVENT_FIELDS = {"window_events", "schedule_prescriptions"}
_PRIMARY_REQUIRED = {
    "route",
    "effective_rank",
    "deferred_to_ensemble",
    "detection_branch",
    "within_lam1",
    "chain_consistency_psi",
    "chain_collinearity",
    "unimodality_gate",
    "projected_within_variance",
    "projected_between_variance",
    "projected_total_variance",
    "projected_target_variance",
    "projected_partition_error",
    "projected_total_reference_error",
    "min_ess_per_grad",
    "split_rhat",
    "mode_weight_est",
    "both_modes_visited_frac",
    "num_divergences",
} | _SEMANTIC_FIELDS | _EVENT_FIELDS
_ABLATION_REQUIRED = {
    "auto_route",
    "auto_effective_rank",
    "auto_imm_kind",
    "lr_min_ess_per_grad",
    "lr_projection_bulk_ess_per_grad",
    "lr_projection_tail_ess_per_grad",
    "lr_total_integration_steps",
    "lr_num_divergences",
    "lr_both_modes_frac",
    "lr_split_rhat",
    "lr_projection_split_rhat",
    "diag_min_ess_per_grad",
    "diag_projection_bulk_ess_per_grad",
    "diag_projection_tail_ess_per_grad",
    "diag_total_integration_steps",
    "diag_num_divergences",
    "diag_both_modes_frac",
    "diag_split_rhat",
    "diag_projection_split_rhat",
    "delta_min_ess_per_grad",
    "projection_bulk_ess_per_grad_ratio",
    "delta_both_modes_frac",
} | _SEMANTIC_FIELDS | _EVENT_FIELDS
_SINGLE_REQUIRED = {
    "route",
    "effective_rank",
    "confidence",
    "mode_coverage",
    "bulk_ess_per_grad",
    "mode_weight_est",
    "both_modes_visited",
    "num_divergences",
} | _SEMANTIC_FIELDS | _EVENT_FIELDS


def _walk_values(value: Any, path: str = "row") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_walk_values(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_walk_values(item, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} is nonfinite")
    return errors


def _expected_key(spec: ArmSpec, sr: float, seed: int) -> tuple[Any, ...]:
    return (
        float(sr),
        int(seed),
        spec.init_kind,
        spec.budget,
        spec.correlated_axes,
        spec.chains,
        spec.draws,
    )


def _row_key(row: dict[str, Any], spec: ArmSpec) -> tuple[Any, ...]:
    chain_field = "n_chains" if spec.kind == "single_chain" else "M"
    return (
        float(row["SR"]),
        int(row["seed"]),
        str(row["init_kind"]),
        int(row["budget"]),
        int(row["correlated_axes"]),
        int(row[chain_field]),
        int(row["n_sample_draws"]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load strict JSONL while retaining the line number in parse errors."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValidationError(
                    f"{path}:{line_number}: row must be a JSON object"
                )
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_arm_file(path: Path, spec: ArmSpec) -> dict[str, Any]:
    """Validate one raw arm and return a compact machine-readable summary."""
    if not path.is_file():
        raise ValidationError(f"missing result file: {path}")

    rows = load_jsonl(path)
    errors: list[str] = []
    if len(rows) != spec.expected_rows:
        errors.append(
            f"row count {len(rows)} != expected {spec.expected_rows}"
        )

    max_err, spike, _ = verify_invariance(spec.srs, spec.correlated_axes)
    expected_spike = analytic_diagnostic(spec.correlated_axes)[
        "marginal_whitened_spike"
    ]
    if max_err >= 1e-12:
        errors.append(f"analytic invariance error {max_err:.3e} >= 1e-12")
    if abs(spike - float(expected_spike)) >= 1e-12:
        errors.append(
            f"analytic spike {spike:.16g} != expected {expected_spike:.16g}"
        )

    expected = {
        _expected_key(spec, sr, seed)
        for seed in spec.seeds
        for sr in spec.srs
    }
    seen: dict[tuple[Any, ...], int] = {}
    required = {
        "primary": _PRIMARY_REQUIRED,
        "matched_diagonal": _ABLATION_REQUIRED,
        "single_chain": _SINGLE_REQUIRED,
    }[spec.kind]

    for line_number, row in enumerate(rows, start=1):
        prefix = f"{path.name}:{line_number}"
        errors.extend(f"{prefix}: {message}" for message in _walk_values(row))
        if row.get("arm_id") != spec.arm_id:
            errors.append(
                f"{prefix}: arm_id {row.get('arm_id')!r} != {spec.arm_id!r}"
            )
        if row.get("error") is not None:
            errors.append(f"{prefix}: experiment error: {row['error']}")
        missing_fields = sorted(required - row.keys())
        if missing_fields:
            errors.append(f"{prefix}: missing fields {missing_fields}")
        null_fields = sorted(
            field for field in required if field in row and row[field] is None
        )
        if null_fields:
            errors.append(f"{prefix}: null required fields {null_fields}")
        if row.get("global_exploration") != "not_established":
            errors.append(
                f"{prefix}: global_exploration must be 'not_established'"
            )
        if row.get("window_events") is not None:
            from window_events import validate_window_events

            event_errors = validate_window_events(
                row["window_events"],
                row.get("schedule_prescriptions"),
            )
            errors.extend(
                f"{prefix}: {event_error}" for event_error in event_errors
            )
        if spec.kind == "single_chain" and row.get("split_rhat") is not None:
            errors.append(f"{prefix}: split_rhat must be null for one chain")
        if spec.kind == "primary" and row.get("error") is None:
            partition_error = row.get("projected_partition_error")
            target_variance = row.get("projected_target_variance")
            if (
                not isinstance(partition_error, (int, float))
                or abs(float(partition_error)) > 1e-10
            ):
                errors.append(
                    f"{prefix}: projected partition invariant failed"
                )
            if (
                not isinstance(target_variance, (int, float))
                or abs(float(target_variance) - 22.0) > 1e-10
            ):
                errors.append(
                    f"{prefix}: projected target variance invariant failed"
                )
        try:
            key = _row_key(row, spec)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: invalid cell key: {exc}")
            continue
        seen[key] = seen.get(key, 0) + 1

    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate cells: {duplicates}")
    actual = set(seen)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"missing cells: {missing}")
    if unexpected:
        errors.append(f"unexpected cells: {unexpected}")

    if errors:
        raise ValidationError("\n".join(errors))
    return {
        "arm_id": spec.arm_id,
        "filename": spec.filename,
        "rows": len(rows),
        "analytic_invariance_max_error": max_err,
        "analytic_marginal_spike": spike,
        "status": "valid",
    }


def validate_suite_dir(
    result_dir: Path,
    *,
    smoke: bool = False,
    require_run_manifest: bool = False,
) -> list[dict[str, Any]]:
    """Validate every expected arm and the recorded execution provenance."""
    provenance_path = result_dir / "provenance.json"
    if not provenance_path.is_file():
        raise ValidationError(f"missing result file: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_errors: list[str] = []
    if provenance.get("blackjax", {}).get("commit") != EXPECTED_BLACKJAX_SHA:
        provenance_errors.append("unexpected BlackJAX commit")
    if provenance.get("blackjax", {}).get("dirty"):
        provenance_errors.append("BlackJAX checkout was dirty")
    if provenance.get("jax", {}).get("backend") != "cpu":
        provenance_errors.append("JAX backend was not CPU")
    if provenance.get("jax", {}).get("x64_enabled") is not True:
        provenance_errors.append("JAX x64 was not enabled")
    if provenance_errors:
        raise ValidationError("; ".join(provenance_errors))

    summaries = [
        validate_arm_file(result_dir / spec.filename, spec)
        for spec in arm_specs(smoke)
    ]
    if require_run_manifest:
        run_manifest_path = result_dir / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise ValidationError(f"missing result file: {run_manifest_path}")
        run_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        manifest_errors: list[str] = []
        expected_mode = "smoke" if smoke else "full"
        if run_manifest.get("status") != "complete":
            manifest_errors.append("run manifest is not complete")
        if run_manifest.get("mode") != expected_mode:
            manifest_errors.append("run manifest mode does not match validation mode")
        expected_arm_ids = [spec.arm_id for spec in arm_specs(smoke)]
        manifest_arm_ids = [
            arm.get("arm_id") for arm in run_manifest.get("arms", [])
        ]
        if manifest_arm_ids != expected_arm_ids:
            manifest_errors.append("run manifest arm order/grid is incorrect")
        recorded_hashes = run_manifest.get("result_sha256", {})
        for spec in arm_specs(smoke):
            actual_hash = _sha256(result_dir / spec.filename)
            if recorded_hashes.get(spec.filename) != actual_hash:
                manifest_errors.append(
                    f"checksum mismatch for {spec.filename}"
                )
        if manifest_errors:
            raise ValidationError("; ".join(manifest_errors))
    return summaries
