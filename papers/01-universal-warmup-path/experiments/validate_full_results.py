#!/usr/bin/env python3
"""Strictly validate and summarize a complete Paper 1 result directory.

The validator rejects incomplete grids, duplicate cells, failed rows,
non-finite numeric values, checksum or provenance mismatches, and untraceable
draw artifacts.  Its JSON summary preserves the three efficiency conventions
separately.  Fisher low rank is the predeclared primary manual comparator;
diagonal is always reported as a control and is never selected after seeing
the results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from run_full_corpus import (
    CURRENT_BLACKJAX_SHA,
    FULL_FIXED_BLOCK_COUNTS,
    FULL_SECTION_COUNTS,
    HISTORICAL_BLACKJAX_SHA,
    RELEASE_RUNCAP,
    SCHEMA_VERSION,
    SMOKE_SECTION_COUNTS,
    TUNINGFORK_SHA,
    build_plan,
    sha256_file,
)
from suite_common import sha256_array
from window_events import validate_window_events


class ValidationError(RuntimeError):
    """Raised when one or more release-blocking checks fail."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: top-level value must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        parse_constant=_reject_constant,
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValidationError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValidationError(
                        f"{path}:{line_number}: row must be an object"
                    )
                rows.append(value)
    except OSError as exc:
        raise ValidationError(f"{path}: cannot read JSONL: {exc}") from exc
    if not rows:
        raise ValidationError(f"{path}: empty JSONL")
    return rows


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _walk_nonfinite(value: Any, path: str = "value") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            errors.extend(_walk_nonfinite(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            errors.extend(_walk_nonfinite(item, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} is non-finite")
    return errors


def _require_fields(
    row: Mapping[str, Any],
    fields: Iterable[str],
    *,
    prefix: str,
) -> list[str]:
    missing = sorted(field for field in fields if field not in row)
    null = sorted(field for field in fields if row.get(field) is None)
    errors: list[str] = []
    if missing:
        errors.append(f"{prefix}: missing fields {missing}")
    if null:
        errors.append(f"{prefix}: null required fields {null}")
    return errors


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _relative_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _check_hash_entry(
    root: Path,
    relative: str,
    entry: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    path = _relative_file(root, relative)
    if path is None:
        return [f"unsafe artifact path {relative!r}"]
    if not path.is_file():
        return [f"missing artifact {relative}"]
    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if not _is_sha256(expected_sha):
        errors.append(f"{relative}: invalid recorded SHA-256")
    elif sha256_file(path) != expected_sha:
        errors.append(f"{relative}: SHA-256 mismatch")
    if expected_bytes != path.stat().st_size:
        errors.append(f"{relative}: byte-count mismatch")
    return errors


def _validate_source_hashes(
    source_dir: Path,
    recorded: Any,
    *,
    required_names: Iterable[str],
    prefix: str,
) -> list[str]:
    if not isinstance(recorded, Mapping):
        return [f"{prefix}: source_sha256 must be an object"]
    errors: list[str] = []
    for name in required_names:
        value = recorded.get(name)
        path = source_dir / name
        if not path.is_file():
            errors.append(f"{prefix}: local source {name} is missing")
        elif not _is_sha256(value):
            errors.append(f"{prefix}: source hash for {name} is invalid")
        elif sha256_file(path) != value:
            errors.append(f"{prefix}: source hash mismatch for {name}")
    for name, value in recorded.items():
        if Path(str(name)).name != name or not _is_sha256(value):
            errors.append(f"{prefix}: malformed source hash entry {name!r}")
    return errors


def _validate_provenance(
    row: Mapping[str, Any],
    *,
    expected_suite: str,
    expected_blackjax_sha: str,
    source_dir: Path,
    runner_name: str,
    prefix: str,
) -> list[str]:
    errors = _require_fields(
        row,
        (
            "schema_version",
            "suite",
            "run_id",
            "command",
            "python",
            "jax",
            "jaxlib",
            "numpy",
            "blackjax",
            "tuningfork",
            "jax_enable_x64",
            "devices",
            "source_sha256",
        ),
        prefix=prefix,
    )
    if row.get("record_type") != "provenance":
        errors.append(f"{prefix}: first row is not provenance")
    if row.get("schema_version") != "paper1-experiments-v1":
        errors.append(f"{prefix}: unexpected experiment schema version")
    if row.get("suite") != expected_suite:
        errors.append(f"{prefix}: unexpected suite {row.get('suite')!r}")
    blackjax = row.get("blackjax", {})
    tuningfork = row.get("tuningfork", {})
    if not isinstance(blackjax, Mapping):
        errors.append(f"{prefix}: blackjax provenance must be an object")
    else:
        if blackjax.get("revision") != expected_blackjax_sha:
            errors.append(f"{prefix}: unexpected BlackJAX revision")
        if blackjax.get("dirty") is not False:
            errors.append(f"{prefix}: BlackJAX checkout was dirty")
    if not isinstance(tuningfork, Mapping):
        errors.append(f"{prefix}: tuningfork provenance must be an object")
    else:
        if tuningfork.get("revision") != TUNINGFORK_SHA:
            errors.append(f"{prefix}: unexpected tuningfork revision")
        if tuningfork.get("dirty") is not False:
            errors.append(f"{prefix}: tuningfork checkout was dirty")
    if row.get("jax_enable_x64") is not True:
        errors.append(f"{prefix}: JAX x64 was not enabled")
    devices = row.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append(f"{prefix}: no JAX devices were recorded")
    elif any(device.get("platform") != "cpu" for device in devices):
        errors.append(f"{prefix}: a non-CPU JAX device was used")
    command = row.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"{prefix}: command was not recorded")
    elif any(
        isinstance(value, str) and Path(value).is_absolute()
        for value in command
    ):
        errors.append(f"{prefix}: command contains an absolute path")
    errors.extend(
        _validate_source_hashes(
            source_dir,
            row.get("source_sha256"),
            required_names=("suite_common.py", runner_name),
            prefix=prefix,
        )
    )
    errors.extend(_walk_nonfinite(row, prefix))
    return errors


def _validate_runner_manifest(
    manifest_path: Path,
    *,
    result_path: Path,
    events_path: Path | None,
    expected_cells: int,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    manifest = _load_json(manifest_path)
    if manifest.get("error_cells") != 0:
        errors.append(f"{prefix}: manifest records failed cells")
    manifest_expected = manifest.get(
        "expected_cells", manifest.get("expected_route_cells")
    )
    if manifest_expected != expected_cells:
        errors.append(
            f"{prefix}: manifest expected count {manifest_expected!r} "
            f"!= {expected_cells}"
        )
    for key, path in (("results", result_path), ("events", events_path)):
        if path is None:
            continue
        entry = manifest.get(key)
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix}: missing {key} manifest entry")
            continue
        if entry.get("path") != path.name:
            errors.append(f"{prefix}: {key} basename mismatch")
        if entry.get("bytes") != path.stat().st_size:
            errors.append(f"{prefix}: {key} byte-count mismatch")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"{prefix}: {key} checksum mismatch")
    errors.extend(_walk_nonfinite(manifest, prefix))
    return errors


def _validate_fixed_array_manifest(
    manifest_path: Path,
    *,
    result_dir: Path,
    result_path: Path,
    cell_rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> list[str]:
    manifest = _load_json(manifest_path)
    errors: list[str] = []
    expected_dir_name = result_path.with_suffix(".arrays").name
    if manifest.get("arrays_dir") != expected_dir_name:
        errors.append(f"{prefix}: arrays directory name mismatch")
    arrays_dir = result_path.parent / expected_dir_name
    if not arrays_dir.is_dir():
        errors.append(f"{prefix}: arrays directory is missing")
        return errors
    recorded = manifest.get("array_artifacts")
    if not isinstance(recorded, list):
        return [f"{prefix}: array_artifacts must be a list"]
    if len(recorded) != len(cell_rows):
        errors.append(
            f"{prefix}: {len(recorded)} array artifacts != "
            f"{len(cell_rows)} cells"
        )
    by_cell: dict[str, Mapping[str, Any]] = {}
    for entry in recorded:
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix}: malformed array-artifact entry")
            continue
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in by_cell:
            errors.append(f"{prefix}: duplicate or invalid array cell id")
            continue
        by_cell[cell_id] = entry
        path = _relative_file(arrays_dir, entry.get("path"))
        if path is None:
            errors.append(f"{prefix}: unsafe array-artifact path")
            continue
        if not path.is_file():
            errors.append(f"{prefix}: missing array artifact {path.name}")
            continue
        if entry.get("bytes") != path.stat().st_size:
            errors.append(f"{prefix}: array artifact byte-count mismatch")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"{prefix}: array artifact checksum mismatch")
        if not isinstance(entry.get("members"), Mapping):
            errors.append(f"{prefix}: array member schema is missing")
    for row in cell_rows:
        cell_id = row.get("cell_id")
        entry = by_cell.get(cell_id)
        reference = _draw_artifact_reference(row)
        row_artifact = row.get("draws_artifact")
        row_members = (
            row_artifact.get("members")
            if isinstance(row_artifact, Mapping)
            else None
        )
        if entry is None:
            errors.append(f"{prefix}: cell {cell_id!r} lacks array manifest")
        elif reference != (entry.get("path"), entry.get("sha256")):
            errors.append(
                f"{prefix}: row/manifest array reference mismatch for "
                f"{cell_id!r}"
            )
        elif entry.get("members") != row_members:
            errors.append(
                f"{prefix}: row/manifest member schema mismatch for "
                f"{cell_id!r}"
            )
    arrays_relative = arrays_dir.relative_to(result_dir)
    if any(
        not path.is_file() or path.suffix != ".npz"
        for path in arrays_dir.iterdir()
    ):
        errors.append(
            f"{prefix}: {arrays_relative} contains a non-NPZ entry"
        )
    expected_names = {
        str(entry.get("path"))
        for entry in recorded
        if isinstance(entry, Mapping)
    }
    actual_names = {path.name for path in arrays_dir.iterdir()}
    if actual_names != expected_names:
        errors.append(f"{prefix}: unmanifested array artifacts are present")
    return errors


def _fixed_expected_keys(block: str, *, smoke: bool) -> set[tuple[Any, ...]]:
    if smoke:
        if block != "smoke":
            raise AssertionError(block)
        return {("isotropic_gaussian_d4", 0, None, 4_000)}
    if block == "illcond":
        return {("ill_cond_50", seed, None, 50_000) for seed in range(7)}
    if block == "german":
        return {("german_credit", seed, None, 50_000) for seed in range(5)}
    if block == "radon-50k":
        return {("radon", seed, None, 50_000) for seed in range(2)}
    if block == "radon-400k":
        return {("radon", seed, None, 400_000) for seed in range(2)}
    if block == "refusals":
        return {
            *(("neals_funnel", seed, None, 50_000) for seed in range(3)),
            *(("banana", seed, None, 50_000) for seed in range(2)),
        }
    if block == "controls":
        return {
            *(("stoch_vol", seed, None, 50_000) for seed in range(2)),
            *(("mvn_10", seed, None, 50_000) for seed in range(2)),
            ("isotropic_gaussian_d20", 0, None, 50_000),
        }
    if block == "clones":
        return {("ill_cond_50", seed, None, 50_000) for seed in range(3)}
    if block in ("manual-illcond", "manual-population-illcond"):
        return {
            ("ill_cond_50", seed, metric, 50_000)
            for seed in range(7)
            for metric in ("fisher_low_rank", "welford_diag")
        }
    if block in ("manual-german", "manual-population-german"):
        return {
            ("german_credit", seed, metric, 50_000)
            for seed in range(5)
            for metric in ("fisher_low_rank", "welford_diag")
        }
    raise AssertionError(block)


def _fixed_row_key(row: Mapping[str, Any], block: str) -> tuple[Any, ...]:
    metric = row.get("metric")
    budget = row.get(
        "nominal_max_grad_budget",
        row.get("max_grad_budget"),
    )
    if block.startswith("manual-") and budget is None:
        budget = 50_000
    return (row.get("model"), row.get("seed"), metric, budget)


def _required_sampling_divergence_fields(block: str) -> tuple[str, ...]:
    """Return the divergence summary fields for one fixed-suite row shape."""

    if block.startswith("manual-population-"):
        return (
            "sampling_divergences_per_chain",
            "sampling_divergences_all_chains",
        )
    return ("sampling_divergences",)


def _draw_artifact_reference(
    row: Mapping[str, Any],
) -> tuple[str, str] | None:
    artifact = row.get("draws_artifact")
    if isinstance(artifact, Mapping):
        path = artifact.get("path")
        checksum = artifact.get("sha256")
        if isinstance(path, str) and isinstance(checksum, str):
            return path, checksum
    pairs = (
        ("draws_artifact_path", "draws_artifact_sha256"),
        ("draws_npz_path", "draws_npz_sha256"),
        ("draw_artifact_path", "draw_artifact_sha256"),
    )
    for path_field, hash_field in pairs:
        path = row.get(path_field)
        checksum = row.get(hash_field)
        if isinstance(path, str) and isinstance(checksum, str):
            return path, checksum
    return None


def _validate_draw_artifact(
    row: Mapping[str, Any],
    *,
    artifact_dir: Path,
    prefix: str,
) -> list[str]:
    reference = _draw_artifact_reference(row)
    if reference is None:
        return [f"{prefix}: missing immutable draw-artifact reference"]
    relative, checksum = reference
    path = _relative_file(artifact_dir, relative)
    if path is None:
        return [f"{prefix}: unsafe draw-artifact path"]
    errors: list[str] = []
    if path.suffix != ".npz":
        errors.append(f"{prefix}: draw artifact must be an NPZ file")
    if not path.is_file():
        errors.append(f"{prefix}: draw artifact is missing")
    elif not _is_sha256(checksum):
        errors.append(f"{prefix}: draw-artifact checksum is invalid")
    elif sha256_file(path) != checksum:
        errors.append(f"{prefix}: draw-artifact checksum mismatch")
    if errors or not path.is_file():
        return errors

    artifact = row.get("draws_artifact")
    recorded_members = (
        artifact.get("members") if isinstance(artifact, Mapping) else None
    )
    if not isinstance(recorded_members, Mapping):
        return [*errors, f"{prefix}: draw-artifact member schema is missing"]

    def _integer_vector(value: Any) -> list[int] | None:
        if not isinstance(value, list) or not all(
            isinstance(item, int) for item in value
        ):
            return None
        return [int(item) for item in value]

    try:
        with np.load(path, allow_pickle=False) as archive:
            observed_names = set(archive.files)
            if observed_names != set(recorded_members):
                errors.append(f"{prefix}: NPZ member names do not match schema")
            for name in sorted(observed_names):
                array = archive[name]
                member = recorded_members.get(name)
                if not isinstance(member, Mapping):
                    errors.append(f"{prefix}: NPZ member {name!r} is unrecorded")
                else:
                    if member.get("dtype") != str(array.dtype):
                        errors.append(
                            f"{prefix}: NPZ member {name!r} dtype mismatch"
                        )
                    if member.get("shape") != list(array.shape):
                        errors.append(
                            f"{prefix}: NPZ member {name!r} shape mismatch"
                        )
                del array

            required = {
                "draws",
                "sampling_integration_steps",
                "sampling_divergences",
                "warmup_integration_steps",
                "warmup_divergences",
            }
            missing = sorted(required - observed_names)
            if missing:
                errors.append(f"{prefix}: NPZ lacks required members {missing}")
                return errors

            policy = row.get("comparison_policy")
            n_chains = row.get("n_chains")
            dimension = row.get("dimension")
            is_population = (
                row.get("population_sampling_performed") is True
                or policy == "equal_split_aggregate_budget_control"
            )

            draws = archive["draws"]
            expected_draw_ndim = 3 if is_population else 2
            if draws.ndim != expected_draw_ndim:
                errors.append(
                    f"{prefix}: draws must have {expected_draw_ndim} dimensions"
                )
            if isinstance(dimension, int) and (
                draws.ndim < 1 or draws.shape[-1] != dimension
            ):
                errors.append(f"{prefix}: draw dimension mismatch")
            expected_draws = row.get(
                "num_sampling_draws_per_chain",
                row.get("num_sampling_draws"),
            )
            observed_draws = (
                draws.shape[1]
                if is_population and draws.ndim == 3
                else draws.shape[0] if not is_population and draws.ndim == 2
                else None
            )
            if (
                isinstance(expected_draws, int)
                and observed_draws != expected_draws
            ):
                errors.append(f"{prefix}: sampling draw count mismatch")
            if is_population:
                if (
                    not isinstance(n_chains, int)
                    or draws.ndim != 3
                    or draws.shape[0] != n_chains
                ):
                    errors.append(f"{prefix}: population draw chain count mismatch")
                expected_hashes = row.get("draws_sha256_per_chain")
                observed_hashes = (
                    [sha256_array(draws[index]) for index in range(draws.shape[0])]
                    if draws.ndim == 3
                    else []
                )
                if expected_hashes != observed_hashes:
                    errors.append(
                        f"{prefix}: per-chain draw SHA-256 mismatch"
                    )
            else:
                expected_hash = row.get("draws_sha256")
                if not _is_sha256(expected_hash):
                    errors.append(f"{prefix}: row draw SHA-256 is missing")
                elif sha256_array(draws) != expected_hash:
                    errors.append(f"{prefix}: row draw SHA-256 mismatch")
            del draws

            sampling_steps = archive["sampling_integration_steps"]
            sampling_divergences = archive["sampling_divergences"]
            if is_population:
                if (
                    not isinstance(n_chains, int)
                    or sampling_steps.ndim < 2
                    or sampling_steps.shape[0] != n_chains
                ):
                    errors.append(
                        f"{prefix}: population sampling-gradient shape mismatch"
                    )
                else:
                    observed_sampling = (
                        sampling_steps.reshape(n_chains, -1)
                        .sum(axis=1)
                        .astype(int)
                        .tolist()
                    )
                    if (
                        _integer_vector(row.get("sampling_grads_per_chain"))
                        != observed_sampling
                    ):
                        errors.append(
                            f"{prefix}: per-chain sampling gradients mismatch"
                        )
                    if row.get("sampling_grads_all_chains") != sum(
                        observed_sampling
                    ):
                        errors.append(
                            f"{prefix}: aggregate sampling gradients mismatch"
                        )
                    observed_sampling_divergences = (
                        sampling_divergences.reshape(n_chains, -1)
                        .sum(axis=1)
                        .astype(int)
                        .tolist()
                    )
                    if (
                        _integer_vector(
                            row.get("sampling_divergences_per_chain")
                        )
                        != observed_sampling_divergences
                    ):
                        errors.append(
                            f"{prefix}: per-chain sampling divergences mismatch"
                        )
                    if row.get("sampling_divergences_all_chains") != sum(
                        observed_sampling_divergences
                    ):
                        errors.append(
                            f"{prefix}: aggregate sampling divergences mismatch"
                        )
            else:
                if row.get("sampling_grads") != int(sampling_steps.sum()):
                    errors.append(f"{prefix}: sampling-gradient total mismatch")
                if row.get("sampling_divergences") != int(
                    sampling_divergences.sum()
                ):
                    errors.append(f"{prefix}: sampling-divergence total mismatch")
            del sampling_steps
            del sampling_divergences

            warmup_steps = archive["warmup_integration_steps"]
            if policy == "automatic_joint_population":
                if warmup_steps.ndim < 2:
                    errors.append(
                        f"{prefix}: automatic warmup-gradient shape mismatch"
                    )
                else:
                    observed_warmup = (
                        warmup_steps.reshape(warmup_steps.shape[0], -1)
                        .sum(axis=0)
                        .astype(int)
                        .tolist()
                    )
                    if (
                        _integer_vector(row.get("warmup_grads_per_chain"))
                        != observed_warmup
                    ):
                        errors.append(
                            f"{prefix}: per-chain warmup gradients mismatch"
                        )
                    if row.get("warmup_grads_all_chains") != sum(
                        observed_warmup
                    ):
                        errors.append(
                            f"{prefix}: aggregate warmup gradients mismatch"
                        )
            elif is_population:
                if (
                    not isinstance(n_chains, int)
                    or warmup_steps.ndim < 2
                    or warmup_steps.shape[0] != n_chains
                ):
                    errors.append(
                        f"{prefix}: population warmup-gradient shape mismatch"
                    )
                else:
                    observed_warmup = (
                        warmup_steps.reshape(n_chains, -1)
                        .sum(axis=1)
                        .astype(int)
                        .tolist()
                    )
                    if (
                        _integer_vector(row.get("warmup_grads_per_chain"))
                        != observed_warmup
                    ):
                        errors.append(
                            f"{prefix}: per-chain warmup gradients mismatch"
                        )
                    if row.get("warmup_grads_all_chains") != sum(
                        observed_warmup
                    ):
                        errors.append(
                            f"{prefix}: aggregate warmup gradients mismatch"
                        )
            elif row.get("warmup_grads") != int(warmup_steps.sum()):
                errors.append(f"{prefix}: warmup-gradient total mismatch")
            del warmup_steps
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"{prefix}: cannot validate NPZ contents: {exc}")
    return errors


def _quality_gate_passed(row: Mapping[str, Any]) -> bool | None:
    for name in (
        "population_quality_pass",
        "population_quality_passed",
        "quality_gate_passed",
    ):
        value = row.get(name)
        if isinstance(value, bool):
            return value
    gate = row.get("population_quality_gate", row.get("quality_gate"))
    if isinstance(gate, bool):
        return gate
    if isinstance(gate, Mapping) and isinstance(gate.get("passed"), bool):
        return bool(gate["passed"])
    return None


def efficiency_estimands(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return named efficiency conventions without collapsing them."""

    marginal = row.get(
        "ess_per_grad_marginal_amortized",
        row.get("ess_per_grad_sc1"),
    )
    marginal_per_chain = row.get("ess_per_grad_marginal_per_chain")
    if marginal_per_chain is None and marginal is not None:
        marginal_per_chain = [marginal]
    one_output = row.get(
        "ess_per_grad_one_output_total",
        row.get("ess_per_grad_all_warmup_charged"),
    )
    if (
        one_output is None
        and row.get("comparison_policy")
        == "historical_single_chain_nominal_B"
    ):
        one_output = row.get("ess_per_grad_sc1")
    return {
        "marginal_amortized_chain0": marginal,
        "marginal_per_chain": marginal_per_chain,
        "one_output_total": one_output,
        "pooled_population_total": row.get(
            "ess_per_grad_pooled_population"
        ),
        "population_quality_passed": _quality_gate_passed(row),
    }


def _close(expected: float, observed: Any) -> bool:
    return isinstance(observed, (int, float)) and math.isclose(
        expected,
        float(observed),
        rel_tol=2e-10,
        abs_tol=1e-14,
    )


def _validate_efficiency_formulas(
    row: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    warmup = row.get("warmup_grads_all_chains")
    n_chains = row.get("n_chains")
    sampling = row.get("sampling_grads")
    min_ess = row.get("min_bulk_ess")
    estimands = efficiency_estimands(row)
    if all(
        isinstance(value, (int, float))
        for value in (warmup, n_chains, sampling, min_ess)
    ):
        expected_marginal = float(min_ess) / (
            float(warmup) / float(n_chains) + float(sampling)
        )
        if estimands["marginal_amortized_chain0"] is not None and not _close(
            expected_marginal,
            estimands["marginal_amortized_chain0"],
        ):
            errors.append(f"{prefix}: marginal-amortized rate is inconsistent")
        expected_one = float(min_ess) / (float(warmup) + float(sampling))
        if estimands["one_output_total"] is not None and not _close(
            expected_one,
            estimands["one_output_total"],
        ):
            errors.append(f"{prefix}: one-output-total rate is inconsistent")

    pooled_ess = row.get("min_bulk_ess_pooled")
    sampling_all = row.get("sampling_grads_all_chains")
    if all(
        isinstance(value, (int, float))
        for value in (warmup, pooled_ess, sampling_all)
    ):
        expected_pooled = float(pooled_ess) / (
            float(warmup) + float(sampling_all)
        )
        if estimands["pooled_population_total"] is not None and not _close(
            expected_pooled,
            estimands["pooled_population_total"],
        ):
            errors.append(f"{prefix}: pooled-population rate is inconsistent")

    policy = row.get("comparison_policy")
    per_chain_ess = row.get("min_bulk_ess_per_chain")
    per_chain_sampling = row.get("sampling_grads_per_chain")
    per_chain_rates = row.get("ess_per_grad_marginal_per_chain")
    if any(
        value is not None
        for value in (per_chain_ess, per_chain_sampling, per_chain_rates)
    ):
        if not all(
            isinstance(value, list)
            for value in (
                per_chain_ess,
                per_chain_sampling,
                per_chain_rates,
            )
        ):
            errors.append(f"{prefix}: per-chain efficiency vectors are malformed")
        elif not (
            len(per_chain_ess)
            == len(per_chain_sampling)
            == len(per_chain_rates)
            == n_chains
        ):
            errors.append(f"{prefix}: per-chain efficiency vector lengths differ")
        else:
            if policy == "automatic_joint_population" and isinstance(
                warmup,
                (int, float),
            ):
                warmup_denominators = [
                    float(warmup) / float(n_chains)
                ] * int(n_chains)
            elif policy == "equal_split_aggregate_budget_control":
                warmup_per_chain = row.get("warmup_grads_per_chain")
                if not isinstance(warmup_per_chain, list) or len(
                    warmup_per_chain
                ) != n_chains:
                    warmup_denominators = []
                    errors.append(
                        f"{prefix}: per-chain warmup gradients are malformed"
                    )
                else:
                    warmup_denominators = [
                        float(value) for value in warmup_per_chain
                    ]
            else:
                warmup_denominators = []
            if warmup_denominators:
                for chain_index, (
                    ess,
                    sampling_cost,
                    observed_rate,
                    warmup_cost,
                ) in enumerate(
                    zip(
                        per_chain_ess,
                        per_chain_sampling,
                        per_chain_rates,
                        warmup_denominators,
                        strict=True,
                    )
                ):
                    expected_rate = float(ess) / (
                        warmup_cost + float(sampling_cost)
                    )
                    if not _close(expected_rate, observed_rate):
                        errors.append(
                            f"{prefix}: marginal rate for chain "
                            f"{chain_index} is inconsistent"
                        )

    if policy == "historical_single_chain_nominal_B":
        historical_warmup = row.get("warmup_grads")
        if all(
            isinstance(value, (int, float))
            for value in (historical_warmup, sampling, min_ess)
        ):
            expected_historical = float(min_ess) / (
                float(historical_warmup) + float(sampling)
            )
            if not _close(
                expected_historical,
                row.get("ess_per_grad_one_output_total"),
            ):
                errors.append(
                    f"{prefix}: historical one-output-total rate is inconsistent"
                )
    return errors


def _validate_fixed_block(
    result_dir: Path,
    source_dir: Path,
    *,
    block: str,
    expected_count: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    stem = block.replace("-", "_")
    result_path = result_dir / "fixed" / f"{stem}.jsonl"
    events_path = result_dir / "fixed" / f"{stem}.events.jsonl"
    manifest_path = result_dir / "fixed" / f"{stem}.manifest.json"
    arrays_dir = result_path.with_suffix(".arrays")
    rows = _load_jsonl(result_path)
    events = _load_jsonl(events_path)
    errors.extend(
        _validate_provenance(
            rows[0],
            expected_suite="fixed-suite",
            expected_blackjax_sha=CURRENT_BLACKJAX_SHA,
            source_dir=source_dir,
            runner_name="run_fixed_suite.py",
            prefix=f"fixed/{block}:provenance",
        )
    )
    if rows[0].get("block") != block:
        errors.append(f"fixed/{block}: provenance block mismatch")
    cell_rows = [row for row in rows if row.get("record_type") == "cell"]
    summaries = [
        row for row in rows if row.get("record_type") == "block_summary"
    ]
    unexpected_types = [
        row.get("record_type")
        for row in rows
        if row.get("record_type")
        not in ("provenance", "cell", "block_summary")
    ]
    if len(cell_rows) != expected_count:
        errors.append(
            f"fixed/{block}: {len(cell_rows)} cells != {expected_count}"
        )
    if len(summaries) != 1:
        errors.append(f"fixed/{block}: expected exactly one block summary")
    elif (
        summaries[0].get("expected_cells") != expected_count
        or summaries[0].get("completed_cells") != expected_count
        or summaries[0].get("error_cells") != 0
    ):
        errors.append(f"fixed/{block}: invalid block summary")
    if unexpected_types:
        errors.append(
            f"fixed/{block}: unexpected record types {unexpected_types}"
        )

    expected_keys = _fixed_expected_keys(block, smoke=smoke)
    observed_keys: list[tuple[Any, ...]] = []
    auto_block = block in {
        "smoke",
        "illcond",
        "german",
        "radon-50k",
        "radon-400k",
        "refusals",
        "controls",
        "clones",
    }
    population_block = block in {
        "illcond",
        "german",
        "manual-population-illcond",
        "manual-population-german",
    } or (smoke and block == "smoke")
    for index, row in enumerate(cell_rows, start=1):
        prefix = f"fixed/{block}:cell[{index}]"
        if row.get("error") is not None:
            errors.append(f"{prefix}: cell contains an error")
        observed_keys.append(_fixed_row_key(row, block))
        errors.extend(_walk_nonfinite(row, prefix))
        errors.extend(
            _require_fields(
                row,
                (
                    "schema_version",
                    "run_id",
                    "cell_id",
                    "model",
                    "seed",
                    *_required_sampling_divergence_fields(block),
                ),
                prefix=prefix,
            )
        )
        if auto_block:
            errors.extend(
                _require_fields(
                    row,
                    (
                        "route",
                        "effective_rank",
                        "max_grad_budget",
                        "warmup_grads_all_chains",
                        "sampling_grads",
                        "min_bulk_ess",
                        "ess_per_grad_marginal_amortized",
                        "ess_per_grad_one_output_total",
                        "comparison_policy",
                        "schedule_prescriptions",
                    ),
                    prefix=prefix,
                )
            )
            if (
                row.get("comparison_policy")
                != "automatic_joint_population"
            ):
                errors.append(f"{prefix}: unexpected automatic comparison policy")
        elif block.startswith("manual-population-"):
            errors.extend(
                _require_fields(
                    row,
                    (
                        "metric",
                        "comparison_policy",
                        "n_chains",
                        "num_warmup_steps_per_chain",
                        "nominal_max_grad_budget",
                        "warmup_grads_all_chains",
                        "warmup_grads_per_chain",
                        "sampling_grads_per_chain",
                        "min_bulk_ess_per_chain",
                        "min_tail_ess_per_chain",
                        "min_bulk_ess_pooled",
                        "min_tail_ess_pooled",
                        "split_rhat_all_finite",
                        "ess_per_grad_marginal_per_chain",
                        "ess_per_grad_pooled_population",
                        "population_quality_pass",
                        "population_quality_rule",
                    ),
                    prefix=prefix,
                )
            )
            if (
                row.get("comparison_policy")
                != "equal_split_aggregate_budget_control"
            ):
                errors.append(
                    f"{prefix}: unexpected population-manual comparison policy"
                )
            if row.get("n_chains") != 8:
                errors.append(f"{prefix}: population comparator must use M=8")
            if row.get("num_warmup_steps_per_chain") != 312:
                errors.append(
                    f"{prefix}: population comparator must use 312 warmup steps"
                )
        else:
            errors.extend(
                _require_fields(
                    row,
                    (
                        "metric",
                        "comparison_policy",
                        "num_warmup_steps",
                        "nominal_max_grad_budget",
                        "warmup_grads",
                        "sampling_grads",
                        "min_bulk_ess",
                        "ess_per_grad_one_output_total",
                    ),
                    prefix=prefix,
                )
            )
            if (
                row.get("comparison_policy")
                != "historical_single_chain_nominal_B"
            ):
                errors.append(
                    f"{prefix}: unexpected historical comparison policy"
                )
        errors.extend(_validate_efficiency_formulas(row, prefix=prefix))
        errors.extend(
            _validate_draw_artifact(
                row,
                artifact_dir=arrays_dir,
                prefix=prefix,
            )
        )
        if population_block:
            population_fields = (
                "split_rhat_all_finite",
                "max_split_rhat_pooled",
                "population_quality_pass",
                "population_quality_rule",
                "ess_per_grad_pooled_population",
                "sampling_divergences_per_chain",
                "sampling_divergences_all_chains",
            )
            missing_population = [
                field for field in population_fields if field not in row
            ]
            if missing_population:
                errors.append(
                    f"{prefix}: missing population fields "
                    f"{missing_population}"
                )
            all_finite = row.get("split_rhat_all_finite")
            max_rhat = row.get("max_split_rhat_pooled")
            quality = row.get("population_quality_pass")
            sampling_divergences_all = row.get(
                "sampling_divergences_all_chains"
            )
            if not isinstance(all_finite, bool):
                errors.append(
                    f"{prefix}: split_rhat_all_finite must be boolean"
                )
            elif all_finite and not isinstance(max_rhat, (int, float)):
                errors.append(
                    f"{prefix}: finite split-Rhat flag requires a maximum"
                )
            elif not all_finite and max_rhat is not None:
                errors.append(
                    f"{prefix}: nonfinite split-Rhat must store a null maximum"
                )
            if not isinstance(quality, bool):
                errors.append(
                    f"{prefix}: population_quality_pass must be boolean"
                )
            else:
                expected_quality = (
                    all_finite is True
                    and isinstance(max_rhat, (int, float))
                    and max_rhat <= 1.01
                    and isinstance(sampling_divergences_all, int)
                    and sampling_divergences_all == 0
                )
                if quality is not expected_quality:
                    errors.append(
                        f"{prefix}: population_quality_pass does not match "
                        "the frozen split-Rhat/divergence rule"
                    )
            if _quality_gate_passed(row) is None:
                errors.append(f"{prefix}: population quality gate is missing")
    counts = Counter(observed_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"fixed/{block}: duplicate keys {duplicates}")
    actual = set(observed_keys)
    if actual != expected_keys:
        errors.append(
            f"fixed/{block}: grid mismatch; "
            f"missing={sorted(expected_keys - actual)}, "
            f"unexpected={sorted(actual - expected_keys)}"
        )

    if events[0].get("record_type") != "provenance":
        errors.append(f"fixed/{block}: event file lacks provenance")
    elif events[0].get("suite") != "fixed-suite-schedule-events":
        errors.append(f"fixed/{block}: event suite mismatch")
    event_rows = [
        row for row in events if row.get("record_type") == "schedule_event"
    ]
    event_keys: list[tuple[Any, ...]] = []
    events_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, event in enumerate(event_rows, start=1):
        prefix = f"fixed/{block}:event[{index}]"
        errors.extend(_walk_nonfinite(event, prefix))
        errors.extend(
            _require_fields(
                event,
                (
                    "cell_id",
                    "model",
                    "seed",
                    "schedule_family",
                    "window_index",
                ),
                prefix=prefix,
            )
        )
        event_keys.append((event.get("cell_id"), event.get("window_index")))
        cell_id = event.get("cell_id")
        if isinstance(cell_id, str):
            events_by_cell[cell_id].append(event)
        if "start_step" in event or "end_step" in event:
            errors.append(
                f"{prefix}: legacy zero-based window fields are forbidden"
            )
    if len(event_keys) != len(set(event_keys)):
        errors.append(f"fixed/{block}: duplicate schedule events")

    auto_rows_by_id = {
        row["cell_id"]: row
        for row in cell_rows
        if row.get("comparison_policy") == "automatic_joint_population"
    }
    unexpected_event_cells = sorted(
        set(events_by_cell) - set(auto_rows_by_id)
    )
    if unexpected_event_cells:
        errors.append(
            f"fixed/{block}: events reference unexpected cells "
            f"{unexpected_event_cells}"
        )
    for cell_id, row in auto_rows_by_id.items():
        cell_events = events_by_cell.get(cell_id, [])
        prescriptions = row.get("schedule_prescriptions")
        event_errors = validate_window_events(cell_events, prescriptions)
        errors.extend(
            f"fixed/{block}:{cell_id}: {error}" for error in event_errors
        )
        if isinstance(prescriptions, Mapping):
            expected_family = prescriptions.get(
                "controller_actual",
                {},
            ).get("name")
            if any(
                event.get("schedule_family") != expected_family
                for event in cell_events
            ):
                errors.append(
                    f"fixed/{block}:{cell_id}: event schedule family mismatch"
                )

    errors.extend(
        _validate_runner_manifest(
            manifest_path,
            result_path=result_path,
            events_path=events_path,
            expected_cells=expected_count,
            prefix=f"fixed/{block}:manifest",
        )
    )
    errors.extend(
        _validate_fixed_array_manifest(
            manifest_path,
            result_dir=result_dir,
            result_path=result_path,
            cell_rows=cell_rows,
            prefix=f"fixed/{block}:arrays",
        )
    )
    return cell_rows, errors


def _validate_simple_suite(
    result_dir: Path,
    source_dir: Path,
    *,
    relative_result: str,
    relative_events: str | None,
    runner_name: str,
    suite_name: str,
    expected_count: int,
    manifest_expected_count: int | None = None,
    expected_blackjax_sha: str = CURRENT_BLACKJAX_SHA,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    result_path = result_dir / relative_result
    events_path = result_dir / relative_events if relative_events else None
    manifest_path = result_path.with_suffix(".manifest.json")
    rows = _load_jsonl(result_path)
    errors.extend(
        _validate_provenance(
            rows[0],
            expected_suite=suite_name,
            expected_blackjax_sha=expected_blackjax_sha,
            source_dir=source_dir,
            runner_name=runner_name,
            prefix=f"{suite_name}:provenance",
        )
    )
    cell_rows = [row for row in rows if row.get("record_type") == "cell"]
    if len(cell_rows) != expected_count:
        errors.append(
            f"{suite_name}: {len(cell_rows)} cells != {expected_count}"
        )
    for index, row in enumerate(cell_rows, start=1):
        prefix = f"{suite_name}:cell[{index}]"
        if row.get("error") is not None:
            errors.append(f"{prefix}: cell contains an error")
        errors.extend(_walk_nonfinite(row, prefix))
        errors.extend(
            _require_fields(
                row,
                ("run_id", "cell_id", "model", "seed"),
                prefix=prefix,
            )
        )
    event_rows: list[dict[str, Any]] = []
    if events_path is not None:
        events = _load_jsonl(events_path)
        if events[0].get("record_type") != "provenance":
            errors.append(f"{suite_name}: event file lacks provenance")
        event_rows = [
            row
            for row in events
            if row.get("record_type") == "schedule_event"
        ]
        for index, row in enumerate(event_rows, start=1):
            errors.extend(
                _walk_nonfinite(row, f"{suite_name}:event[{index}]")
            )
    errors.extend(
        _validate_runner_manifest(
            manifest_path,
            result_path=result_path,
            events_path=events_path,
            expected_cells=(
                expected_count
                if manifest_expected_count is None
                else manifest_expected_count
            ),
            prefix=f"{suite_name}:manifest",
        )
    )
    return rows, event_rows, errors


def _exact_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    expected: set[tuple[Any, ...]],
    prefix: str,
) -> list[str]:
    keys = [tuple(row.get(field) for field in key_fields) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    errors: list[str] = []
    if duplicates:
        errors.append(f"{prefix}: duplicate cells {duplicates}")
    actual = set(keys)
    if actual != expected:
        errors.append(
            f"{prefix}: grid mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return errors


def _summarize_fixed_efficiency(
    fixed_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    automatic = [
        *fixed_rows.get("illcond", ()),
        *fixed_rows.get("german", ()),
    ]
    historical = [
        *fixed_rows.get("manual-illcond", ()),
        *fixed_rows.get("manual-german", ()),
    ]
    population = [
        *fixed_rows.get("manual-population-illcond", ()),
        *fixed_rows.get("manual-population-german", ()),
    ]
    historical_by_key = {
        (row["model"], row["seed"], row["metric"]): row
        for row in historical
    }
    population_by_key = {
        (row["model"], row["seed"], row["metric"]): row
        for row in population
    }
    joined: list[dict[str, Any]] = []
    for auto in sorted(automatic, key=lambda row: (row["model"], row["seed"])):
        key = (auto["model"], auto["seed"])
        auto_estimands = efficiency_estimands(auto)
        comparators: dict[str, Any] = {}
        for metric, role in (
            ("fisher_low_rank", "predeclared_primary"),
            ("welford_diag", "control"),
        ):
            short = population_by_key[(key[0], key[1], metric)]
            legacy = historical_by_key[(key[0], key[1], metric)]
            short_estimands = efficiency_estimands(short)
            both_quality = (
                auto_estimands["population_quality_passed"] is True
                and short_estimands["population_quality_passed"] is True
            )
            pooled_ratio = None
            if both_quality:
                pooled_ratio = (
                    auto_estimands["pooled_population_total"]
                    / short_estimands["pooled_population_total"]
                )
            comparators[metric] = {
                "role": role,
                "equal_split_population": short_estimands,
                "historical_nominal_B_sensitivity": efficiency_estimands(
                    legacy
                ),
                "automatic_to_manual_pooled_ratio": pooled_ratio,
                "pooled_ratio_reportable": both_quality,
            }
        joined.append(
            {
                "model": key[0],
                "seed": key[1],
                "automatic": auto_estimands,
                "comparators": comparators,
            }
        )
    return {
        "selection_policy": (
            "fisher_low_rank_predeclared_primary;"
            "welford_diag_reported_as_control;no_post_hoc_best_arm"
        ),
        "estimand_definitions": {
            "marginal_amortized": (
                "chain output divided by its sampling gradients plus one "
                "M-th share of joint warmup gradients"
            ),
            "one_output_total": (
                "one chain output divided by its sampling gradients plus all "
                "joint warmup gradients"
            ),
            "pooled_population_total": (
                "pooled-chain ESS divided by all warmup and sampling gradients"
            ),
        },
        "primary_reporting_gate": (
            "pooled ratios are emitted only when automatic and comparator "
            "population quality gates both pass"
        ),
        "cells": joined,
    }


def _summarize_kernel(
    rows: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    calibrations = [
        row
        for row in rows
        if row.get("record_type") == "trajectory_calibration"
    ]
    cells = [row for row in rows if row.get("record_type") == "cell"]
    models = ("ill_cond_50",) if smoke else ("ill_cond_50", "german_credit")
    seeds = (0,) if smoke else (0, 1, 2)
    algorithms = ("hmc",) if smoke else ("hmc", "multinomial_hmc")
    expected_calibrations = {(model, seed) for model in models for seed in seeds}
    errors.extend(
        _exact_grid(
            calibrations,
            key_fields=("model", "seed"),
            expected=expected_calibrations,
            prefix="kernel calibrations",
        )
    )
    expected_cells = {
        (model, seed, algorithm, arm)
        for model in models
        for seed in seeds
        for algorithm in algorithms
        for arm in ("automatic", "matched_manual")
    }
    errors.extend(
        _exact_grid(
            cells,
            key_fields=("model", "seed", "algorithm", "arm"),
            expected=expected_cells,
            prefix="kernel results",
        )
    )
    by_key = {
        (row["model"], row["seed"], row["algorithm"], row["arm"]): row
        for row in cells
    }
    pairs: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            for algorithm in algorithms:
                automatic = by_key[(model, seed, algorithm, "automatic")]
                manual = by_key[(model, seed, algorithm, "matched_manual")]
                errors.extend(
                    _require_fields(
                        automatic,
                        (
                            "route",
                            "effective_rank",
                            "sampling_divergences",
                            "ess_per_grad_sc1",
                        ),
                        prefix=f"kernel:{model}:{seed}:{algorithm}:automatic",
                    )
                )
                errors.extend(
                    _require_fields(
                        manual,
                        (
                            "sampling_divergences",
                            "ess_per_grad_sc1",
                            "automatic_to_manual_ratio",
                        ),
                        prefix=f"kernel:{model}:{seed}:{algorithm}:manual",
                    )
                )
                ratio = automatic["ess_per_grad_sc1"] / manual["ess_per_grad_sc1"]
                if not _close(ratio, manual["automatic_to_manual_ratio"]):
                    errors.append(
                        f"kernel:{model}:{seed}:{algorithm}: stored ratio mismatch"
                    )
                pairs.append(
                    {
                        "model": model,
                        "seed": seed,
                        "algorithm": algorithm,
                        "route": automatic["route"],
                        "effective_rank": automatic["effective_rank"],
                        "automatic_to_manual_ess_per_grad_ratio": ratio,
                        "automatic_divergences": automatic[
                            "sampling_divergences"
                        ],
                        "manual_divergences": manual[
                            "sampling_divergences"
                        ],
                    }
                )
    return {"cells": pairs}, errors


def _summarize_schedule(
    cells: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], list[str]]:
    models = ("ill_cond_50",) if smoke else ("ill_cond_50", "radon")
    seeds = (0,) if smoke else (0, 1, 2)
    schedules = ("stan_doubling",) if smoke else (
        "stan_doubling",
        "proportional_growing",
    )
    policies = ("reset",) if smoke else ("reset", "accumulating")
    expected = {
        (model, seed, schedule, policy)
        for model in models
        for seed in seeds
        for schedule in schedules
        for policy in policies
    }
    errors = _exact_grid(
        cells,
        key_fields=("model", "seed", "schedule_family", "buffer_policy"),
        expected=expected,
        prefix="schedule configuration",
    )
    for index, row in enumerate(cells, start=1):
        errors.extend(
            _require_fields(
                row,
                (
                    "ess_per_grad",
                    "warmup_grads",
                    "sampling_grads",
                    "sampling_divergences",
                ),
                prefix=f"schedule:cell[{index}]",
            )
        )
    contrasts: list[dict[str, Any]] = []
    if not smoke:
        by_key = {
            (
                row["model"],
                row["seed"],
                row["schedule_family"],
                row["buffer_policy"],
            ): row
            for row in cells
        }
        for model in models:
            for seed in seeds:
                for policy in policies:
                    stan = by_key[(model, seed, "stan_doubling", policy)]
                    growing = by_key[
                        (model, seed, "proportional_growing", policy)
                    ]
                    contrasts.append(
                        {
                            "contrast": "proportional_growing_over_stan_doubling",
                            "model": model,
                            "seed": seed,
                            "buffer_policy": policy,
                            "log_ess_per_grad_ratio": math.log(
                                growing["ess_per_grad"]
                                / stan["ess_per_grad"]
                            ),
                            "warmup_grad_ratio": growing["warmup_grads"]
                            / stan["warmup_grads"],
                            "divergence_delta": growing[
                                "sampling_divergences"
                            ]
                            - stan["sampling_divergences"],
                        }
                    )
                for schedule in schedules:
                    reset = by_key[(model, seed, schedule, "reset")]
                    accumulating = by_key[
                        (model, seed, schedule, "accumulating")
                    ]
                    contrasts.append(
                        {
                            "contrast": "accumulating_over_reset",
                            "model": model,
                            "seed": seed,
                            "schedule_family": schedule,
                            "log_ess_per_grad_ratio": math.log(
                                accumulating["ess_per_grad"]
                                / reset["ess_per_grad"]
                            ),
                            "warmup_grad_ratio": accumulating["warmup_grads"]
                            / reset["warmup_grads"],
                            "divergence_delta": accumulating[
                                "sampling_divergences"
                            ]
                            - reset["sampling_divergences"],
                        }
                    )
    return {
        "interpretation": (
            "configuration comparison; schedule changes boundaries and "
            "dual-averaging cadence"
        ),
        "cells": list(cells),
        "predeclared_contrasts": contrasts,
    }, errors


def _summarize_restart(
    cells: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], list[str]]:
    models = ("ill_cond_50",) if smoke else ("ill_cond_50", "german_credit")
    seeds = (0,) if smoke else (0, 1, 2)
    expected = {
        (model, seed, policy)
        for model in models
        for seed in seeds
        for policy in ("per_window_reseed", "continuous")
    }
    errors = _exact_grid(
        cells,
        key_fields=("model", "seed", "restart_policy"),
        expected=expected,
        prefix="restart ablation",
    )
    by_key = {
        (row["model"], row["seed"], row["restart_policy"]): row
        for row in cells
    }
    pairs: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            reseed = by_key[(model, seed, "per_window_reseed")]
            continuous = by_key[(model, seed, "continuous")]
            for label, row in (("reseed", reseed), ("continuous", continuous)):
                errors.extend(
                    _require_fields(
                        row,
                        (
                            "ess_per_grad_sc1",
                            "warmup_grads_all_chains",
                            "sampling_divergences",
                        ),
                        prefix=f"restart:{model}:{seed}:{label}",
                    )
                )
            pairs.append(
                {
                    "model": model,
                    "seed": seed,
                    "log_ess_per_grad_ratio_continuous_over_reseed": math.log(
                        continuous["ess_per_grad_sc1"]
                        / reseed["ess_per_grad_sc1"]
                    ),
                    "warmup_grad_ratio_continuous_over_reseed": (
                        continuous["warmup_grads_all_chains"]
                        / reseed["warmup_grads_all_chains"]
                    ),
                    "divergence_delta_continuous_minus_reseed": (
                        continuous["sampling_divergences"]
                        - reseed["sampling_divergences"]
                    ),
                }
            )
    return {
        "claim_policy": (
            "descriptive paired ratios only; no equivalence claim without a "
            "predeclared equivalence margin"
        ),
        "pairs": pairs,
    }, errors


def _summarize_shared(
    current: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
    *,
    smoke: bool,
) -> tuple[dict[str, Any], list[str]]:
    models_and_seeds = (
        (("ill_cond_50", 0),)
        if smoke
        else (
            *(("ill_cond_50", seed) for seed in range(7)),
            *(("german_credit", seed) for seed in range(5)),
        )
    )
    expected_current = {
        (model, seed, "current") for model, seed in models_and_seeds
    }
    expected_historical = {
        (model, seed, "historical") for model, seed in models_and_seeds
    }
    errors = _exact_grid(
        current,
        key_fields=("model", "seed", "revision_arm"),
        expected=expected_current,
        prefix="shared current",
    )
    errors.extend(
        _exact_grid(
            historical,
            key_fields=("model", "seed", "revision_arm"),
            expected=expected_historical,
            prefix="shared historical",
        )
    )
    current_by_key = {(row["model"], row["seed"]): row for row in current}
    historical_by_key = {
        (row["model"], row["seed"]): row for row in historical
    }
    pairs: list[dict[str, Any]] = []
    for key in models_and_seeds:
        new = current_by_key[key]
        old = historical_by_key[key]
        required = (
            "warmup_grads_all_chains",
            "settled_log_amplitude_log10",
            "shipped_to_equilibrium_ratio",
            "sampling_divergences",
            "route",
        )
        errors.extend(
            _require_fields(new, required, prefix=f"shared:{key}:current")
        )
        errors.extend(
            _require_fields(old, required, prefix=f"shared:{key}:historical")
        )
        pairs.append(
            {
                "model": key[0],
                "seed": key[1],
                "warmup_grad_ratio_historical_over_current": (
                    old["warmup_grads_all_chains"]
                    / new["warmup_grads_all_chains"]
                ),
                "settled_log_amplitude_log10": {
                    "historical": old["settled_log_amplitude_log10"],
                    "current": new["settled_log_amplitude_log10"],
                },
                "shipped_to_equilibrium_ratio": {
                    "historical": old["shipped_to_equilibrium_ratio"],
                    "current": new["shipped_to_equilibrium_ratio"],
                },
                "route": {
                    "historical": old["route"],
                    "current": new["route"],
                },
                "sampling_divergences": {
                    "historical": old["sampling_divergences"],
                    "current": new["sampling_divergences"],
                },
            }
        )
    return {
        "comparison_scope": (
            "shipped controller bundles: historical sequential/uncapped "
            "versus current mean-pooled/warmup-depth-capped"
        ),
        "pairs": pairs,
    }, errors


def _gmm_summary(result_dir: Path, *, smoke: bool) -> dict[str, Any]:
    try:
        from gmm_suite import (
            arm_specs,
            load_jsonl,
            validate_suite_dir,
        )
    except ImportError as exc:
        raise ValidationError(f"cannot import GMM validator: {exc}") from exc
    gmm_dir = result_dir / "gmm"
    validated = validate_suite_dir(
        gmm_dir,
        smoke=smoke,
        require_run_manifest=True,
    )
    arms: list[dict[str, Any]] = []
    for spec in arm_specs(smoke):
        rows = load_jsonl(gmm_dir / spec.filename)
        arm: dict[str, Any] = {
            "arm_id": spec.arm_id,
            "kind": spec.kind,
            "rows": len(rows),
            "sha256": sha256_file(gmm_dir / spec.filename),
        }
        if spec.kind == "primary":
            by_sr: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_sr[float(row["SR"])].append(row)
            arm["by_sr"] = [
                {
                    "SR": sr,
                    "route_counts": dict(
                        sorted(Counter(row["route"] for row in group).items())
                    ),
                    "deferred_count": sum(
                        bool(row["deferred_to_ensemble"]) for row in group
                    ),
                    "within_lam1_range": [
                        min(row["within_lam1"] for row in group),
                        max(row["within_lam1"] for row in group),
                    ],
                    "min_ess_per_grad_range": [
                        min(row["min_ess_per_grad"] for row in group),
                        max(row["min_ess_per_grad"] for row in group),
                    ],
                }
                for sr, group in sorted(by_sr.items())
            ]
        elif spec.kind == "matched_diagonal":
            ratios = [
                row["projection_bulk_ess_per_grad_ratio"] for row in rows
            ]
            arm["projection_bulk_ess_per_grad_ratio"] = {
                "median": statistics.median(ratios),
                "range": [min(ratios), max(ratios)],
                "cells": [
                    {
                        "SR": row["SR"],
                        "seed": row["seed"],
                        "ratio": row[
                            "projection_bulk_ess_per_grad_ratio"
                        ],
                    }
                    for row in rows
                ],
            }
        else:
            arm["cells"] = [
                {
                    "SR": row["SR"],
                    "seed": row["seed"],
                    "route": row["route"],
                    "bulk_ess_per_grad": row["bulk_ess_per_grad"],
                    "mode_weight_est": row["mode_weight_est"],
                }
                for row in rows
            ]
        arms.append(arm)
    return {
        "validation": validated,
        "arms": arms,
        "global_exploration_policy": (
            "current controlled-mixture rows must report not_established"
        ),
    }


def _validate_execution_manifest(
    result_dir: Path,
    source_dir: Path,
    *,
    smoke: bool,
    execution_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    provenance = _load_json(result_dir / "orchestrator_provenance.json")
    execution = (
        dict(execution_override)
        if execution_override is not None
        else _load_json(result_dir / "execution_manifest.json")
    )
    expected_mode = "smoke" if smoke else "full"
    for label, value in (("provenance", provenance), ("execution", execution)):
        if value.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{label}: unexpected schema version")
        if value.get("mode") != expected_mode:
            errors.append(f"{label}: mode mismatch")
        if value.get("excluded_examples") != ["gmm25"]:
            errors.append(f"{label}: excluded-example declaration mismatch")
        if value.get("resource_limits") != {
            "outer_memory_cap": RELEASE_RUNCAP,
            "mechanism": "runcap",
            "scope": "full_corpus_orchestrator_and_children",
        }:
            errors.append(f"{label}: release resource limit is missing")
        dependencies = value.get("dependencies")
        if not isinstance(dependencies, Mapping):
            errors.append(f"{label}: dependencies are missing")
            continue
        expected_revisions = {
            "current_blackjax": CURRENT_BLACKJAX_SHA,
            "historical_blackjax": HISTORICAL_BLACKJAX_SHA,
            "tuningfork": TUNINGFORK_SHA,
        }
        for dependency, revision in expected_revisions.items():
            state = dependencies.get(dependency, {})
            if state.get("revision") != revision or state.get("dirty") is not False:
                errors.append(f"{label}: invalid {dependency} state")
    required_sources = {
        "run_full_corpus.py",
        "validate_full_results.py",
        *(task.script for task in build_plan(smoke=smoke)),
        "suite_common.py",
    }
    errors.extend(
        _validate_source_hashes(
            source_dir,
            execution.get("source_sha256"),
            required_names=required_sources,
            prefix="execution manifest",
        )
    )
    output_files = execution.get("output_files")
    if not isinstance(output_files, Mapping):
        errors.append("execution manifest: output_files must be an object")
    else:
        for relative, entry in output_files.items():
            if not isinstance(entry, Mapping):
                errors.append(f"execution manifest: malformed {relative}")
            else:
                errors.extend(_check_hash_entry(result_dir, relative, entry))
    plan = build_plan(smoke=smoke)
    tasks = execution.get("tasks")
    if not isinstance(tasks, list):
        errors.append("execution manifest: tasks must be a list")
        tasks = []
    if [task.get("task_id") for task in tasks] != [
        task.task_id for task in plan
    ]:
        errors.append("execution manifest: producer task order mismatch")
    for task_record, task_spec in zip(tasks, plan):
        prefix = f"task {task_spec.task_id}"
        if (
            task_record.get("returncode") != 0
            or task_record.get("missing_outputs") != []
        ):
            errors.append(f"{prefix}: task was not successful")
        command = task_record.get("command")
        if not isinstance(command, list) or not command:
            errors.append(f"{prefix}: portable command missing")
        elif any(
            isinstance(value, str) and Path(value).is_absolute()
            for value in command
        ):
            errors.append(f"{prefix}: command leaks an absolute path")
        env = task_record.get("environment_overrides", {})
        expected_pythonpath = (
            "<historical-blackjax>:<tuningfork>"
            if task_spec.environment == "historical"
            else "<current-blackjax>:<tuningfork>"
        )
        if env.get("PYTHONPATH") != expected_pythonpath:
            errors.append(f"{prefix}: PYTHONPATH was not isolated")
        if (
            env.get("JAX_PLATFORM_NAME") != "cpu"
            or env.get("JAX_ENABLE_X64") != "1"
            or env.get("PYTHONUNBUFFERED") != "1"
            or env.get("PAPER1_RUNCAP") != RELEASE_RUNCAP
        ):
            errors.append(f"{prefix}: runtime environment mismatch")
        for relative in task_spec.expected_outputs:
            entry = task_record.get("outputs", {}).get(relative)
            if not isinstance(entry, Mapping):
                errors.append(f"{prefix}: missing output record for {relative}")
            else:
                errors.extend(_check_hash_entry(result_dir, relative, entry))
    if any("gmm25" in task.task_id.lower() for task in plan):
        errors.append("execution plan includes excluded gmm25 example")
    errors.extend(_walk_nonfinite(provenance, "orchestrator provenance"))
    errors.extend(_walk_nonfinite(execution, "execution manifest"))
    return execution, errors


def validate_full_results(
    result_dir: Path,
    *,
    smoke: bool,
    execution_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    source_dir = Path(__file__).resolve().parent
    errors: list[str] = []
    execution, execution_errors = _validate_execution_manifest(
        result_dir,
        source_dir,
        smoke=smoke,
        execution_override=execution_override,
    )
    errors.extend(execution_errors)

    fixed_counts = {"smoke": 1} if smoke else FULL_FIXED_BLOCK_COUNTS
    fixed_rows: dict[str, list[dict[str, Any]]] = {}
    for block, count in fixed_counts.items():
        rows, block_errors = _validate_fixed_block(
            result_dir,
            source_dir,
            block=block,
            expected_count=count,
            smoke=smoke,
        )
        fixed_rows[block] = rows
        errors.extend(block_errors)

    section_counts = SMOKE_SECTION_COUNTS if smoke else FULL_SECTION_COUNTS
    kernel_rows, _kernel_events, suite_errors = _validate_simple_suite(
        result_dir,
        source_dir,
        relative_result="kernel/kernel_family.jsonl",
        relative_events="kernel/kernel_family.events.jsonl",
        runner_name="run_kernel_family.py",
        suite_name="kernel-family",
        expected_count=section_counts["kernel_result_rows"],
        manifest_expected_count=section_counts["kernel_route_cells"],
    )
    errors.extend(suite_errors)
    kernel_summary, section_errors = _summarize_kernel(
        kernel_rows,
        smoke=smoke,
    )
    errors.extend(section_errors)

    schedule_rows, schedule_events, suite_errors = _validate_simple_suite(
        result_dir,
        source_dir,
        relative_result="schedule/schedule_configuration.jsonl",
        relative_events="schedule/schedule_configuration.events.jsonl",
        runner_name="run_schedule_comparison.py",
        suite_name="schedule-configuration",
        expected_count=section_counts["schedule_cells"],
    )
    errors.extend(suite_errors)
    schedule_cells = [
        row for row in schedule_rows if row.get("record_type") == "cell"
    ]
    schedule_summary, section_errors = _summarize_schedule(
        schedule_cells,
        smoke=smoke,
    )
    schedule_summary["stored_event_rows"] = len(schedule_events)
    errors.extend(section_errors)

    restart_rows, _events, suite_errors = _validate_simple_suite(
        result_dir,
        source_dir,
        relative_result="restart/restart_ablation.jsonl",
        relative_events=None,
        runner_name="run_restart_ablation.py",
        suite_name="restart-ablation",
        expected_count=section_counts["restart_cells"],
    )
    errors.extend(suite_errors)
    restart_summary, section_errors = _summarize_restart(
        [row for row in restart_rows if row.get("record_type") == "cell"],
        smoke=smoke,
    )
    errors.extend(section_errors)

    current_rows, _events, suite_errors = _validate_simple_suite(
        result_dir,
        source_dir,
        relative_result="shared_step/current.jsonl",
        relative_events=None,
        runner_name="run_shared_step_size.py",
        suite_name="shared-step-size",
        expected_count=section_counts["shared_current_cells"],
    )
    errors.extend(suite_errors)
    historical_rows, _events, suite_errors = _validate_simple_suite(
        result_dir,
        source_dir,
        relative_result="shared_step/historical.jsonl",
        relative_events=None,
        runner_name="run_shared_step_size.py",
        suite_name="shared-step-size",
        expected_count=section_counts["shared_historical_cells"],
        expected_blackjax_sha=HISTORICAL_BLACKJAX_SHA,
    )
    errors.extend(suite_errors)
    shared_summary, section_errors = _summarize_shared(
        [row for row in current_rows if row.get("record_type") == "cell"],
        [row for row in historical_rows if row.get("record_type") == "cell"],
        smoke=smoke,
    )
    errors.extend(section_errors)

    figures: dict[str, Any] = {}
    for relative in (
        "figures/figure_bbp.pdf",
        "figures/schedule_evidence.pdf",
        "figures/schedule_evidence.png",
    ):
        path = result_dir / relative
        if not path.is_file() or path.stat().st_size < 100:
            errors.append(f"figure is missing or empty: {relative}")
            continue
        prefix = path.read_bytes()[:8]
        valid_signature = (
            prefix.startswith(b"%PDF")
            if path.suffix == ".pdf"
            else prefix == b"\x89PNG\r\n\x1a\n"
        )
        if not valid_signature:
            errors.append(f"figure signature is invalid: {relative}")
            continue
        figures[Path(relative).name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    gmm_summary: dict[str, Any] = {}
    try:
        gmm_summary = _gmm_summary(result_dir, smoke=smoke)
    except Exception as exc:
        errors.append(f"GMM validation failed: {exc}")

    if errors:
        raise ValidationError("\n".join(errors))

    fixed_efficiency = (
        {
            "selection_policy": "not evaluated in smoke mode",
            "cells": [],
        }
        if smoke
        else _summarize_fixed_efficiency(fixed_rows)
    )
    counts = {
        "fixed_cells": sum(len(rows) for rows in fixed_rows.values()),
        "kernel_calibrations": len(
            [
                row
                for row in kernel_rows
                if row.get("record_type") == "trajectory_calibration"
            ]
        ),
        "kernel_result_rows": len(
            [row for row in kernel_rows if row.get("record_type") == "cell"]
        ),
        "kernel_route_cells": len(
            [row for row in kernel_rows if row.get("record_type") == "cell"]
        )
        // 2,
        "schedule_cells": len(schedule_cells),
        "restart_cells": len(restart_summary["pairs"]) * 2,
        "shared_current_cells": len(shared_summary["pairs"]),
        "shared_historical_cells": len(shared_summary["pairs"]),
        "gmm_cells": sum(
            arm["rows"] for arm in gmm_summary.get("arms", [])
        ),
    }
    if counts != section_counts:
        raise ValidationError(
            f"final section counts {counts} != frozen plan {section_counts}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "mode": "smoke" if smoke else "full",
        "run_id": execution["run_id"],
        "counts": counts,
        "efficiency": fixed_efficiency,
        "kernel_family": kernel_summary,
        "schedule_configuration": schedule_summary,
        "restart_ablation": restart_summary,
        "shared_step_size": shared_summary,
        "controlled_gmm": gmm_summary,
        "figures": figures,
        "traceability": {
            "execution_manifest": {
                "path": "execution_manifest.json",
                "sha256": sha256_file(
                    result_dir / "execution_manifest.json"
                ),
            },
            "raw_result_policy": "immutable exclusive-create outputs",
            "number_source_policy": (
                "manuscript numbers are generated from this frozen summary "
                "and its checksummed raw inputs"
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary_out.exists():
        print(
            "validation failed: summary output already exists",
            file=sys.stderr,
            flush=True,
        )
        return 1
    try:
        summary = validate_full_results(args.result_dir, smoke=args.smoke)
        _write_json_exclusive(args.summary_out, summary)
    except (ValidationError, OSError, ValueError, ZeroDivisionError) as exc:
        print(f"validation failed:\n{exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "mode": summary["mode"],
                "counts": summary["counts"],
                "summary_sha256": sha256_file(args.summary_out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
