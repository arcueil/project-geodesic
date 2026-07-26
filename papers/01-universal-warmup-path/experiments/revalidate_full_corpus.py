#!/usr/bin/env python3
"""Revalidate an immutable corpus after a validator-only schema correction.

The original result directory is never modified.  This command verifies that
every producer source and raw-output hash still matches the execution manifest,
changes only the recorded validator hash in an explicit overlay, and writes a
separate checksummed validation attestation plus frozen summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from run_full_corpus import sha256_file
from validate_full_results import validate_full_results


SCHEMA_VERSION = "paper1-validation-attestation-v1"
ORIGINAL_VALIDATOR_SHA256 = (
    "612a8fefeafa367f311315dcba5b7a1a02042b328aba4bb3055ae09b88d860fa"
)
VALIDATOR_NAME = "validate_full_results.py"


class RevalidationError(RuntimeError):
    """Raised when a validation replay cannot establish clean lineage."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RevalidationError(f"{path.name} must contain a JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def build_execution_overlay(
    execution: Mapping[str, Any],
    *,
    repaired_validator_sha256: str,
) -> dict[str, Any]:
    """Return an overlay that changes only validator provenance semantics."""

    overlay = copy.deepcopy(dict(execution))
    sources = overlay.get("source_sha256")
    if not isinstance(sources, dict):
        raise RevalidationError("execution manifest source hashes are missing")
    recorded = sources.get(VALIDATOR_NAME)
    if recorded != ORIGINAL_VALIDATOR_SHA256:
        raise RevalidationError(
            "execution manifest does not contain the expected failing "
            "validator hash"
        )
    if (
        len(repaired_validator_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in repaired_validator_sha256
        )
    ):
        raise RevalidationError("repaired validator hash is invalid")
    if repaired_validator_sha256 == recorded:
        raise RevalidationError("validator correction did not change the source")

    sources[VALIDATOR_NAME] = repaired_validator_sha256
    overlay["validation_overlay"] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "validator_only_schema_correction",
        "original_validator_sha256": recorded,
        "repaired_validator_sha256": repaired_validator_sha256,
        "changed_execution_fields": [
            f"source_sha256.{VALIDATOR_NAME}",
            "validation_overlay",
        ],
        "producer_sources_changed": False,
        "raw_results_changed": False,
    }
    return overlay


def _check_failure_lineage(
    execution: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> None:
    if execution.get("status") != "producers_complete":
        raise RevalidationError("producers were not recorded as complete")
    if failure.get("failed_task") != "validate-full-results":
        raise RevalidationError("the original failure was not validator-only")
    if failure.get("run_id") != execution.get("run_id"):
        raise RevalidationError("failure and execution run IDs differ")
    tasks = execution.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RevalidationError("execution task records are missing")
    if any(
        task.get("returncode") != 0 or task.get("missing_outputs") != []
        for task in tasks
    ):
        raise RevalidationError("at least one producer task was unsuccessful")


def revalidate(result_dir: Path, attestation_dir: Path) -> Path:
    result_dir = result_dir.resolve()
    attestation_dir = Path(os.path.abspath(attestation_dir.expanduser()))
    if not result_dir.is_dir():
        raise RevalidationError("result directory does not exist")
    if attestation_dir.exists():
        raise RevalidationError(
            "attestation directory already exists; outputs are immutable"
        )

    execution_path = result_dir / "execution_manifest.json"
    failure_path = result_dir / "failure.json"
    execution = _load_json(execution_path)
    failure = _load_json(failure_path)
    _check_failure_lineage(execution, failure)

    source_dir = Path(__file__).resolve().parent
    validator_path = source_dir / VALIDATOR_NAME
    repaired_validator_sha256 = sha256_file(validator_path)
    overlay = build_execution_overlay(
        execution,
        repaired_validator_sha256=repaired_validator_sha256,
    )
    overlay_bytes = _json_bytes(overlay)
    overlay_sha256 = _sha256_bytes(overlay_bytes)

    summary = validate_full_results(
        result_dir,
        smoke=execution.get("mode") == "smoke",
        execution_override=overlay,
    )

    summary["validation_lineage"] = {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": execution["run_id"],
        "source_execution_manifest_sha256": sha256_file(execution_path),
        "execution_manifest_overlay_sha256": overlay_sha256,
        "original_validator_sha256": ORIGINAL_VALIDATOR_SHA256,
        "repaired_validator_sha256": repaired_validator_sha256,
        "raw_results_changed": False,
        "producer_computation_rerun": False,
    }
    summary["traceability"]["execution_manifest"] = {
        "path": "execution_manifest.overlay.json",
        "sha256": overlay_sha256,
    }
    summary["traceability"]["source_execution_manifest"] = {
        "run_id": execution["run_id"],
        "sha256": sha256_file(execution_path),
    }
    summary_bytes = _json_bytes(summary)
    summary_sha256 = _sha256_bytes(summary_bytes)

    failing_log = result_dir / "logs" / "validate-full-results.log"
    test_path = source_dir / "test_full_corpus.py"
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_run_id": execution["run_id"],
        "source_run_directory_name": result_dir.name,
        "reason": (
            "The original validator required a single-chain scalar divergence "
            "field on population-manual rows. Those rows correctly store "
            "per-chain and all-chain divergence summaries. The correction is "
            "validation-only."
        ),
        "correction_scope": {
            "producer_sources_changed": False,
            "raw_jsonl_changed": False,
            "raw_npz_changed": False,
            "figures_changed": False,
            "producer_computation_rerun": False,
        },
        "inputs": {
            "source_execution_manifest_sha256": sha256_file(execution_path),
            "source_failure_manifest_sha256": sha256_file(failure_path),
            "source_validator_failure_log_sha256": sha256_file(failing_log),
            "original_validator_sha256": ORIGINAL_VALIDATOR_SHA256,
        },
        "repaired_validation": {
            "validator_sha256": repaired_validator_sha256,
            "revalidation_script_sha256": sha256_file(Path(__file__)),
            "regression_test_source_sha256": sha256_file(test_path),
            "execution_manifest_overlay_sha256": overlay_sha256,
            "frozen_summary_sha256": summary_sha256,
        },
    }
    attestation_bytes = _json_bytes(attestation)

    attestation_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "execution_manifest.overlay.json": overlay_bytes,
        "frozen_summary.json": summary_bytes,
        "validation_attestation.json": attestation_bytes,
    }
    for name, contents in outputs.items():
        with (attestation_dir / name).open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    checksum_lines = [
        f"{_sha256_bytes(contents)}  {name}"
        for name, contents in sorted(outputs.items())
    ]
    with (attestation_dir / "checksums.sha256").open(
        "x",
        encoding="utf-8",
    ) as stream:
        stream.write("\n".join(checksum_lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return attestation_dir / "validation_attestation.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--attestation-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        attestation = revalidate(args.result_dir, args.attestation_dir)
    except (OSError, RevalidationError, ValueError) as exc:
        print(f"revalidation failed: {exc}", flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "attestation": attestation.name,
                "attestation_sha256": sha256_file(attestation),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
