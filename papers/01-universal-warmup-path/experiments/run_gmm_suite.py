#!/usr/bin/env python
"""Run the complete controlled-GMM artifact against a pinned BlackJAX checkout."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from gmm_boundary import (
    _append_jsonl_row,
    _open_exclusive_jsonl,
    run_ablation,
    run_point,
    run_single_chain,
)
from gmm_suite import (
    EXPECTED_BLACKJAX_SHA,
    ArmSpec,
    arm_specs,
    validate_arm_file,
    validate_suite_dir,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _prepare_blackjax(blackjax_root: Path) -> dict[str, Any]:
    root = blackjax_root.resolve()
    if not (root / "blackjax").is_dir():
        raise RuntimeError(f"not a BlackJAX source checkout: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    dirty_output = _git(root, "status", "--porcelain")
    if commit != EXPECTED_BLACKJAX_SHA:
        raise RuntimeError(
            "BlackJAX revision mismatch: "
            f"{commit} != {EXPECTED_BLACKJAX_SHA}"
        )
    if dirty_output:
        raise RuntimeError("BlackJAX checkout has uncommitted changes")

    sys.path.insert(0, str(root))
    import blackjax

    resolved_module = Path(blackjax.__file__).resolve()
    if not resolved_module.is_relative_to(root):
        raise RuntimeError(
            f"imported BlackJAX from {resolved_module}, expected checkout {root}"
        )
    return {
        "commit": commit,
        "dirty": False,
        "checkout_name": root.name,
        "module_relative_path": str(resolved_module.relative_to(root)),
    }


def _source_hashes(source_dir: Path) -> dict[str, str]:
    paths = sorted(source_dir.glob("*.py"))
    manifest = source_dir / "artifact_manifest.json"
    if manifest.is_file():
        paths.append(manifest)
    return {path.name: _sha256(path) for path in paths}


def _provenance(
    blackjax: dict[str, Any],
    source_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    package_names = ("arviz", "jax", "matplotlib", "numpy")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "entry_point": Path(sys.argv[0]).name,
        "mode": "smoke" if smoke else "full",
        "python": {
            "executable_name": Path(sys.executable).name,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "jax": {
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "blackjax": blackjax,
        "source_sha256": _source_hashes(source_dir),
        "environment": {
            key: os.environ.get(key)
            for key in ("JAX_ENABLE_X64", "JAX_PLATFORM_NAME", "XLA_FLAGS")
        },
        "excluded_examples": ["gmm25"],
    }


def _run_arm(spec: ArmSpec, output_dir: Path) -> dict[str, Any]:
    output_path = output_dir / spec.filename
    completed = 0
    with _open_exclusive_jsonl(output_path) as stream:
        for seed in spec.seeds:
            for sr in spec.srs:
                started = time.monotonic()
                if spec.kind == "primary":
                    row = run_point(
                        sr,
                        seed=seed,
                        init_kind=spec.init_kind,
                        budget=spec.budget,
                        M=spec.chains,
                        n_sample_draws=spec.draws,
                        correlated_axes=spec.correlated_axes,
                    )
                elif spec.kind == "single_chain":
                    row = run_single_chain(
                        sr,
                        seed=seed,
                        budget=spec.budget,
                        n_sample_draws=spec.draws,
                        correlated_axes=spec.correlated_axes,
                    )
                else:
                    row = run_ablation(
                        sr,
                        seed=seed,
                        budget=spec.budget,
                        M=spec.chains,
                        n_sample_draws=spec.draws,
                        correlated_axes=spec.correlated_axes,
                    )
                row["arm_id"] = spec.arm_id
                row["elapsed_seconds"] = time.monotonic() - started
                _append_jsonl_row(stream, row)
                completed += 1
                print(
                    f"{spec.arm_id}: {completed}/{spec.expected_rows} "
                    f"seed={seed} SR={sr:g} error={row['error'] is not None}",
                    flush=True,
                )
                if row["error"] is not None:
                    raise RuntimeError(
                        f"{spec.arm_id} failed at seed={seed}, SR={sr:g}: "
                        f"{row['error']}"
                    )
    return validate_arm_file(output_path, spec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and validate the manuscript's complete controlled-GMM grid."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for raw JSONL and provenance (must not exist)",
    )
    parser.add_argument(
        "--blackjax-root",
        required=True,
        type=Path,
        help=f"Clean BlackJAX checkout at {EXPECTED_BLACKJAX_SHA}",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Exercise every arm with a small cell/draw grid",
    )
    args = parser.parse_args()

    blackjax = _prepare_blackjax(args.blackjax_root)
    if jax.default_backend() != "cpu":
        parser.error(f"JAX backend must be CPU, got {jax.default_backend()}")
    if not jax.config.x64_enabled:
        parser.error("JAX x64 must be enabled")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(
            f"output directory already exists; raw results are never overwritten: "
            f"{args.output_dir}"
        )

    source_dir = Path(__file__).resolve().parent
    specs = arm_specs(args.smoke)
    _write_json_exclusive(
        args.output_dir / "provenance.json",
        _provenance(blackjax, source_dir, smoke=args.smoke),
    )
    started_utc = datetime.now(UTC)
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        for spec in specs:
            summaries.append(_run_arm(spec, args.output_dir))
        validate_suite_dir(args.output_dir, smoke=args.smoke)
    except Exception as exc:
        _write_json_exclusive(
            args.output_dir / "failure.json",
            {
                "failed_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_arms": summaries,
            },
        )
        raise

    result_files = sorted(args.output_dir.glob("*.jsonl"))
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "started_utc": started_utc.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "arms": [spec.as_json() for spec in specs],
        "validation": summaries,
        "result_sha256": {
            path.name: _sha256(path) for path in result_files
        },
    }
    _write_json_exclusive(args.output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": run_manifest["mode"],
                "arms": len(specs),
                "rows": sum(spec.expected_rows for spec in specs),
                "elapsed_seconds": run_manifest["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
