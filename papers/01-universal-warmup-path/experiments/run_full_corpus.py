#!/usr/bin/env python3
"""Run the complete Paper 1 experiment corpus serially and immutably.

The command creates a new result directory, runs one producer at a time, and
stops at the first non-zero exit.  Every producer has an unbuffered, persistent
log.  Commands recorded in the manifests use portable checkout placeholders;
the local absolute paths supplied on the command line are never copied into
the publication artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "paper1-full-corpus-v1"
CURRENT_BLACKJAX_SHA = "29d2468857be4de1644ca4470c2a4aa7f8137656"
HISTORICAL_BLACKJAX_SHA = "2f62921848a93e7dc544ba9de8e29ef177e373b6"
TUNINGFORK_SHA = "79ffb73250f5024dc511b3035d373d11474c2195"
RELEASE_RUNCAP = "80G"

FULL_FIXED_BLOCK_COUNTS: dict[str, int] = {
    "illcond": 7,
    "german": 5,
    "radon-50k": 2,
    "radon-400k": 2,
    "refusals": 5,
    "controls": 5,
    "clones": 3,
    "manual-illcond": 14,
    "manual-german": 10,
    "manual-population-illcond": 14,
    "manual-population-german": 10,
}

FULL_SECTION_COUNTS: dict[str, int] = {
    "fixed_cells": sum(FULL_FIXED_BLOCK_COUNTS.values()),
    "kernel_route_cells": 12,
    "kernel_calibrations": 6,
    "kernel_result_rows": 24,
    "schedule_cells": 24,
    "restart_cells": 12,
    "shared_current_cells": 12,
    "shared_historical_cells": 12,
    "gmm_cells": 239,
}

SMOKE_SECTION_COUNTS: dict[str, int] = {
    "fixed_cells": 1,
    "kernel_route_cells": 1,
    "kernel_calibrations": 1,
    "kernel_result_rows": 2,
    "schedule_cells": 1,
    "restart_cells": 2,
    "shared_current_cells": 1,
    "shared_historical_cells": 1,
    "gmm_cells": 9,
}


@dataclass(frozen=True)
class TaskSpec:
    """One serial producer with portable argument templates."""

    task_id: str
    suite: str
    environment: str
    script: str
    arguments: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    expected_cells: int | None = None

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


class CorpusRunError(RuntimeError):
    """Raised after a producer or validation task fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class _JsonlLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8")

    def emit(self, value: Mapping[str, Any]) -> None:
        self._stream.write(
            json.dumps(
                value,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "_JsonlLedger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _fixed_task(block: str, expected_cells: int) -> TaskSpec:
    stem = block.replace("-", "_")
    return TaskSpec(
        task_id=f"fixed-{block}",
        suite="fixed-suite",
        environment="current",
        script="run_fixed_suite.py",
        arguments=(
            "--block",
            block,
            "--out",
            f"{{output}}/fixed/{stem}.jsonl",
            "--events-out",
            f"{{output}}/fixed/{stem}.events.jsonl",
            "--arrays-dir",
            f"{{output}}/fixed/{stem}.arrays",
            "--run-id",
            "{run_id}",
        ),
        expected_outputs=(
            f"fixed/{stem}.jsonl",
            f"fixed/{stem}.events.jsonl",
            f"fixed/{stem}.manifest.json",
        ),
        expected_cells=expected_cells,
    )


def build_plan(*, smoke: bool) -> tuple[TaskSpec, ...]:
    """Return the frozen serial task order."""

    if smoke:
        fixed = (_fixed_task("smoke", 1),)
        smoke_flag = ("--smoke",)
    else:
        fixed = tuple(
            _fixed_task(block, count)
            for block, count in FULL_FIXED_BLOCK_COUNTS.items()
        )
        smoke_flag = ()

    tasks: list[TaskSpec] = [*fixed]
    tasks.extend(
        [
            TaskSpec(
                "kernel-family",
                "kernel-family",
                "current",
                "run_kernel_family.py",
                (
                    "--out",
                    "{output}/kernel/kernel_family.jsonl",
                    "--events-out",
                    "{output}/kernel/kernel_family.events.jsonl",
                    "--run-id",
                    "{run_id}",
                    *smoke_flag,
                ),
                (
                    "kernel/kernel_family.jsonl",
                    "kernel/kernel_family.events.jsonl",
                    "kernel/kernel_family.manifest.json",
                ),
                SMOKE_SECTION_COUNTS["kernel_route_cells"]
                if smoke
                else FULL_SECTION_COUNTS["kernel_route_cells"],
            ),
            TaskSpec(
                "schedule-configuration",
                "schedule-configuration",
                "current",
                "run_schedule_comparison.py",
                (
                    "--out",
                    "{output}/schedule/schedule_configuration.jsonl",
                    "--events-out",
                    "{output}/schedule/schedule_configuration.events.jsonl",
                    "--run-id",
                    "{run_id}",
                    *smoke_flag,
                ),
                (
                    "schedule/schedule_configuration.jsonl",
                    "schedule/schedule_configuration.events.jsonl",
                    "schedule/schedule_configuration.manifest.json",
                ),
                SMOKE_SECTION_COUNTS["schedule_cells"]
                if smoke
                else FULL_SECTION_COUNTS["schedule_cells"],
            ),
            TaskSpec(
                "restart-ablation",
                "restart-ablation",
                "current",
                "run_restart_ablation.py",
                (
                    "--out",
                    "{output}/restart/restart_ablation.jsonl",
                    "--run-id",
                    "{run_id}",
                    *smoke_flag,
                ),
                (
                    "restart/restart_ablation.jsonl",
                    "restart/restart_ablation.manifest.json",
                ),
                SMOKE_SECTION_COUNTS["restart_cells"]
                if smoke
                else FULL_SECTION_COUNTS["restart_cells"],
            ),
            TaskSpec(
                "shared-step-current",
                "shared-step-size",
                "current",
                "run_shared_step_size.py",
                (
                    "--revision",
                    "current",
                    "--out",
                    "{output}/shared_step/current.jsonl",
                    "--run-id",
                    "{run_id}",
                    *smoke_flag,
                ),
                (
                    "shared_step/current.jsonl",
                    "shared_step/current.manifest.json",
                ),
                SMOKE_SECTION_COUNTS["shared_current_cells"]
                if smoke
                else FULL_SECTION_COUNTS["shared_current_cells"],
            ),
            TaskSpec(
                "shared-step-historical",
                "shared-step-size",
                "historical",
                "run_shared_step_size.py",
                (
                    "--revision",
                    "historical",
                    "--out",
                    "{output}/shared_step/historical.jsonl",
                    "--run-id",
                    "{run_id}",
                    *smoke_flag,
                ),
                (
                    "shared_step/historical.jsonl",
                    "shared_step/historical.manifest.json",
                ),
                SMOKE_SECTION_COUNTS["shared_historical_cells"]
                if smoke
                else FULL_SECTION_COUNTS["shared_historical_cells"],
            ),
            TaskSpec(
                "controlled-gmm",
                "controlled-gmm",
                "current",
                "run_gmm_suite.py",
                (
                    "--output-dir",
                    "{output}/gmm",
                    "--blackjax-root",
                    "{current_blackjax}",
                    *smoke_flag,
                ),
                (
                    "gmm/provenance.json",
                    "gmm/run_manifest.json",
                ),
                SMOKE_SECTION_COUNTS["gmm_cells"]
                if smoke
                else FULL_SECTION_COUNTS["gmm_cells"],
            ),
            TaskSpec(
                "bbp-calibration",
                "bbp-calibration",
                "current",
                "make_figure_bbp.py",
                (
                    "--out",
                    "{output}/figures/figure_bbp.pdf",
                    *(("--dimension", "20", "--trials", "1") if smoke else ()),
                ),
                ("figures/figure_bbp.pdf",),
            ),
            TaskSpec(
                "schedule-evidence-pdf",
                "schedule-evidence",
                "current",
                "plot_schedule_evidence.py",
                (
                    "--input",
                    "{output}/gmm/gmm_k2_primary_60k.jsonl",
                    "--out",
                    "{output}/figures/schedule_evidence.pdf",
                    "--arm-id",
                    "gmm_k2_primary_60k",
                    "--sr",
                    "1.5" if smoke else "5.0",
                    "--seed",
                    "42",
                ),
                ("figures/schedule_evidence.pdf",),
            ),
            TaskSpec(
                "schedule-evidence-png",
                "schedule-evidence",
                "current",
                "plot_schedule_evidence.py",
                (
                    "--input",
                    "{output}/gmm/gmm_k2_primary_60k.jsonl",
                    "--out",
                    "{output}/figures/schedule_evidence.png",
                    "--arm-id",
                    "gmm_k2_primary_60k",
                    "--sr",
                    "1.5" if smoke else "5.0",
                    "--seed",
                    "42",
                ),
                ("figures/schedule_evidence.png",),
            ),
        ]
    )
    return tuple(tasks)


def _git_state(root: Path, expected_revision: str, label: str) -> dict[str, Any]:
    if not (root / ".git").exists() or not (root / label).is_dir():
        raise CorpusRunError(f"{label} root is not a source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    )
    if revision != expected_revision:
        raise CorpusRunError(
            f"{label} revision mismatch: {revision} != {expected_revision}"
        )
    if dirty:
        raise CorpusRunError(f"{label} checkout is dirty")
    return {"checkout_name": root.name, "revision": revision, "dirty": False}


def _interpreter_state(python: Path) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise CorpusRunError("the requested Python interpreter is not executable")
    code = (
        "import importlib.metadata as m,json,platform;"
        "p=('arviz','jax','jaxlib','matplotlib','numpy','numpyro','scipy');"
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{x:m.version(x) for x in p}},sort_keys=True))"
    )
    output = subprocess.check_output(
        [str(python), "-c", code],
        text=True,
        stderr=subprocess.STDOUT,
    )
    state = json.loads(output)
    state["executable_name"] = python.name
    return state


def _source_hashes(source_dir: Path) -> dict[str, str]:
    paths = sorted(source_dir.glob("*.py"))
    artifact_manifest = source_dir / "artifact_manifest.json"
    if artifact_manifest.is_file():
        paths.append(artifact_manifest)
    return {path.name: sha256_file(path) for path in paths}


def _resolve_template(
    value: str,
    *,
    output_dir: Path,
    run_id: str,
    python: Path,
    current_blackjax: Path,
    historical_blackjax: Path,
    tuningfork: Path,
) -> str:
    return value.format(
        output=str(output_dir),
        run_id=run_id,
        python=str(python),
        current_blackjax=str(current_blackjax),
        historical_blackjax=str(historical_blackjax),
        tuningfork=str(tuningfork),
    )


def _portable_command(task: TaskSpec, python_name: str) -> list[str]:
    replacements = {
        "{output}": "<run-dir>",
        "{current_blackjax}": "<current-blackjax>",
        "{historical_blackjax}": "<historical-blackjax>",
        "{tuningfork}": "<tuningfork>",
        "{run_id}": "<run-id>",
    }
    arguments: list[str] = []
    for raw in task.arguments:
        value = raw
        for source, replacement in replacements.items():
            value = value.replace(source, replacement)
        arguments.append(value)
    return [python_name, "-u", task.script, *arguments]


def _task_environment(
    task: TaskSpec,
    *,
    current_blackjax: Path,
    historical_blackjax: Path,
    tuningfork: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    blackjax_root = (
        historical_blackjax
        if task.environment == "historical"
        else current_blackjax
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(blackjax_root), str(tuningfork))
            ),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "JAX_PLATFORM_NAME": "cpu",
            "JAX_ENABLE_X64": "1",
            "MPLCONFIGDIR": "/tmp/paper1-matplotlib",
        }
    )
    portable = {
        "PYTHONPATH": (
            "<historical-blackjax>:<tuningfork>"
            if task.environment == "historical"
            else "<current-blackjax>:<tuningfork>"
        ),
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_ENABLE_X64": "1",
        "MPLCONFIGDIR": "/tmp/paper1-matplotlib",
    }
    runcap = os.environ.get("PAPER1_RUNCAP")
    if runcap is not None:
        env["PAPER1_RUNCAP"] = runcap
        portable["PAPER1_RUNCAP"] = runcap
    return env, portable


def _portable_log_line(
    line: str,
    *,
    source_dir: Path,
    output_dir: Path,
    python: Path,
    current_blackjax: Path,
    historical_blackjax: Path,
    tuningfork: Path,
) -> str:
    replacements = (
        (str(output_dir), "<run-dir>"),
        (str(source_dir), "<experiments>"),
        (str(current_blackjax), "<current-blackjax>"),
        (str(historical_blackjax), "<historical-blackjax>"),
        (str(tuningfork), "<tuningfork>"),
        (str(python), "<python>"),
        ("/tmp/paper1-matplotlib", "<matplotlib-cache>"),
        (str(Path.home()), "<home>"),
    )
    portable = line
    for local, replacement in replacements:
        if local:
            portable = portable.replace(local, replacement)
    return portable


def _run_task(
    task: TaskSpec,
    *,
    source_dir: Path,
    output_dir: Path,
    run_id: str,
    python: Path,
    current_blackjax: Path,
    historical_blackjax: Path,
    tuningfork: Path,
) -> dict[str, Any]:
    script_path = source_dir / task.script
    if not script_path.is_file():
        raise CorpusRunError(f"missing producer script {task.script}")
    arguments = [
        _resolve_template(
            value,
            output_dir=output_dir,
            run_id=run_id,
            python=python,
            current_blackjax=current_blackjax,
            historical_blackjax=historical_blackjax,
            tuningfork=tuningfork,
        )
        for value in task.arguments
    ]
    command = [str(python), "-u", str(script_path), *arguments]
    env, portable_env = _task_environment(
        task,
        current_blackjax=current_blackjax,
        historical_blackjax=historical_blackjax,
        tuningfork=tuningfork,
    )
    log_path = output_dir / "logs" / f"{task.task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(UTC)
    started = time.monotonic()
    print(f"[corpus] START {task.task_id}", flush=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=source_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                portable_line = _portable_log_line(
                    line,
                    source_dir=source_dir,
                    output_dir=output_dir,
                    python=python,
                    current_blackjax=current_blackjax,
                    historical_blackjax=historical_blackjax,
                    tuningfork=tuningfork,
                )
                log.write(portable_line)
                log.flush()
                print(
                    f"[{task.task_id}] {portable_line}",
                    end="",
                    flush=True,
                )
        returncode = process.wait()
        log.flush()
        os.fsync(log.fileno())
    elapsed = time.monotonic() - started
    observed_outputs: dict[str, dict[str, Any]] = {}
    missing_outputs: list[str] = []
    for relative in task.expected_outputs:
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing_outputs.append(relative)
        else:
            observed_outputs[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    record = {
        "record_type": "task_result",
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "suite": task.suite,
        "environment": task.environment,
        "command": _portable_command(task, python.name),
        "environment_overrides": portable_env,
        "started_utc": started_utc.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "log": {
            "path": log_path.relative_to(output_dir).as_posix(),
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        },
        "outputs": observed_outputs,
        "missing_outputs": missing_outputs,
    }
    status = "PASS" if returncode == 0 and not missing_outputs else "FAIL"
    print(
        f"[corpus] {status} {task.task_id} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return record


def _all_output_hashes(
    output_dir: Path,
    *,
    excluded: Iterable[Path] = (),
) -> dict[str, dict[str, Any]]:
    excluded_resolved = {path.resolve() for path in excluded}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.resolve() in excluded_resolved:
            continue
        relative = path.relative_to(output_dir).as_posix()
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, or dash"
        )
    return value


def _absolute_without_resolving(path: Path) -> Path:
    """Make a CLI path absolute while preserving venv interpreter symlinks."""

    return Path(os.path.abspath(path.expanduser()))


def _release_resource_limits() -> dict[str, str]:
    runcap = os.environ.get("PAPER1_RUNCAP")
    if runcap != RELEASE_RUNCAP:
        raise CorpusRunError(
            f"PAPER1_RUNCAP must be {RELEASE_RUNCAP!r} for a release run"
        )
    return {
        "outer_memory_cap": runcap,
        "mechanism": "runcap",
        "scope": "full_corpus_orchestrator_and_children",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=_validate_run_id)
    parser.add_argument("--current-blackjax-root", required=True, type=Path)
    parser.add_argument("--historical-blackjax-root", required=True, type=Path)
    parser.add_argument("--tuningfork-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a downscaled cell from every producer before the full corpus.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    source_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    current_blackjax = args.current_blackjax_root.resolve()
    historical_blackjax = args.historical_blackjax_root.resolve()
    tuningfork = args.tuningfork_root.resolve()
    python = _absolute_without_resolving(args.python)
    plan = build_plan(smoke=args.smoke)
    resource_limits = _release_resource_limits()

    if output_dir.exists():
        raise CorpusRunError(
            "output directory already exists; corpus results are immutable"
        )
    dependencies = {
        "current_blackjax": _git_state(
            current_blackjax, CURRENT_BLACKJAX_SHA, "blackjax"
        ),
        "historical_blackjax": _git_state(
            historical_blackjax, HISTORICAL_BLACKJAX_SHA, "blackjax"
        ),
        "tuningfork": _git_state(tuningfork, TUNINGFORK_SHA, "tuningfork"),
    }
    interpreter = _interpreter_state(python)
    for task in plan:
        if not (source_dir / task.script).is_file():
            raise CorpusRunError(f"missing producer script {task.script}")

    output_dir.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(UTC)
    source_hashes = _source_hashes(source_dir)
    section_counts = (
        SMOKE_SECTION_COUNTS if args.smoke else FULL_SECTION_COUNTS
    )
    _write_json_exclusive(
        output_dir / "orchestrator_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "started",
            "mode": "smoke" if args.smoke else "full",
            "run_id": args.run_id,
            "created_utc": started_utc.isoformat(),
            "host": {
                "architecture": platform.machine(),
                "os": platform.system(),
            },
            "interpreter": interpreter,
            "dependencies": dependencies,
            "source_sha256": source_hashes,
            "planned_section_counts": section_counts,
            "resource_limits": resource_limits,
            "excluded_examples": ["gmm25"],
            "tasks": [
                {
                    **task.as_json(),
                    "command": _portable_command(task, python.name),
                }
                for task in plan
            ],
        },
    )

    records: list[dict[str, Any]] = []
    failed_task: str | None = None
    try:
        with _JsonlLedger(output_dir / "tasks.jsonl") as ledger:
            for task in plan:
                record = _run_task(
                    task,
                    source_dir=source_dir,
                    output_dir=output_dir,
                    run_id=args.run_id,
                    python=python,
                    current_blackjax=current_blackjax,
                    historical_blackjax=historical_blackjax,
                    tuningfork=tuningfork,
                )
                records.append(record)
                ledger.emit(record)
                if record["returncode"] != 0 or record["missing_outputs"]:
                    failed_task = task.task_id
                    raise CorpusRunError(
                        f"producer {task.task_id} failed; see its immutable log"
                    )

            execution_manifest_path = output_dir / "execution_manifest.json"
            _write_json_exclusive(
                execution_manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "producers_complete",
                    "mode": "smoke" if args.smoke else "full",
                    "run_id": args.run_id,
                    "started_utc": started_utc.isoformat(),
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": sum(
                        record["elapsed_seconds"] for record in records
                    ),
                    "dependencies": dependencies,
                    "source_sha256": source_hashes,
                    "planned_section_counts": section_counts,
                    "resource_limits": resource_limits,
                    "excluded_examples": ["gmm25"],
                    "tasks": records,
                    "output_files": _all_output_hashes(
                        output_dir,
                        excluded=(
                            execution_manifest_path,
                            output_dir / "tasks.jsonl",
                        ),
                    ),
                },
            )

            validation_spec = TaskSpec(
                "validate-full-results",
                "full-corpus-validation",
                "current",
                "validate_full_results.py",
                (
                    "--result-dir",
                    "{output}",
                    "--summary-out",
                    "{output}/frozen_summary.json",
                    *(("--smoke",) if args.smoke else ()),
                ),
                ("frozen_summary.json",),
            )
            validation_record = _run_task(
                validation_spec,
                source_dir=source_dir,
                output_dir=output_dir,
                run_id=args.run_id,
                python=python,
                current_blackjax=current_blackjax,
                historical_blackjax=historical_blackjax,
                tuningfork=tuningfork,
            )
            records.append(validation_record)
            ledger.emit(validation_record)
            if (
                validation_record["returncode"] != 0
                or validation_record["missing_outputs"]
            ):
                failed_task = validation_spec.task_id
                raise CorpusRunError(
                    "full-corpus validation failed; see its immutable log"
                )

        run_manifest_path = output_dir / "run_manifest.json"
        _write_json_exclusive(
            run_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "mode": "smoke" if args.smoke else "full",
                "run_id": args.run_id,
                "started_utc": started_utc.isoformat(),
                "completed_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": sum(
                    record["elapsed_seconds"] for record in records
                ),
                "dependencies": dependencies,
                "source_sha256": source_hashes,
                "planned_section_counts": section_counts,
                "resource_limits": resource_limits,
                "excluded_examples": ["gmm25"],
                "tasks": records,
                "result_files": _all_output_hashes(
                    output_dir,
                    excluded=(run_manifest_path,),
                ),
            },
        )
    except BaseException as exc:
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "mode": "smoke" if args.smoke else "full",
                    "run_id": args.run_id,
                    "failed_utc": datetime.now(UTC).isoformat(),
                    "failed_task": failed_task,
                    "resource_limits": resource_limits,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_tasks": [
                        record["task_id"] for record in records
                    ],
                },
            )
        raise
    return output_dir / "run_manifest.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run(args)
    except (CorpusRunError, OSError, subprocess.SubprocessError) as exc:
        print(f"full corpus failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {"status": "complete", "manifest": manifest.name},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
