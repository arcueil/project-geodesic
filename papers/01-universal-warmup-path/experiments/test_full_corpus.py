"""Fast tests for full-corpus orchestration and summary invariants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_full_corpus import (
    FULL_FIXED_BLOCK_COUNTS,
    FULL_SECTION_COUNTS,
    TaskSpec,
    _absolute_without_resolving,
    _portable_command,
    _run_task,
    _write_json_exclusive,
    build_plan,
    sha256_file,
)
from revalidate_full_corpus import (
    ORIGINAL_VALIDATOR_SHA256,
    RevalidationError,
    build_execution_overlay,
)
from validate_full_results import (
    _required_sampling_divergence_fields,
    _validate_draw_artifact,
    _validate_efficiency_formulas,
    _walk_nonfinite,
    efficiency_estimands,
)
from suite_common import sha256_array


HERE = Path(__file__).resolve().parent


class PlanTests(unittest.TestCase):
    def test_full_plan_freezes_every_section(self) -> None:
        plan = build_plan(smoke=False)
        task_ids = [task.task_id for task in plan]
        expected_fixed = [
            f"fixed-{block}" for block in FULL_FIXED_BLOCK_COUNTS
        ]
        self.assertEqual(task_ids[: len(expected_fixed)], expected_fixed)
        self.assertEqual(
            task_ids[len(expected_fixed) :],
            [
                "kernel-family",
                "schedule-configuration",
                "restart-ablation",
                "shared-step-current",
                "shared-step-historical",
                "controlled-gmm",
                "bbp-calibration",
                "schedule-evidence-pdf",
                "schedule-evidence-png",
            ],
        )
        self.assertEqual(FULL_SECTION_COUNTS["fixed_cells"], 77)
        self.assertEqual(FULL_SECTION_COUNTS["gmm_cells"], 239)
        self.assertFalse(any("gmm25" in task.task_id for task in plan))

    def test_smoke_plan_exercises_every_producer(self) -> None:
        plan = build_plan(smoke=True)
        self.assertEqual(plan[0].task_id, "fixed-smoke")
        self.assertEqual(len(plan), 10)
        gmm = next(task for task in plan if task.task_id == "controlled-gmm")
        self.assertIn("--smoke", gmm.arguments)
        schedule = next(
            task
            for task in plan
            if task.task_id == "schedule-evidence-pdf"
        )
        self.assertIn("1.5", schedule.arguments)
        self.assertIn("gmm_k2_primary_60k", schedule.arguments)
        self.assertTrue(
            any(task.task_id == "schedule-evidence-png" for task in plan)
        )

    def test_portable_commands_do_not_record_local_roots(self) -> None:
        task = build_plan(smoke=False)[0]
        command = _portable_command(task, "python")
        self.assertEqual(command[:3], ["python", "-u", "run_fixed_suite.py"])
        self.assertTrue(any("<run-dir>" in value for value in command))
        self.assertFalse(any(Path(value).is_absolute() for value in command))

    def test_divergence_schema_matches_population_row_shape(self) -> None:
        self.assertEqual(
            _required_sampling_divergence_fields(
                "manual-population-illcond"
            ),
            (
                "sampling_divergences_per_chain",
                "sampling_divergences_all_chains",
            ),
        )
        self.assertEqual(
            _required_sampling_divergence_fields("manual-illcond"),
            ("sampling_divergences",),
        )

    def test_validator_overlay_changes_no_producer_hashes(self) -> None:
        execution = {
            "source_sha256": {
                "run_fixed_suite.py": "1" * 64,
                "validate_full_results.py": ORIGINAL_VALIDATOR_SHA256,
            }
        }
        repaired = "2" * 64
        overlay = build_execution_overlay(
            execution,
            repaired_validator_sha256=repaired,
        )
        self.assertEqual(
            overlay["source_sha256"]["run_fixed_suite.py"],
            execution["source_sha256"]["run_fixed_suite.py"],
        )
        self.assertEqual(
            overlay["source_sha256"]["validate_full_results.py"],
            repaired,
        )
        self.assertEqual(
            execution["source_sha256"]["validate_full_results.py"],
            ORIGINAL_VALIDATOR_SHA256,
        )
        self.assertFalse(
            overlay["validation_overlay"]["raw_results_changed"]
        )

    def test_validator_overlay_rejects_unrelated_lineage(self) -> None:
        with self.assertRaises(RevalidationError):
            build_execution_overlay(
                {
                    "source_sha256": {
                        "validate_full_results.py": "0" * 64
                    }
                },
                repaired_validator_sha256="2" * 64,
            )


class ArtifactMetadataTests(unittest.TestCase):
    def test_human_readable_patch_metadata_and_checksum(self) -> None:
        manifest = json.loads(
            (HERE / "blackjax-patch-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        patch_path = HERE / manifest["series"]["file"]
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertNotIn("OpenAI Codex", patch_text)
        self.assertEqual(
            sha256_file(patch_path),
            manifest["series"]["sha256"],
        )
        self.assertEqual(
            patch_path.stat().st_size,
            manifest["series"]["size_bytes"],
        )
        self.assertEqual(
            manifest["reconstruction"]["source_equivalent_patch"][
                "expected_tree"
            ],
            "769093b891ff539a9a9acbe97c1cea95645d3842",
        )
        self.assertEqual(
            manifest["reconstruction"]["exact_commit"]["expected_head"],
            "29d2468857be4de1644ca4470c2a4aa7f8137656",
        )

    def test_artifact_manifest_declares_path_convention(self) -> None:
        manifest = json.loads(
            (HERE / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["path_convention"]["absolute_paths"], "none")
        for artifact in manifest["artifacts"]:
            if artifact["id"].startswith("gmm_") and "output" in artifact:
                self.assertTrue(artifact["output"].startswith("gmm/"))


class ImmutabilityTests(unittest.TestCase):
    def test_shipped_runcap_enforces_and_records_the_same_limit(self) -> None:
        wrapper = HERE / "runcap"
        self.assertTrue(os.access(wrapper, os.X_OK))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_systemd = root / "systemd-run"
            captured = root / "systemd-arguments.txt"
            fake_systemd.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$@\" > \"$PAPER1_TEST_SYSTEMD_ARGS\"\n"
                "while [[ $# -gt 0 && \"$1\" != -- ]]; do shift; done\n"
                "shift\n"
                "exec \"$@\"\n",
                encoding="utf-8",
            )
            fake_systemd.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{root}:{environment['PATH']}"
            environment["PAPER1_TEST_SYSTEMD_ARGS"] = str(captured)
            result = subprocess.run(
                [
                    str(wrapper),
                    "80G",
                    sys.executable,
                    "-c",
                    (
                        "import json;"
                        "from run_full_corpus import _release_resource_limits;"
                        "print(json.dumps(_release_resource_limits(),"
                        "sort_keys=True))"
                    ),
                ],
                cwd=HERE,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "mechanism": "runcap",
                    "outer_memory_cap": "80G",
                    "scope": "full_corpus_orchestrator_and_children",
                },
            )
            arguments = captured.read_text(encoding="utf-8").splitlines()
            self.assertIn("--user", arguments)
            self.assertIn("--scope", arguments)
            self.assertIn("--same-dir", arguments)
            self.assertIn("--property=MemoryMax=80G", arguments)
            self.assertIn("--property=MemorySwapMax=80G", arguments)

    def test_interpreter_path_keeps_venv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base-python"
            target.touch()
            link = root / "venv-python"
            link.symlink_to(target)
            observed = _absolute_without_resolving(link)
            self.assertEqual(observed, link.absolute())
            self.assertTrue(observed.is_symlink())

    def test_json_manifest_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            _write_json_exclusive(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(path, {"status": "second"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "first"},
            )

    def test_task_runner_streams_log_and_checks_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "result"
            source.mkdir()
            output.mkdir()
            producer = source / "producer.py"
            producer.write_text(
                "import os,pathlib,sys\n"
                "path=pathlib.Path(sys.argv[sys.argv.index('--out')+1])\n"
                "path.parent.mkdir(parents=True,exist_ok=True)\n"
                "path.write_text(os.environ['PYTHONPATH'],encoding='utf-8')\n"
                "print(f'producer complete {path}',flush=True)\n",
                encoding="utf-8",
            )
            task = TaskSpec(
                "synthetic",
                "synthetic",
                "historical",
                "producer.py",
                ("--out", "{output}/data.txt"),
                ("data.txt",),
                1,
            )
            record = _run_task(
                task,
                source_dir=source,
                output_dir=output,
                run_id="unit",
                python=Path(sys.executable),
                current_blackjax=root / "current",
                historical_blackjax=root / "historical",
                tuningfork=root / "tuningfork",
            )
            self.assertEqual(record["returncode"], 0)
            self.assertEqual(record["missing_outputs"], [])
            self.assertEqual(
                (output / "data.txt").read_text(encoding="utf-8"),
                f"{root / 'historical'}:{root / 'tuningfork'}",
            )
            log = (output / "logs" / "synthetic.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("producer complete <run-dir>/data.txt", log)
            self.assertNotIn(str(root), log)
            self.assertEqual(
                record["environment_overrides"]["PYTHONPATH"],
                "<historical-blackjax>:<tuningfork>",
            )


class EstimandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "comparison_policy": "automatic_joint_population",
            "n_chains": 2,
            "warmup_grads_all_chains": 100.0,
            "sampling_grads": 50.0,
            "sampling_grads_all_chains": 110.0,
            "min_bulk_ess": 20.0,
            "min_bulk_ess_pooled": 42.0,
            "ess_per_grad_marginal_amortized": 0.2,
            "ess_per_grad_one_output_total": 20.0 / 150.0,
            "ess_per_grad_pooled_population": 0.2,
            "population_quality_gate": {"passed": True},
        }

    def test_three_estimands_remain_separate(self) -> None:
        estimands = efficiency_estimands(self.row)
        self.assertEqual(estimands["marginal_amortized_chain0"], 0.2)
        self.assertAlmostEqual(
            estimands["one_output_total"],
            20.0 / 150.0,
        )
        self.assertEqual(estimands["pooled_population_total"], 0.2)
        self.assertIs(estimands["population_quality_passed"], True)
        self.assertEqual(
            _validate_efficiency_formulas(self.row, prefix="unit"),
            [],
        )

    def test_corrupt_rate_is_rejected(self) -> None:
        self.row["ess_per_grad_pooled_population"] = 0.3
        errors = _validate_efficiency_formulas(self.row, prefix="unit")
        self.assertTrue(any("pooled-population" in error for error in errors))

    def test_per_chain_marginal_rates_are_checked(self) -> None:
        self.row.update(
            {
                "min_bulk_ess_per_chain": [20.0, 30.0],
                "sampling_grads_per_chain": [50.0, 70.0],
                "ess_per_grad_marginal_per_chain": [
                    20.0 / 100.0,
                    30.0 / 120.0,
                ],
            }
        )
        self.assertEqual(
            _validate_efficiency_formulas(self.row, prefix="unit"),
            [],
        )
        self.row["ess_per_grad_marginal_per_chain"][1] = 0.5
        errors = _validate_efficiency_formulas(self.row, prefix="unit")
        self.assertTrue(any("marginal rate" in error for error in errors))

    def test_historical_one_output_rate_is_checked(self) -> None:
        row = {
            "comparison_policy": "historical_single_chain_nominal_B",
            "min_bulk_ess": 20.0,
            "warmup_grads": 50.0,
            "sampling_grads": 50.0,
            "ess_per_grad_one_output_total": 0.2,
        }
        self.assertEqual(
            _validate_efficiency_formulas(row, prefix="unit"),
            [],
        )
        row["ess_per_grad_one_output_total"] = 0.3
        errors = _validate_efficiency_formulas(row, prefix="unit")
        self.assertTrue(
            any("historical one-output" in error for error in errors)
        )

    def test_nonfinite_walk_is_recursive(self) -> None:
        errors = _walk_nonfinite(
            {"outer": [{"valid": 1.0}, {"invalid": float("inf")}]},
            "row",
        )
        self.assertEqual(errors, ["row.outer[1].invalid is non-finite"])


class DrawArtifactTests(unittest.TestCase):
    def test_npz_is_semantically_joined_to_the_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "draws.npz"
            arrays = {
                "draws": np.arange(12.0).reshape(2, 3, 2),
                "initial_positions": np.zeros((2, 2)),
                "warmup_end_positions": np.ones((2, 2)),
                "warmup_integration_steps": np.array([[1, 2], [3, 4]]),
                "warmup_divergences": np.zeros((2, 2), dtype=bool),
                "sampling_integration_steps": np.array(
                    [[1, 2, 3], [4, 5, 6]]
                ),
                "sampling_divergences": np.zeros((2, 3), dtype=bool),
            }
            np.savez_compressed(path, **arrays)
            members = {
                name: {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in arrays.items()
            }
            row = {
                "n_chains": 2,
                "population_sampling_performed": True,
                "comparison_policy": "automatic_joint_population",
                "num_sampling_draws": 3,
                "dimension": 2,
                "draws_artifact": {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "members": members,
                },
                "draws_sha256_per_chain": [
                    sha256_array(arrays["draws"][index])
                    for index in range(2)
                ],
                "warmup_grads_all_chains": 10,
                "warmup_grads_per_chain": [4, 6],
                "sampling_grads_all_chains": 21,
                "sampling_grads_per_chain": [6, 15],
                "sampling_divergences_per_chain": [0, 0],
                "sampling_divergences_all_chains": 0,
            }
            self.assertEqual(
                _validate_draw_artifact(
                    row,
                    artifact_dir=root,
                    prefix="unit",
                ),
                [],
            )
            row["draws_sha256_per_chain"][0] = "0" * 64
            errors = _validate_draw_artifact(
                row,
                artifact_dir=root,
                prefix="unit",
            )
            self.assertTrue(
                any("per-chain draw SHA-256" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
