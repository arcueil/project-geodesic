"""Fast consistency tests for the consumer-facing artifact metadata."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "paper1-full-29d246-20260726-v1-npz.tar.zst"
ARCHIVE_SHA256 = (
    "227c74b129570662d27200a208822bcb8bf2a0940cccca1521da091c2cde7fe6"
)
ARCHIVE_SIZE = 470882063
MEMBER_LIST_SHA256 = (
    "989d0abe8fedb63bc20a3a3d9be6af15bc1073af0b12ff0cd306f0d02d3edf2e"
)
CHECKSUM_LIST_SHA256 = (
    "68a3d3c52101296044b8dbe66455d26775a60af568cb462a8bf90e0a8ce37f19"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(artifact_id: str) -> dict[str, object]:
    manifest = json.loads(
        (ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    matches = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["id"] == artifact_id
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {artifact_id!r} artifact, found {len(matches)}"
        )
    return matches[0]


class ArchiveMetadataTests(unittest.TestCase):
    def test_archive_descriptor_and_manifest_agree(self) -> None:
        descriptor = ROOT / f"{ARCHIVE_NAME}.sha256"
        expected_line = f"{ARCHIVE_SHA256}  {ARCHIVE_NAME}\n"
        self.assertEqual(descriptor.read_text(encoding="utf-8"), expected_line)

        artifact = _artifact("full_corpus_raw_npz_archive")
        self.assertEqual(artifact["filename"], ARCHIVE_NAME)
        self.assertEqual(artifact["size_bytes"], ARCHIVE_SIZE)
        self.assertEqual(artifact["sha256"], ARCHIVE_SHA256)
        self.assertEqual(artifact["member_count"], 77)
        self.assertEqual(
            artifact["member_list_sha256"],
            MEMBER_LIST_SHA256,
        )
        self.assertEqual(
            artifact["execution_manifest_checksum_list_sha256"],
            CHECKSUM_LIST_SHA256,
        )
        self.assertEqual(artifact["archive_member_mtime_epoch"], 0)

        checksum = artifact["checksum_descriptor"]
        self.assertEqual(checksum["path"], descriptor.name)
        self.assertEqual(checksum["sha256"], _sha256(descriptor))

        extraction = artifact["extraction_policy"]
        self.assertIs(extraction["overwrite"], False)
        self.assertEqual(
            extraction["target"],
            "results/paper1-full-29d246-20260726-v1",
        )
        arguments = extraction["gnu_tar_arguments"]
        self.assertIn("--zstd", arguments)
        self.assertIn("--keep-old-files", arguments)

    def test_metadata_checksum_file_is_current(self) -> None:
        checksum_file = ROOT / "reproducibility-metadata.sha256"
        for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, relative = raw_line.split("  ", 1)
            self.assertEqual(_sha256(ROOT / relative), expected, relative)


class AnalyticGeometryTests(unittest.TestCase):
    def test_manuscript_denseness_values_are_frozen(self) -> None:
        artifact = _artifact("gmm_analytic_denseness")
        observed = artifact["correlated_axes"]
        expected = [
            (2, 1.9130434782608696, 1),
            (3, 2.75, 3),
            (4, 3.52, 6),
            (5, 4.230769230769231, 10),
        ]
        self.assertEqual(len(observed), len(expected))

        for row, (k, spike, off_diagonals) in zip(observed, expected):
            self.assertEqual(row["k"], k)
            self.assertAlmostEqual(
                row["marginal_whitened_spike"],
                1.0 + (k - 1) * 21.0 / (k + 21.0),
            )
            self.assertEqual(row["marginal_whitened_spike"], spike)
            self.assertEqual(
                row["off_diagonal_correlations"],
                math.comb(k, 2),
            )
            self.assertEqual(row["off_diagonal_correlations"], off_diagonals)

        self.assertEqual(
            [f"{row['marginal_whitened_spike']:.3f}" for row in observed],
            ["1.913", "2.750", "3.520", "4.231"],
        )
        self.assertEqual(artifact["invariance_tolerance"], 1e-12)
        self.assertIs(artifact["mcmc_rerun"], False)


class ReadmeCommandTests(unittest.TestCase):
    def test_reporters_use_locked_interpreter_after_bootstrap(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assignment = readme.index(
            'REPRO_PYTHON="$REPRO_ROOT/.venv/bin/python"'
        )
        commands = list(
            re.finditer(
                r'(?m)^\s*(?P<prefix>"\$REPRO_PYTHON"|python)'
                r"\s+report_paper_numbers\.py\b",
                readme,
            )
        )
        self.assertGreater(len(commands), 0)
        self.assertTrue(
            all(match.group("prefix") == '"$REPRO_PYTHON"' for match in commands)
        )
        self.assertTrue(all(match.start() > assignment for match in commands))

    def test_required_consumer_checks_are_documented(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for required in (
            "jq --version",
            "tar --version",
            "zstd --version",
            "--keep-old-files",
            ARCHIVE_SHA256,
            str(ARCHIVE_SIZE),
            MEMBER_LIST_SHA256,
            CHECKSUM_LIST_SHA256,
            "--analytic",
            "--correlated-axes",
            "post-warmup sampling divergences are zero",
        ):
            self.assertIn(required, readme)


if __name__ == "__main__":
    unittest.main()
