#!/usr/bin/env python3
"""Compare regenerated Paper 1 figures with canonical manuscript figures.

Matplotlib's PDF backend records a wall-clock CreationDate unless the producer
supplies custom metadata.  The reference producers did not override it.  This
verifier therefore compares PDF bytes after replacing exactly that metadata
field and compares the GMM PNG byte-for-byte.  It writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


FIGURES = (
    ("figure_bbp.pdf", "pdf_creation_date_normalized"),
    ("schedule_evidence.pdf", "pdf_creation_date_normalized"),
    ("gmm_money_panel.png", "byte_exact"),
)
CREATION_DATE = re.compile(rb"/CreationDate \(D:[^)]+\)")
NORMALIZED_CREATION_DATE = b"/CreationDate (D:<normalized>)"


class FigureVerificationError(ValueError):
    """Raised when figure inputs are missing or structurally unexpected."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_figure_bytes(path: Path, policy: str) -> bytes:
    value = path.read_bytes()
    if policy == "byte_exact":
        return value
    if policy != "pdf_creation_date_normalized":
        raise FigureVerificationError(f"unknown comparison policy {policy!r}")
    normalized, replacements = CREATION_DATE.subn(
        NORMALIZED_CREATION_DATE,
        value,
    )
    if replacements != 1:
        raise FigureVerificationError(
            f"{path} contains {replacements} PDF CreationDate fields; expected 1"
        )
    return normalized


def verify(reference_dir: Path, generated_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, policy in FIGURES:
        reference = reference_dir / name
        generated = generated_dir / name
        if not reference.is_file():
            raise FigureVerificationError(f"missing reference figure: {reference}")
        if not generated.is_file():
            raise FigureVerificationError(f"missing generated figure: {generated}")
        reference_raw = reference.read_bytes()
        generated_raw = generated.read_bytes()
        reference_content = normalized_figure_bytes(reference, policy)
        generated_content = normalized_figure_bytes(generated, policy)
        report[name] = {
            "policy": policy,
            "reference_raw_sha256": _sha256(reference_raw),
            "generated_raw_sha256": _sha256(generated_raw),
            "reference_content_sha256": _sha256(reference_content),
            "generated_content_sha256": _sha256(generated_content),
            "content_match": reference_content == generated_content,
        }
    return {
        "status": (
            "valid"
            if all(value["content_match"] for value in report.values())
            else "mismatch"
        ),
        "figures": report,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify(args.reference_dir, args.generated_dir)
    except (OSError, FigureVerificationError) as exc:
        print(f"figure verification failed: {exc}", file=sys.stderr)
        return 1
    json.dump(report, sys.stdout, allow_nan=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
