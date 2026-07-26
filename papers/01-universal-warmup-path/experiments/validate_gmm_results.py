#!/usr/bin/env python
"""Validate a complete controlled-GMM result directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmm_suite import ValidationError, validate_suite_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GMM JSONL completeness, uniqueness, finiteness, and invariants."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validate the downscaled end-to-end grid",
    )
    args = parser.parse_args()

    try:
        summaries = validate_suite_dir(
            args.result_dir,
            smoke=args.smoke,
            require_run_manifest=True,
        )
    except ValidationError as exc:
        parser.exit(1, f"validation failed:\n{exc}\n")
    print(json.dumps({"status": "valid", "arms": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
