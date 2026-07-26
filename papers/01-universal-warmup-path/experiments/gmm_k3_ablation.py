"""Run the direct k=3 matched-metric GMM ablation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gmm_boundary import run_ablation


DEFAULT_SRS = (3.0, 3.5, 4.0, 4.5)
DEFAULT_SEEDS = tuple(range(1001, 1017))


def comma_separated_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def comma_separated_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct k=3 low-rank versus matched-diagonal GMM sweep."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--srs", type=comma_separated_floats, default=DEFAULT_SRS
    )
    parser.add_argument(
        "--seeds", type=comma_separated_ints, default=DEFAULT_SEEDS
    )
    parser.add_argument("--budget", type=int, default=60_000)
    parser.add_argument("--draws", type=int, default=4_000)
    args = parser.parse_args()

    with args.out.open("x", encoding="utf-8") as output:
        for seed in args.seeds:
            for sr in args.srs:
                started = time.monotonic()
                row = run_ablation(
                    sr,
                    seed=seed,
                    budget=args.budget,
                    M=8,
                    n_sample_draws=args.draws,
                    correlated_axes=3,
                )
                row["elapsed_seconds"] = time.monotonic() - started
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(
                    f"seed={seed} SR={sr:g} "
                    f"route={row['auto_route']} "
                    f"imm={row['auto_imm_kind']} "
                    f"ratio={row['projection_bulk_ess_per_grad_ratio']} "
                    f"error={row['error']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
