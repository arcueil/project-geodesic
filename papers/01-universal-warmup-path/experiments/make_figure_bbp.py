#!/usr/bin/env python
"""Figure 1 --- iid Gaussian spiked-PCA calibration.

Theory curves (closed-form BBP / Marchenko-Pastur) with fixed-seed spiked-Wishart
Monte-Carlo markers. The null lambda_max tracks (1+sqrt(gamma))^2; ordinary PCA
separates a spike and recovers its direction when
ell > 1+sqrt(gamma), i.e. n > d/(ell-1)^2 in this iid model.

Reproducible: numpy float64, seed 20260716. Regenerate:
    python experiments/make_figure_bbp.py --out figures/figure_bbp.pdf --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
})

# Okabe-Ito subset, validated colorblind-safe (dataviz validator, ALL PASS).
COL = {1.5: "#0072B2", 2.0: "#D55E00", 3.0: "#009E73"}
SPIKES = [1.5, 2.0, 3.0]
EDGE = "#222222"
BULK = "#d9d9d6"

def bbp_lambda(ell, g):
    """Sample top-eigenvalue location: separated above threshold, else at edge."""
    edge = (1 + np.sqrt(g)) ** 2
    sep = ell * (1 + g / (ell - 1))               # BGN / Paul outlier location
    return np.where(ell > 1 + np.sqrt(g), sep, edge)


def bbp_overlap(ell, g):
    """Squared eigenvector overlap; 0 below threshold (Paul 2007)."""
    val = (1 - g / (ell - 1) ** 2) / (1 + g / (ell - 1))
    return np.where(ell > 1 + np.sqrt(g), np.clip(val, 0, 1), 0.0)


def simulate(ell, g, rng, d=200, trials=24):
    """Spiked-Wishart MC: pop cov I + (ell-1) e1 e1^T; return mean lambda_max, overlap^2."""
    n = max(int(round(d / g)), 2)
    v = np.zeros(d); v[0] = 1.0
    scale = np.ones(d); scale[0] = np.sqrt(ell)     # sqrt of population eigenvalues
    lam, ov = [], []
    for _ in range(trials):
        X = (rng.standard_normal((n, d)) * scale)    # rows ~ N(0, diag)
        S = (X.T @ X) / n
        w, U = np.linalg.eigh(S)
        lam.append(w[-1])
        ov.append(U[:, -1].dot(v) ** 2)
    return np.mean(lam), np.mean(ov)


g = np.linspace(0.02, 1.3, 400)
g_mc = np.array([0.15, 0.35, 0.55, 0.75, 1.0, 1.2])

def render(output: Path, *, d: int = 200, trials: int = 24, force: bool = False) -> None:
    """Render the fixed-seed calibration to ``output``."""
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass --force to replace the derived figure"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260716)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # Panel (a): top eigenvalue versus the BBP edge.
    lo = (1 - np.sqrt(np.minimum(g, 1))) ** 2
    hi = (1 + np.sqrt(g)) ** 2
    axL.fill_between(g, lo, hi, color=BULK, zorder=0, label="MP bulk")
    axL.plot(
        g,
        hi,
        color=EDGE,
        lw=1.4,
        zorder=3,
        label=r"BBP edge $(1{+}\sqrt{\gamma})^2$",
    )

    null_lam = [simulate(1.0, gg, rng, d=d, trials=trials)[0] for gg in g_mc]
    axL.scatter(
        g_mc,
        null_lam,
        s=16,
        facecolor="none",
        edgecolor=EDGE,
        lw=0.8,
        zorder=4,
        label=r"null $\ell{=}1$ (sim)",
    )

    for ell in SPIKES:
        axL.plot(g, bbp_lambda(ell, g), color=COL[ell], lw=1.8, zorder=3)
        ys = [
            simulate(ell, gg, rng, d=d, trials=trials)[0] for gg in g_mc
        ]
        axL.scatter(g_mc, ys, s=20, color=COL[ell], zorder=5)
        axL.text(
            0.035,
            bbp_lambda(ell, 0.02) + 0.05,
            rf"$\ell={ell:g}$",
            color=COL[ell],
            fontsize=8,
            va="bottom",
            ha="left",
        )

    axL.set_xlabel(r"aspect ratio $\gamma = d/n$")
    axL.set_ylabel(r"top sample eigenvalue $\widehat{\lambda}_1$")
    axL.set_title(r"(a) detectability")
    axL.set_xlim(0, 1.3)
    axL.set_ylim(0.5, 6.2)
    axL.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.4,
        borderpad=0.2,
    )

    # Panel (b): eigenvector overlap.
    axR.axhline(0, color="#999999", lw=0.6, zorder=1)
    for ell in SPIKES:
        axR.plot(
            g,
            bbp_overlap(ell, g),
            color=COL[ell],
            lw=1.8,
            zorder=3,
            label=rf"$\ell={ell:g}$",
        )
        ys = [
            simulate(ell, gg, rng, d=d, trials=trials)[1] for gg in g_mc
        ]
        axR.scatter(g_mc, ys, s=20, color=COL[ell], zorder=5)
        threshold = (ell - 1) ** 2
        if threshold <= 1.3:
            axR.scatter(
                [threshold],
                [0.0],
                s=30,
                marker="v",
                color=COL[ell],
                zorder=6,
            )
    axR.set_xlabel(r"aspect ratio $\gamma = d/n$")
    axR.set_ylabel(
        r"eigenvector overlap $|\langle\widehat{v}_1,v_1\rangle|^2$"
    )
    axR.set_title(r"(b) direction recovery")
    axR.set_xlim(0, 1.3)
    axR.set_ylim(-0.05, 1.02)
    axR.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.4,
        borderpad=0.2,
        title=r"$\blacktriangledown$: threshold $\gamma=(\ell{-}1)^2$",
        title_fontsize=7,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "figures"
        / "figure_bbp.pdf",
    )
    parser.add_argument("--dimension", type=int, default=200)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dimension < 2:
        parser.error("--dimension must be at least 2")
    if args.trials < 1:
        parser.error("--trials must be positive")
    render(
        args.out,
        d=args.dimension,
        trials=args.trials,
        force=args.force,
    )


if __name__ == "__main__":
    main()
