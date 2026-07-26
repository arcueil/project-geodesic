# adaptation_paper — Paper 1: The Universal Warmup Path

Circulation source for the BlackJAX adaptation paper. The paper presents a
guarded route-plus-metric warmup procedure: a universal covariance reference,
route-indexed metric targets, finite-transcript limits, and refusal when a
constant metric or local evidence is not enough.

## Public files

- `universal_warmup_path_v1.tex` — manuscript source.
- `references.bib` — bibliography.
- `figures/figure_bbp.pdf` — iid spiked-PCA calibration.
- `figures/gmm_money_panel.png` — canonical current-corpus \(k=2\) GMM panel,
  generated from the frozen primary JSONL.
- `figures/schedule_evidence.pdf` — validated schedule prescription and
  controller-trace figure.

The GMM panel and the full appendix table both report the evaluated final
implementation.

## Repository-only audit

`QA_FINDINGS.md` records the repository claim, number, source, build, and visual
audit. It is versioned for repository maintenance but excluded from the
curated circulation artifact.

## Empirical authority

The canonical number source is:

`experiments/results/paper1-full-29d246-20260726-v1-validation-v2/frozen_summary.json`

Its validation attestation is beside it. Checksummed raw inputs are under:

`experiments/results/paper1-full-29d246-20260726-v1/`

The controller revision is
`29d2468857be4de1644ca4470c2a4aa7f8137656`. The shared-step historical
comparator is `2f62921848a93e7dc544ba9de8e29ef177e373b6`.

## Build

From this directory:

```sh
tectonic --only-cached universal_warmup_path_v1.tex
```

The circulation build was validated with Tectonic 0.16.9. No production sweep
is required to build or audit the paper. `experiments/README.md` is the
authority for the Python environment and empirical reproduction commands.

## Claim scope

- “Universal” describes the guarded procedure, not a universally optimal
  estimator or schedule.
- The covariance reference and route-indexed theorem targets remain distinct.
- Deployed scalar gates do not implement the theorem-level confidence-ball
  conjunction.
- Funnels and scale coupling route to reparameterization. Genuinely regional
  mixtures route to the companion population method over regional attractors.
- A selected mixture metric remains local or within-chain-conditioned;
  `global_exploration` is `not_established`.
- Refusal is a supported controller outcome and connects the companion
  local-trajectory sampler to the companion population/regional ensemble.

## Current status

The repaired full corpus is integrated, and the manuscript, source audit,
figures, references, and build have completed the repository circulation
checks. The author name is intentionally the only circulation metadata;
acknowledgements and affiliation remain author-owned.
