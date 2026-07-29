# Paper 1: The Universal Warmup Path — Automatic Preconditioner Selection for HMC

**arXiv:** [arXiv:2607.23788](https://arxiv.org/abs/2607.23788)

Circulation source for the BlackJAX adaptation paper. The paper presents a
multi-chain automatic warmup controller for HMC. It begins with a diagonal
inverse mass matrix, evaluates evidence at dimension-derived scheduled window
endpoints, and either retains the diagonal matrix or promotes to a
low-rank-plus-diagonal constant preconditioner, selecting the retained rank
subject to dimension and sample-support caps. Inconclusive evidence triggers
another scheduled warmup window. The final output includes the frozen step
size and inverse mass matrix; the verdict reports `route` and
`effective_rank`, while its flags report `metric_scope`,
`observed_ensemble_evidence`, and `handoff`. Poor held-out score–position
linearity produces reparameterization advice, while persistent
within-/between-chain disagreement produces advice to use a population or
tempering method for regional exploration.

## Public files

- `universal_warmup_path_v2.tex` — manuscript source.
- `references.bib` — bibliography.
- `figures/figure_bbp.pdf` — iid spiked-PCA calibration.
- `figures/gmm_money_panel.png` — canonical current-corpus \(k=2\) GMM panel,
  generated from the frozen experiment JSONL.
- `figures/schedule_evidence.pdf` — validated schedule prescription and
  controller-trace figure.

The GMM panel and the full appendix table both report the evaluated final
implementation.

## Repository-only audit

`QA_FINDINGS.md` records the repository claim, number, source, build, and visual
audit. It is versioned for repository maintenance but excluded from the
curated circulation artifact.

## Empirical authority

The public experiment package is archived in the
[Project Geodesic Paper 1 experiment folder](https://github.com/arcueil/project-geodesic/tree/paper1-circulation-v1/papers/01-universal-warmup-path/experiments).
The `paper1-circulation-v1` tag pins the circulation copy.

The canonical number source is:

`experiments/results/paper1-full-29d246-20260726-v1-validation-v2/frozen_summary.json`

Its validation attestation is beside it. Checksummed raw inputs are under:

`experiments/results/paper1-full-29d246-20260726-v1/`

The frozen corpus was executed at controller revision
`29d2468857be4de1644ca4470c2a4aa7f8137656`. Its
`blackjax/adaptation/meta` tree was merged unchanged as BlackJAX revision
`2103a4275b4d29b1650ba06458d5703eb7302b2e` in PR
[#1011](https://github.com/blackjax-devs/blackjax/pull/1011). The shared-step
historical comparator is `2f62921848a93e7dc544ba9de8e29ef177e373b6`.

## Build

From this directory:

```sh
tectonic --only-cached universal_warmup_path_v2.tex
```

The circulation build was validated with Tectonic 0.16.9. No production sweep
is required to build or audit the paper. `experiments/README.md` is the
authority for the Python environment and empirical reproduction commands.

## Claim scope

- “Universal” means one automatic controller across the evaluated HMC-family
  kernels. The method is scoped to between-window selection of a constant
  Euclidean inverse mass matrix and step size for HMC.
- Euclidean HMC uses a step size and constant mass matrix. In the paper's
  BlackJAX convention, `inverse_mass_matrix` is the inverse of the HMC mass
  matrix and equals the target covariance for a Gaussian. Geometry and
  covariance statements use the unconstrained coordinates supplied to HMC.
- The posterior covariance reference and estimator-indexed theorem targets
  remain distinct. The theory analyzes confidence-calibrated selection; the
  evaluated controller uses fixed diagnostic thresholds.
- The controller adapts between warmup windows and freezes the selected
  constant inverse mass matrix and step size before posterior sampling;
  within-orbit and position-dependent adaptation are separate methods.
- Initial positions are caller supplied. Between-chain evidence reflects the
  regions exposed by those starts and the subsequent warmup.
- Estimator identification means recovering covariance structure or retained
  rank from the warmup transcript, not causal identification or
  statistical-model identifiability.
- A selected preconditioner retains its reported conditioning scope;
  `global_exploration` remains `not_established`.
- Empirical contribution and efficiency claims are restricted to the evaluated
  HMC-family kernels. The headline comparison is pooled ESS per gradient across
  eight warmup/sampling chains on the synthetic ill-conditioned Gaussian and
  German-credit Bayesian logistic-regression geometry benchmarks against
  prespecified Fisher low-rank warmup and Welford diagonal warmup baselines.
  German credit carries no causal interpretation. Both pooled baselines used
  312 warmup transitions per chain and the same proportional growing-window
  schedule; they differed in the mass-matrix estimator.

## Current status

The repaired full corpus is integrated, and the manuscript, source audit,
figures, references, and build have completed the repository circulation
checks. The author name is intentionally the only circulation metadata;
acknowledgements and affiliation remain author-owned.
