# Project Geodesic: research program

## Central question

How can a sampler recover and use global posterior geometry when its direct
observations are local, correlated, finite, and potentially confined to only
part of the target?

The program separates three jobs that are often bundled into one sampler:

1. infer and assess a geometric reference;
2. move efficiently conditional on that reference; and
3. represent and explore multiple regional geometries when one constant metric
   is inadequate.

The posterior covariance $\Sigma_\pi$ provides a common global reference
across the program. A deployable metric remains conditional on the route and
the regions represented in the finite transcript. Agreement between local and
global geometry licenses efficient specialization; persistent disagreement is
evidence to wait, reparameterize, or use a population method—not evidence to
certify unseen-region coverage.

## Paper 1 — The Universal Warmup Path

**Question.** Given a finite warmup budget, when does the observed transcript
support changing metric structure, continuing adaptation, or refusing a local
constant-metric answer?

**Scope.** The paper develops a guarded act/wait/refuse warmup path. It relates
the global covariance reference to route-indexed metric targets, makes the
finite-transcript limitation explicit, and evaluates a gradient-based
realization against predetermined warmup configurations.

**Boundary.** A selected metric is not a certificate of global exploration. A
persistent within/between-chain split can instead identify a regional-geometry
problem and hand it to Paper 3.

**Status.** Circulation draft, frozen experiment corpus, and reproduction
package are available in
[`papers/01-universal-warmup-path`](papers/01-universal-warmup-path/).

## Paper 2 — The Universal Sampler

**Question.** Once an appropriate metric has been identified, how should a
sampler tune its local trajectory while preserving the target distribution?

**Scope.** The proposed paper studies Gibbs self-tuning (GIST) kernels as the
local-motion counterpart to Paper 1. Paper 1 decides which geometry the current
evidence supports; Paper 2 studies how to move efficiently under that geometry.

**Boundary.** Efficient local motion cannot by itself establish mode coverage
or repair a globally inappropriate constant metric.

**Status.** The kernel family is implemented in BlackJAX. A paper-level theory
and comparative empirical campaign remain to be completed.

## Paper 3 — The Nested Ensemble

**Question.** What replaces a single global metric when posterior geometry is
regional, as in separated mixtures or continuously changing scale?

**Scope.** The proposal treats global geometry as a field of regional metric
targets. Inner populations identify and exploit regional structure; an outer
population mechanism exchanges information and mass across regions. The
single-region limit recovers the first two papers.

**Boundary.** Multi-chain disagreement is a routing signal, not a proof that
all relevant regions have been found or correctly weighted. Global exploration
and normalization require the population-level mechanism and separate
diagnostics.

**Status.** Conceptual proposal. The generic and 25-mode mixture examples
deliberately reserved from Paper 1 belong here.

## Shared principles

- **One compass, multiple deployable targets.** $\Sigma_\pi$ is a common
  reference; route-conditioned and regional metrics are the objects a sampler
  can responsibly deploy.
- **Local and global awareness are distinct.** Local trajectory efficiency and
  global exploration require different evidence and different mechanisms.
- **Refusal is an algorithmic outcome.** A warmup routine should say when its
  evidence is insufficient or when the model needs a different
  parameterization or sampler class.
- **Evidence claims are one-sided.** Observed agreement can describe the
  transcript. It cannot certify unobserved regions.
- **Reproducibility is part of the result.** Each numerical claim should map to
  a frozen summary, immutable inputs, an environment record, and executable
  validation.

## Dependency map

Paper 1 supplies the geometric inference and routing layer. Paper 2 supplies
the local-motion layer. Paper 3 composes both inside a population architecture
that can represent multiple regional geometries.

The papers share notation and experimental infrastructure, but each must remain
independently readable and must state which guarantees belong to its own layer.
