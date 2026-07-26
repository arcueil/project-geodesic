# Paper 2 — The Universal Sampler

## Working proposal

Paper 2 studies the local-motion half of Project Geodesic. Conditional on a
metric supported by the warmup evidence, a Gibbs self-tuning (GIST) kernel
adapts trajectory choices locally while preserving the target distribution.

The intended contribution is not another global-coverage claim. It is a clean
separation of responsibilities:

- Paper 1 infers and assesses the geometric reference.
- Paper 2 moves efficiently under the supported reference.
- Paper 3 supplies the population mechanism when one reference is not globally
  adequate.

## Current status

The GIST kernel family is implemented in BlackJAX. Theory consolidation,
benchmark design, and comparative validation are pending. No manuscript or
headline empirical claim is frozen yet.
