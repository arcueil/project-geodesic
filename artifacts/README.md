# Artifacts

Large immutable artifacts are published as GitHub release assets rather than
committed to Git history.

## Paper 1 circulation artifacts

The curated manuscript and reproducibility package is versioned in
[`papers/01-universal-warmup-path`](../papers/01-universal-warmup-path/).
The 471 MB archive of immutable raw NumPy sidecars is stored at:

`papers/01-universal-warmup-path/experiments/paper1-full-29d246-20260726-v1-npz.tar.zst`

The archive is a Git LFS object. After cloning, retrieve and verify it with:

```sh
git lfs pull --include="papers/01-universal-warmup-path/experiments/paper1-full-29d246-20260726-v1-npz.tar.zst"
cd papers/01-universal-warmup-path/experiments
sha256sum -c paper1-full-29d246-20260726-v1-npz.tar.zst.sha256
```

The adjacent metadata also records the exact size, member inventory, extraction
policy, and per-file integrity source. Always verify the sidecar before
extraction.
