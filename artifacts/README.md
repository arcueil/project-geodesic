# Artifacts

Large immutable artifacts are published as GitHub release assets rather than
committed to Git history.

## Paper 1 circulation artifacts

The curated package is versioned in the Paper 1 directory. The following
standalone release assets are prepared, but the public large-asset upload is
pending:

- `universal-warmup-path-circulation-3262037.tar.zst` — curated manuscript and
  reproducibility package.
- `paper1-full-29d246-20260726-v1-npz.tar.zst` — immutable raw NumPy sidecars.

The SHA-256 records are stored beside the experiment metadata in
[`papers/01-universal-warmup-path/experiments`](../papers/01-universal-warmup-path/experiments/).
Always verify the downloaded sidecar before extraction.
