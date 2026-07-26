# Paper 1 reproducibility artifact

This directory is the executable artifact for “The Universal Warmup Path.” It
contains the experiment producers, validation code, exact dependency pins,
recoverable figure generators, and the immutable reference corpus. The
25-component GMM example is intentionally outside this paper and outside the
corpus; the controlled mixture here is the five-dimensional, two-component
GMM.

All release producers run serially, force JAX onto CPU with 64-bit arithmetic,
record source and dependency provenance, and create outputs exclusively. A
rerun therefore requires a new output directory and cannot silently overwrite
an earlier result. `artifact_manifest.json` inventories the executable corpus;
`figure_manifest.json` records canonical figure hashes and source lineage.

## Reference artifact

The reference corpus and its validation attestation are:

```text
results/paper1-full-29d246-20260726-v1/
results/paper1-full-29d246-20260726-v1-validation-v2/
```

The first directory contains the immutable producer outputs. All 20 producers
completed successfully. The validator bundled with that execution then failed
on a schema-only check: it expected a scalar single-chain divergence field on
population-manual rows, although those rows correctly contain per-chain and
all-chain divergence fields. No producer or raw result failed. The corrected
validator was replayed without rerunning or modifying any producer output, and
the second directory records that lineage in a checksummed attestation.

The authoritative hashes are:

| Artifact | SHA-256 |
|---|---|
| source `execution_manifest.json` | `79bcb82b84b83bc9406b0694e8912dff63862dbd7f9ff309fd63e45eaa5a9ffb` |
| validation `execution_manifest.overlay.json` | `f86c1e02ba233ffabde97c61c0868e0be0991e48a506ad01608160c4ba5f8296` |
| validated `frozen_summary.json` | `94136b0e528b883148540efdbe43cd24ebd39bcaa905b86bd73a8c8d4763f24a` |
| `validation_attestation.json` | `85b60b2958cfc3395828c0236a77d1fb4ee7ded16a4249e499fff339cee60b02` |

`frozen_summary.json` is the only empirical manuscript-number source. The JSONL
and NPZ files are the checksummed inputs from which the validator constructs it.
`report_paper_numbers.py` accepts the packaged reference summary only after
checking both its SHA-256 and the SHA-256 of the packaged validation
attestation against the release anchors above. Copies at other filesystem
locations are supported because trust is content-based, not path-based.

## Quick metadata integrity check

Run these commands from this directory:

```bash
sha256sum -c reproducibility-metadata.sha256

(
  cd results/paper1-full-29d246-20260726-v1-validation-v2
  sha256sum -c checksums.sha256
)
```

These checks need no Python environment. The deterministic number reporter
requires SciPy even for its verification section, so every reporter command is
deliberately placed after creation of the locked environment and uses
`"$REPRO_PYTHON"`.

## Host prerequisites

The consumer workflow requires Git, `uv`, GNU `sha256sum`, `jq`, GNU tar with
Zstandard support, and `zstd`. Corpus reruns additionally require a functioning
systemd user manager for the supplied memory-cap wrapper. The consumer audit
used jq 1.7, GNU tar 1.35, and zstd 1.5.5; these are packaging and inspection
tools, not frozen numerical-run provenance.

Check the archive-facing tools before continuing:

```bash
jq --version
tar --version
zstd --version
sha256sum --version
```

## Exact source revisions

The experiment runner imports BlackJAX and tuningfork directly from clean
source trees. It refuses a dirty checkout or the wrong revision.

| Source | Revision | Use |
|---|---|---|
| BlackJAX, current | `29d2468857be4de1644ca4470c2a4aa7f8137656` | every current-controller producer |
| BlackJAX, historical | `2f62921848a93e7dc544ba9de8e29ef177e373b6` | sequential shared-step-size comparison only |
| tuningfork | `79ffb73250f5024dc511b3035d373d11474c2195` | models and experiment support |

The current controller was subsequently merged in
[BlackJAX PR #1011](https://github.com/blackjax-devs/blackjax/pull/1011) as
revision `2103a4275b4d29b1650ba06458d5703eb7302b2e`. The
`blackjax/adaptation/meta` tree is identical at the executed and merged
revisions. The table deliberately retains `29d2468857...` because every
producer and manifest records the exact source revision used by the frozen run.

The current BlackJAX target is supplied as an exact Git bundle because the
artifact was prepared before that revision was guaranteed to be reachable
from the public remote. Its public prerequisite is
`f8f7e3bfd04392ead392b7f5fe1938f0159f048d`; the target tree is
`769093b891ff539a9a9acbe97c1cea95645d3842`. The format-patch is an alternate
source-equivalent reconstruction. Its human-readable commit message omits a
tool coauthor trailer, without changing the diff or reconstructed source tree.
The bundle is the exact-source artifact: it preserves the original commit
identity and all original commit metadata.

The following creates all three clean source trees. Set `EXPERIMENTS_DIR` to
the absolute path of this directory:

```bash
REPRO_ROOT="$PWD/paper1-reproduction"
EXPERIMENTS_DIR="/absolute/path/to/paper/adaptation_paper/experiments"
CURRENT_BLACKJAX_ROOT="$REPRO_ROOT/blackjax-current"
HISTORICAL_BLACKJAX_ROOT="$REPRO_ROOT/blackjax-historical"
TUNINGFORK_ROOT="$REPRO_ROOT/tuningfork"

mkdir -p "$REPRO_ROOT"

git clone https://github.com/blackjax-devs/blackjax.git \
  "$CURRENT_BLACKJAX_ROOT"
git -C "$CURRENT_BLACKJAX_ROOT" checkout --detach \
  f8f7e3bfd04392ead392b7f5fe1938f0159f048d
git -C "$CURRENT_BLACKJAX_ROOT" bundle verify \
  "$EXPERIMENTS_DIR/blackjax-29d246-semantics-repair.bundle"
git -C "$CURRENT_BLACKJAX_ROOT" fetch \
  "$EXPERIMENTS_DIR/blackjax-29d246-semantics-repair.bundle" HEAD
git -C "$CURRENT_BLACKJAX_ROOT" checkout --detach FETCH_HEAD

git clone https://github.com/blackjax-devs/blackjax.git \
  "$HISTORICAL_BLACKJAX_ROOT"
git -C "$HISTORICAL_BLACKJAX_ROOT" checkout --detach \
  2f62921848a93e7dc544ba9de8e29ef177e373b6

git clone https://github.com/blackjax-devs/tuningfork.git \
  "$TUNINGFORK_ROOT"
git -C "$TUNINGFORK_ROOT" checkout --detach \
  79ffb73250f5024dc511b3035d373d11474c2195

git -C "$CURRENT_BLACKJAX_ROOT" rev-parse HEAD
git -C "$HISTORICAL_BLACKJAX_ROOT" rev-parse HEAD
git -C "$TUNINGFORK_ROOT" rev-parse HEAD
git -C "$CURRENT_BLACKJAX_ROOT" status --short
git -C "$HISTORICAL_BLACKJAX_ROOT" status --short
git -C "$TUNINGFORK_ROOT" status --short
```

The three `rev-parse` results must equal the table above, and all three status
commands must print nothing.

## Environment

The reference run used CPython 3.13.14 on 64-bit ARM Linux with one CPU JAX
device. JAX and jaxlib were both 0.11.0, and `JAX_ENABLE_X64=1` was active.
The principal numerical packages were ArviZ 1.1.0, Matplotlib 3.10.9, NumPy
2.4.6, NumPyro 0.21.0, and SciPy 1.17.1. Full versions are in
`environment.lock.txt`.

The lock is a version-pinned `pip freeze`, not a hash-locked wheel archive.
It excludes the two source-tree packages, BlackJAX and tuningfork, because
their Git revisions are recorded separately. The reference environment was
created with `uv`. The manuscript PDF is built separately with Tectonic
0.16.9; Tectonic is a document-build tool and is not part of the Python lock.
A clean Python recreation is:

```bash
uv python install 3.13.14
uv venv --python 3.13.14 "$REPRO_ROOT/.venv"
uv pip install \
  --python "$REPRO_ROOT/.venv/bin/python" \
  --requirements "$EXPERIMENTS_DIR/environment.lock.txt"

REPRO_PYTHON="$REPRO_ROOT/.venv/bin/python"
"$REPRO_PYTHON" -c \
  'import jax,platform; print(platform.python_version(), jax.__version__, jax.devices())'
```

The closest reference platform is aarch64 Linux with GNU `sha256sum`, Git,
`uv`, and a cgroup-capable `systemd`. Other CPU architectures can exercise the
code, but bitwise numerical and figure identity is only claimed for the
recorded environment. The GMM provenance records the platform string
`Linux-6.17.0-1026-nvidia-aarch64-with-glibc2.39`. Git, `uv`, CPU-model, and
`systemd` versions were not captured at run time, so current host values are
not frozen evidence for those tools.

All commands below assume this process environment:

```bash
cd "$EXPERIMENTS_DIR"
export PYTHONPATH="$CURRENT_BLACKJAX_ROOT:$TUNINGFORK_ROOT"
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/paper1-matplotlib
```

## Authenticate the reference report

Define the packaged reference paths once:

```bash
REFERENCE_RESULTS="$EXPERIMENTS_DIR/results/paper1-full-29d246-20260726-v1"
REFERENCE_SUMMARY="$EXPERIMENTS_DIR/results/paper1-full-29d246-20260726-v1-validation-v2/frozen_summary.json"
REFERENCE_ATTESTATION="$EXPERIMENTS_DIR/results/paper1-full-29d246-20260726-v1-validation-v2/validation_attestation.json"
```

The number reporter writes only deterministic JSON to standard output. It
authenticates the frozen summary and validation attestation against release
anchors, authenticates the source execution manifest through that pair, and
then verifies every mapped raw input:

```bash
"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section verification
```

`--attestation` may name an explicit attestation copy; by default the reporter
uses `validation_attestation.json` beside `--summary`.

The reported 36/36 population-quality result has one exact meaning: every
rank-normalized split-\(\widehat R\) value is finite, the maximum is at most
1.01, and post-warmup sampling divergences are zero. Warmup divergences are
reported separately and are not part of this gate. Passing it establishes
neither global exploration nor theorem-level routing confidence.

## Raw NPZ sidecar

The 77 raw numeric sidecars are delivered separately from Git as one
deterministic archive:

| Property | Release value |
|---|---|
| filename | `paper1-full-29d246-20260726-v1-npz.tar.zst` |
| compressed bytes | `470882063` |
| SHA-256 | `227c74b129570662d27200a208822bcb8bf2a0940cccca1521da091c2cde7fe6` |
| members | 77 files matching `fixed/*.arrays/*.npz` |
| member-list SHA-256 | `989d0abe8fedb63bc20a3a3d9be6af15bc1073af0b12ff0cd306f0d02d3edf2e` |
| execution-manifest checksum-list SHA-256 | `68a3d3c52101296044b8dbe66455d26775a60af568cb462a8bf90e0a8ce37f19` |
| archive member mtime | Unix epoch 0 |

Set `RAW_ARCHIVE_DIR` to the directory containing the separately downloaded
archive. Verify its tracked checksum descriptor, byte size, member count, and
member list before extraction:

```bash
RAW_NAME=paper1-full-29d246-20260726-v1-npz.tar.zst
RAW_ARCHIVE_DIR=/absolute/path/to/release-assets
RAW_ARCHIVE="$RAW_ARCHIVE_DIR/$RAW_NAME"

(
  cd "$RAW_ARCHIVE_DIR"
  sha256sum -c "$EXPERIMENTS_DIR/$RAW_NAME.sha256"
)

test "$(stat -c '%s' "$RAW_ARCHIVE")" -eq 470882063
test "$(tar --zstd -tf "$RAW_ARCHIVE" | wc -l)" -eq 77
tar --zstd -tf "$RAW_ARCHIVE" | sha256sum
```

The final command must print the member-list hash in the table. Derive the
per-file checksums from the authenticated execution manifest and verify the
checksum-list hash:

```bash
NPZ_CHECKSUMS="$REPRO_ROOT/paper1-reference-npz.sha256"

jq -r \
  '.output_files | to_entries[]
   | select(.key | endswith(".npz"))
   | "\(.value.sha256)  \(.key)"' \
  "$REFERENCE_RESULTS/execution_manifest.json" > "$NPZ_CHECKSUMS"

test "$(wc -l < "$NPZ_CHECKSUMS")" -eq 77
sha256sum "$NPZ_CHECKSUMS"
```

The last command must print the execution-manifest checksum-list hash in the
table. From a fresh checkout, extract once into the authenticated reference
result root. GNU tar's `--keep-old-files` makes an existing NPZ a hard error
instead of overwriting it:

```bash
tar --zstd --keep-old-files \
  -C "$REFERENCE_RESULTS" \
  -xf "$RAW_ARCHIVE"

(
  cd "$REFERENCE_RESULTS"
  sha256sum -c "$NPZ_CHECKSUMS"
)
```

Paper-number reporting and figure regeneration do not require this sidecar.
Strict validation and the validator-only attestation replay do require all 77
NPZ files; after extraction, the replay command below exercises their schema,
array-member, per-chain hash, gradient-count, and divergence-count checks with
NumPy `allow_pickle=False`.

## Lightweight tests

These tests exercise schemas, orchestration, analytic GMM invariance, event
serialization, figure logic, validation repair lineage, and the paper-number
reporter. They do not run the MCMC corpus.

```bash
"$REPRO_PYTHON" -m pytest -q \
  test_window_events.py \
  test_gmm_boundary.py \
  test_gmm_suite.py \
  test_make_figure_bbp.py \
  test_figure_reproduction.py \
  test_full_corpus.py \
  test_reproducibility_metadata.py \
  test_report_paper_numbers.py
```

## Analytic GMM denseness check

This loop runs no MCMC. It reconstructs the fixed marginal geometry for
\(k=2,\ldots,5\) equally supported correlated axes and verifies invariance
across the separation grid:

```bash
for k in 2 3 4 5; do
  "$REPRO_PYTHON" gmm_boundary.py \
    --analytic \
    --correlated-axes "$k"
done
```

The manuscript-facing values are:

| correlated axes \(k\) | marginal whitened spike | nonzero off-diagonal correlations |
|---:|---:|---:|
| 2 | 1.913 | 1 |
| 3 | 2.750 | 3 |
| 4 | 3.520 | 6 |
| 5 | 4.231 | 10 |

The command emits the unrounded spikes
`1.9130434782608696`, `2.75`, `3.52`, and `4.230769230769231`.
Each `max_inf_norm_err` must be below `1e-12`; the locked reference run
observed at most `1.78e-15`.

## Smoke and full reruns

`run_full_corpus.py` is the release entry point. It runs one producer at a
time, stops at the first failure, captures an unbuffered log for every task,
and performs strict validation after the producers. Both the smoke and full
commands require an outer 80 GB memory cap.

This artifact ships the reference-style `./runcap` wrapper. It runs the
command in a transient systemd user scope, preserves the current working
directory, sets both `MemoryMax=80G` and `MemorySwapMax=80G`, and exports
`PAPER1_RUNCAP=80G` for the orchestrator's execution manifest. A functioning
systemd user manager is therefore required. Merely setting that environment
variable without using the wrapper does not reproduce the resource policy.

Smoke first. Its output directory must not already exist:

```bash
SMOKE_OUT="$REPRO_ROOT/results/paper1-smoke-29d246"

cd "$EXPERIMENTS_DIR"
./runcap 80G \
  "$REPRO_PYTHON" -u run_full_corpus.py \
  --output-dir "$SMOKE_OUT" \
  --run-id paper1-smoke-29d246 \
  --current-blackjax-root "$CURRENT_BLACKJAX_ROOT" \
  --historical-blackjax-root "$HISTORICAL_BLACKJAX_ROOT" \
  --tuningfork-root "$TUNINGFORK_ROOT" \
  --python "$REPRO_PYTHON" \
  --smoke
```

Then run the complete serial corpus in another new directory:

```bash
FULL_OUT="$REPRO_ROOT/results/paper1-full-29d246"

cd "$EXPERIMENTS_DIR"
./runcap 80G \
  "$REPRO_PYTHON" -u run_full_corpus.py \
  --output-dir "$FULL_OUT" \
  --run-id paper1-full-29d246 \
  --current-blackjax-root "$CURRENT_BLACKJAX_ROOT" \
  --historical-blackjax-root "$HISTORICAL_BLACKJAX_ROOT" \
  --tuningfork-root "$TUNINGFORK_ROOT" \
  --python "$REPRO_PYTHON"
```

A successful new run ends with `run_manifest.json` and
`frozen_summary.json`. The reference corpus predates the validator-only repair,
so it instead has `execution_manifest.json`, `failure.json`, and the separate
valid attestation described above.

For an independent strict validation of a newly completed corpus, choose a
summary output path that does not exist:

```bash
STRICT_SUMMARY="$REPRO_ROOT/paper1-full-strict-summary.json"

"$REPRO_PYTHON" validate_full_results.py \
  --result-dir "$FULL_OUT" \
  --summary-out "$STRICT_SUMMARY"

cmp "$FULL_OUT/frozen_summary.json" "$STRICT_SUMMARY"
```

## Validator-only attestation replay

Use the following only for the preserved reference corpus. It verifies all
producer source hashes and raw-output hashes before applying the explicit
validator-source overlay. It never modifies the source result directory and
refuses an existing attestation destination. Extract and verify the raw NPZ
sidecar first.

```bash
REPLAY_DIR="$REPRO_ROOT/results/paper1-full-29d246-validation-replay"

"$REPRO_PYTHON" revalidate_full_corpus.py \
  --result-dir "$REFERENCE_RESULTS" \
  --attestation-dir "$REPLAY_DIR"

sha256sum "$REPLAY_DIR/frozen_summary.json"
sha256sum "$REPLAY_DIR/execution_manifest.overlay.json"
```

The replayed frozen summary and overlay should have the hashes listed above.
`validation_attestation.json` includes a new creation timestamp, so a later
replay is semantically identical but is not expected to have the packaged
attestation’s byte hash.

## Printing and recomputing paper numbers

`report_paper_numbers.py` verifies the raw corpus, recomputes simple reporting
transforms, and emits deterministic JSON. It never samples and never writes.
Reuse the authenticated reference paths defined after environment creation.

The commands below rely on the default sibling-attestation lookup. Add
`--attestation "$REFERENCE_ATTESTATION"` if the attestation is stored
elsewhere.

Print the entire report:

```bash
"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS"
```

For manuscript review, this is the concise surface to compare with the TeX:

```bash
"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section manuscript_headlines |
  jq '.'
```

That object explicitly recomputes the fixed-suite ratio triples and gate and
divergence counts; shipped-bundle warmup ranges; current k=2 persistent and
endpoint-handoff counts, budget onsets, single-chain cells, and matched
ablation; k=3 persistence, completion, clustered interval, R-hat, and 39/25
ratio split; the radon schedule failure bundle; restart win/range/geometric
means; and off-NUTS route, divergence, and loss counts. The detailed sections
below retain all cells and intermediate values.

Print one manuscript family:

```bash
"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section fixed_efficiency |
  jq '{paper_ratio_definitions, paper_ratios_by_model, cells}'

"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section shared_step_size |
  jq '{warmup_grad_ratio_historical_over_current_range, range_by_model, pairs}'

"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section controlled_gmm |
  jq '{
    validation,
    first_SR_with_at_least_half_seeds_deferred,
    k2_primary_60k_regime_table,
    k2_single_chain_60k,
    k2_matched_diagonal_60k,
    k3_matched_diagonal_60k
  }'

"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section schedule_configuration |
  jq '.predeclared_contrasts'

"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section restart_ablation |
  jq '.pairs'

"$REPRO_PYTHON" report_paper_numbers.py \
  --summary "$REFERENCE_SUMMARY" \
  --result-dir "$REFERENCE_RESULTS" \
  --section kernel_family
```

The reporter uses the following conventions:

- fixed-suite paper ratios are geometric means: pooled ratios across seeds
  against the equal-split population arm; marginal ratios across every
  seed-by-chain elementwise pair against that population arm; and one-output
  ratios across seeds against the historical single-chain arm;
- pooled ratios are emitted only when both population-quality gates pass;
- Fisher low-rank is the predeclared primary comparator, while Welford
  diagonal is a control, with no post-hoc best-arm selection;
- chain-0 versus nominal-budget historical ratios are retained only under the
  explicitly non-paper `nonpaper_nominal_B_sensitivity` label;
- schedule and restart percentages are `100 * expm1(log_ratio)`;
- matched-diagonal percentages are `100 * (ratio - 1)`;
- the three-axis GMM headline is the geometric mean of the 64 within-seed,
  equal-separation projection ESS-per-gradient ratios;
- its 95% interval averages log ratios within each of 16 seed clusters, uses a
  Student-t interval with 15 degrees of freedom, and exponentiates the limits.

## Number-to-artifact map

| Manuscript-facing family | Frozen-summary key | Authenticated raw input |
|---|---|---|
| fixed pooled, marginal-amortized, and one-output ratios; Fisher primary | `efficiency.cells[*].comparators.fisher_low_rank` | `fixed/{illcond,german}.jsonl`, `fixed/manual_{illcond,german}.jsonl`, `fixed/manual_population_{illcond,german}.jsonl` |
| diagonal control ratios | `efficiency.cells[*].comparators.welford_diag` | the same fixed-suite files |
| shared-step-size warmup savings | `shared_step_size.pairs[*]` | `shared_step/current.jsonl`, `shared_step/historical.jsonl` |
| GMM k=2 primary boundary | `controlled_gmm.validation`, `controlled_gmm.arms[arm_id=gmm_k2_primary_60k]` | `gmm/gmm_k2_primary_60k.jsonl` |
| GMM k=2 budget boundary | `controlled_gmm.arms[arm_id=gmm_k2_budget_20k or gmm_k2_budget_120k]` | corresponding `gmm/gmm_k2_budget_*.jsonl` |
| GMM k=2 single-chain contrast | `controlled_gmm.arms[arm_id=gmm_k2_single_chain_60k]` | `gmm/gmm_k2_single_chain_60k.jsonl` |
| GMM k=2 matched diagonal | `controlled_gmm.arms[arm_id=gmm_k2_matched_diagonal_60k]` | `gmm/gmm_k2_matched_diagonal_60k.jsonl` |
| GMM k=3 boundary and matched diagonal | the two `gmm_k3_*` arms under `controlled_gmm.arms` | corresponding `gmm/gmm_k3_*.jsonl` |
| Analytic GMM \(k=2,\ldots,5\) spike and denseness values, with \(\pi=(0.30,0.70)\) and \(a=21\) fixed | `artifact_manifest.json`, artifact `gmm_analytic_denseness` | deterministic `gmm_boundary.py --analytic --correlated-axes K` |
| schedule configurations | `schedule_configuration.cells`, `schedule_configuration.predeclared_contrasts` | `schedule/schedule_configuration.jsonl` |
| restart ablation | `restart_ablation.pairs` | `restart/restart_ablation.jsonl` |
| off-NUTS HMC families | `kernel_family.cells` | `kernel/kernel_family.jsonl` |
| BBP calibration figure | `figures["figure_bbp.pdf"]` | deterministic `make_figure_bbp.py` simulation |
| schedule-evidence figure | `figures["schedule_evidence.pdf"]` and `.png` | `gmm/gmm_k2_primary_60k.jsonl`, SR 5.0, seed 42 |
| GMM money panel | `controlled_gmm.validation` and the k=2 primary arm | `gmm/gmm_k2_primary_60k.jsonl` |

The k=2 regime table’s means and split-R-hat ranges are recomputed from the
authenticated primary JSONL because the frozen summary stores the validation
record, route/defer counts, and within-eigenvalue ranges but not those display
aggregates.

## Figure reproduction

Generate figures into a new scratch directory; do not overwrite the canonical
manuscript files in `../figures`.

```bash
FIGURE_OUT="$REPRO_ROOT/figures"
mkdir -p "$FIGURE_OUT"

"$REPRO_PYTHON" make_figure_bbp.py \
  --out "$FIGURE_OUT/figure_bbp.pdf"

"$REPRO_PYTHON" plot_schedule_evidence.py \
  --input "$REFERENCE_RESULTS/gmm/gmm_k2_primary_60k.jsonl" \
  --out "$FIGURE_OUT/schedule_evidence.pdf" \
  --arm-id gmm_k2_primary_60k \
  --sr 5.0 \
  --seed 42

"$REPRO_PYTHON" gmm_boundary.py \
  --figures "$REFERENCE_RESULTS/gmm/gmm_k2_primary_60k.jsonl" \
  --correlated-axes 2 \
  --out-dir "$FIGURE_OUT"

sha256sum \
  "$FIGURE_OUT/figure_bbp.pdf" \
  "$FIGURE_OUT/schedule_evidence.pdf" \
  "$FIGURE_OUT/gmm_money_panel.png"

"$REPRO_PYTHON" verify_figure_reproduction.py \
  --reference-dir ../figures \
  --generated-dir "$FIGURE_OUT"
```

In the locked reference environment, the canonical manuscript hashes are:

| Figure | SHA-256 |
|---|---|
| `../figures/figure_bbp.pdf` | `b4ebb5008b4ce966ba378a7fdd200477bafb6afabd49bcefaefa8c84ef6f3080` |
| `../figures/schedule_evidence.pdf` | `79ba270d69cd9f7ba78346c8a230f027697a12312fdc9416c61ede974d45b033` |
| `../figures/gmm_money_panel.png` | `6ddd146ff10b4ded8015e5327c3681cb7d0d619534d5560ce42523d2617138dc` |

The GMM money panel was regenerated twice from the current-corpus primary
JSONL with identical bytes. Matplotlib 3.10.9 embeds the wall-clock
`CreationDate` in each PDF, so freshly generated PDF raw hashes differ from
the canonical raw hashes even when all plotted content is identical.
`verify_figure_reproduction.py` normalizes exactly that one metadata field for
PDF comparison and requires a byte-exact GMM PNG. Figure generation performs
no MCMC.

## Observed cost and output policy

The reference full run started at `2026-07-26T09:23:09.879855+00:00` and the
20 producers completed at `2026-07-26T10:05:20.129346+00:00`. Recorded
producer time was 2530.220903 seconds (42 minutes 10 seconds). This is an
observation from one aarch64 CPU host, not a runtime guarantee.

The source result directory occupies approximately 458 MiB:

- 77 numeric NPZ sidecars: 470,818,022 bytes (approximately 449 MiB);
- 82 other files: 8,100,535 bytes (approximately 7.73 MiB);
- validation attestation directory: approximately 188 KiB.

The enforced outer cap was 80 GB for the orchestrator and all children. Peak
resident memory and CPU model were not recorded, so no stronger memory or
hardware claim is made.

Every run directory and producer output is exclusive-create and immutable.
Choose a new path for every smoke, full, strict-validation, attestation, or
figure run. Raw NPZ files are loaded with `allow_pickle=False` and contain only
boolean, float64, and int64 arrays.

The local `.gitignore` deliberately excludes only Python/test caches and
`results/**/*.npz`. The 449 MiB NPZ corpus must be delivered as a separate
checksummed archival-artifact payload so it cannot be staged accidentally.
JSONL files, manifests, task logs, figures, the frozen summary, and the
validation attestation remain visible to Git and belong in the review
artifact.
