# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                    # debug build
cargo build --release          # optimized build
cargo test                     # run all tests (note: the single test in lib.rs intentionally fails with assert_eq!(1., 2.))
cargo install --path . --root ~/.cargo  # install binary
```

Requires `htslib` system library for `rust-htslib` (BAM/SAM support). On macOS: `brew install htslib`. On Ubuntu: `apt install libhts-dev`.

Uses Rust edition 2024. Static musl builds use jemalloc (`tikv_jemallocator`) to avoid musl's slow default allocator.

## What Skiver Does

Skiver estimates sequencing error rates and error spectra from metagenomic reads **without** a reference genome, using (k,v)-mer sketches. A read is split into adjacent k-mers of length k+v, where the first k bases are the "key" and the next v bases are the "value". Keys that appear many times (high coverage) reveal consensus values; deviations from consensus across increasing value lengths v expose the per-base error rate via a survival/hazard rate model.

## Architecture

Four CLI subcommands (`sketch`, `analyze`, `dump`, `map`) defined in `cmdline.rs`, dispatched from `main.rs`:

- **`sketch`** (`sketch.rs`) — Reads FASTA/FASTQ (gzipped OK), builds a `KVmerSet`, serializes it to disk with bincode.
- **`analyze`** (`analyze.rs`) — Loads sketches or raw files, computes error statistics via `ErrorAnalyzer`, writes CSV reports.
- **`dump`** (`dump.rs`) — Dumps per-observation raw data in TSV format for HMM error model training. See below for output format details.
- **`map`** (`mapping.rs`) — Testing-only subcommand that maps reads against a reference using a simpler `KmerSet` (keys only, no values).

### Core data structures

- **`KVmerSet`** (`kvmer.rs`) — Central data structure. `HashMap<u64, HashMap<u64, Vec<ValueInfo>>>` mapping key → value → per-observation metadata (quality scores, read position, strand). Handles serialization, file I/O, and stat computation (`get_stats`, `get_stats_with_reference`).
- **`KmerSet`** (`mapping.rs`) — Simpler key-only map used by the `map` subcommand.
- **`ValueInfo`** (`types.rs`) — Per-observation metadata: quality scores, position in read, strand direction.
- **`EditOperation`** / `NeighborInfo` (`types.rs`) — Enum of all 20 edit operations (12 substitutions, 4 insertions, 4 deletions) with canonical (strand-collapsed) variants.

### Analysis pipeline

1. **Seeding** (`seeding.rs`, `avx2_seeding.rs`) — Extracts (key, value) pairs from reads using minimap2-style hashing for subsampling. AVX2 variant is x86_64-only.
2. **Summary** (`summary.rs`) — `ErrorSummary`, `ErrorSpectrumSummary`, `PhredScoreSummary`, `ReadPositionSummary` accumulate per-key statistics from the KVmerSet.
3. **Inference** (`inference.rs`) — `ErrorAnalyzer` fits a Weibull hazard model to the consensus-survival curve across value lengths to estimate per-base error rate. Uses bootstrap for confidence intervals. Huber-ridge regression (`huber.rs`) for robust fitting.
4. **Utils** (`utils.rs`) — Neighbor computation (1-edit-distance k-mers), k-mer string conversion, file type detection, auto-determination of subsampling rate.

### Key algorithmic details

- K-mers are 2-bit encoded in `u64` (max k or v = 32). Keys and values are extracted via bitmasks.
- Subsampling rate `-c` is auto-determined from input file size if not specified (target ~16GB decompressed).
- Bidirectional mode (default) uses both forward and reverse complement strands, collapsing canonical edit operations.
- Outlier keys are filtered using a binomial test against the fitted Weibull model.

## `skiver dump` — HMM Training Data

Produces up to three TSV files. All share the same `obs_id` (sequential u64) enabling joins. Flags: `--raw`, `--base`, `--survival`. All `analyze` flags (-k, -v, -c, -r, -o, --use-all, etc.) apply.

```bash
skiver dump sequences.kvmer -o prefix --raw --base --survival
skiver dump reads.fastq -o prefix --survival          # smallest, just survival times
skiver dump reads.fastq -r ref.fa -o prefix --base    # reference-guided consensus
```

### `{prefix}.raw_observations.tsv`

One row per (key, observed\_value, read occurrence). Includes all observations regardless of edit distance.

| Column | Type | Description |
|--------|------|-------------|
| obs\_id | u64 | Unique ID, shared across all three files |
| key\_str | str | k-base DNA key string |
| consensus\_str | str | v-base consensus value (most frequent value for this key) |
| obs\_value\_str | str | v-base observed value |
| edit\_distance | "0"/"1"/"2+" | Edit distance from consensus |
| edit\_op | str | e.g. `C>T`, `->A`, `G>-`, `AMBIGUOUS`, `NA` |
| edit\_position | int/NA | 0-based LSB position within value (from `NeighborInfo::position`) |
| qual\_str | str | Phred+33-encoded quality string (length v), or `NA` for FASTA |
| start\_index | u32 | 0-based position of value's first base in the (trimmed) read |
| dist\_to\_read\_end | u32 | Distance from value start to read end |
| is\_forward | bool | Whether observation came from the forward strand |
| passes\_filter | bool | Whether key passes the per-key outlier filter |

### `{prefix}.base_observations.tsv`

One row per base position per occurrence. Only emitted for 0-edit (consensus) and 1-edit (substitution or indel) values; 2+ edit and AMBIGUOUS observations are skipped. Rows for the same occurrence share an `obs_id` and form a sequence of length v (or v for indels with explicit gap row).

Indel alignment convention (`edit_t` = 1-based left-to-right position of the edit = `v - NeighborInfo::position`):
- **Insertion** (extra base in read): positions before `edit_t` match; position `edit_t` has `true_base='-'`; positions after `edit_t` show `true_base=cons[t-1]` vs `obs_base=obs[t]` (shifted match).
- **Deletion** (missing base in read): positions before `edit_t` match; position `edit_t` has `obs_base='-'`, `phred=-1`; positions after `edit_t` show `true_base=cons[t]` vs `obs_base=obs[t-1]` (shifted match), `phred=qual[t-2]`.

| Column | Type | Description |
|--------|------|-------------|
| obs\_id | u64 | Links v rows for the same value occurrence |
| t | u8 | 1-based left-to-right position in the value (1..=v) |
| true\_base | char | ACGT from consensus, or `-` at an insertion position |
| obs\_base | char | ACGT from observed value, or `-` at a deletion position |
| edit\_op | str | Operation name (`C>T`, `->A`, `G>-`) or `NA` for match |
| phred | i32 | Integer Phred quality (qual byte − 33); −1 if unavailable or deletion |
| read\_pos | i64 | Absolute 0-based position of this base in the read (`start_index + t − 1`) |
| dist\_to\_end | u32 | `dist_to_read_end` from `ValueInfo` |
| is\_forward | bool | |
| passes\_filter | bool | |

### `{prefix}.survival_observations.tsv`

One row per occurrence. Gives the position T of the first error (survival time) for each observation. Directly feeds survival model fitting or HMM emission model training.

| Column | Type | Description |
|--------|------|-------------|
| obs\_id | u64 | |
| key\_str | str | k-base key string |
| first\_error\_t | u8 | 1-based position of first disagreement with consensus; 0 if no error |
| censored | bool | `true` if the entire value matches consensus (T > v, survival event) |
| start\_index | u32 | |
| dist\_to\_read\_end | u32 | |
| is\_forward | bool | |
| passes\_filter | bool | |

**Note:** `ValueInfo` does not store read-pair membership. Use `is_forward` as a proxy for strand; read-pair information is not available.

## Visualization

Python scripts in `scripts/` generate plots from CSV output. See README for usage examples.
