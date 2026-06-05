# Performance & Memory: Design Choices and Profiling Evidence

This document records the data-structure / concurrency design choices in skiver's
hot paths, the profiling methodology behind them, and the **measured** evidence for
what helped and what did not. It is the durable engineering reference; the raw
artifacts and per-step tables live alongside the example that was used as the
workload:

- `examples/shotgun_synthetic_recovery_v2/MEMORY_PROFILE.md` — memory profiling run.
- `examples/shotgun_synthetic_recovery_v2/PERFORMANCE_PROFILE.md` — CPU profiling run.
- `examples/shotgun_synthetic_recovery_v2/profile_work/perf/` — `sample`/cProfile output.

**Guiding principle:** optimisations were profiling-driven and **correctness-preserving**.
Every change is guarded by a golden CLI test suite (`tests/cli_golden.rs`) plus unit
tests (`tests/core.rs`) that assert **byte-identical** `analyze`/`dump` output, so the
work below changes resource usage, not results. (The one deliberate behaviour change is
that consensus tie-breaking is now *deterministic* — see below — which the old code left
to HashMap iteration order.)

## 1. Profiling methodology

So the numbers are reproducible and not guesswork:

- **Peak memory:** `/usr/bin/time -l` "maximum resident set size" per command; **dhat**
  heap profiler for the call-stack of bytes-at-peak (build with `--features dhat-heap`,
  which swaps in the dhat global allocator and writes `dhat-heap.json`; view at
  <https://nnethercote.github.io/dh_view/dh_view.html>, sort by *At t-gmax*).
- **CPU / wall:** `/usr/bin/time -l` real/user/sys (single-thread shows `user ≈ real`);
  a **free phase breakdown** from skiver's existing timestamped `info!` logs
  (`Pass 1 … / Finished pass 1 / Pass 2 … / Building stats from N pairs …`); and
  function-level **sampling** via macOS `/usr/bin/sample` (or `samply`).
- **Python:** `cProfile` (cumulative + self time) and `py-spy`; `torch.get_num_threads()`.

Workload: the `run_marginal_rate_diagnostics.sh` pipeline, exercised on real reads
(`ERR10889718`, 20K–200K pairs) and the synthetic genome-blender reads. Measurements
were taken on arm64 macOS, 8 perf cores, on **uncompressed** FASTQ unless noted — see
*Caveats* (§6).

## 2. Memory design choices (and evidence)

The two-pass `dump` / reference-guided `analyze` paths use a **flat count map**,
`KVmerCountMap.counts: HashMap<(u64,u64), [u32;2]>` (`src/kvmer.rs`), instead of the
nested `KVmerSet` (`HashMap<u64, HashMap<u64, Vec<ValueInfo>>>`) that stores a
per-observation `ValueInfo` (with a heap `qual: Vec<u8>`). Storing only counts removed
the per-observation `ValueInfo`/quality storage from the analysis path. The remaining
peaks, found by dhat, drove these choices:

- **Allocation-free per-key grouping.** Building per-key statistics previously rebuilt a
  nested `HashMap<u64, HashMap<u64,[u32;2]>>` — one inner map per key. dhat showed this as
  **3.67 M live allocations** (~279 MB of empty-ish hashbrown tables) at scale. It is
  replaced by `for_each_consensus_group` (`src/kvmer.rs`): sort a single `Vec` by key and
  reuse **one** scratch map across keys. The `ErrorSummary` updaters keep their exact
  signatures.
  *Evidence:* `analyze -c1` peak RSS **1219 → 510 MiB (−58%)**; dhat at t-gmax
  **1206 MB / 3.67 M blocks → 317 MB / 42 blocks**.
- **Deterministic consensus tie-break.** With the old per-key HashMap, the most-frequent
  value (consensus) on a count tie was chosen by arbitrary iteration order, so output was
  non-reproducible. Consensus is now "highest count, ties broken by smallest value"
  (`for_each_consensus_group`, `build_consensus_only`). Output is byte-identical
  run-to-run; the only diffs vs the old binary are on count-tied keys (previously
  arbitrary).
- **`(u64,u64)` tuple key, not a packed `u64`.** Packing key+value into one `u64` needs
  `k + v ≤ 32`. The pipeline uses **k=21, v=13 (k+v=34 ⇒ 68 bits)**, which does not fit;
  a `u128` is the same 16 bytes as the tuple. So the tuple is already minimal for this
  regime — packing was implemented, hit the assertion, and reverted. *(Non-gain — §5.)*
- **Pre-size only where the bound is safe.** `KVmerCountMap::with_capacity` is used in
  **reference-guided mode**, where the reference key count is a true upper bound, to avoid
  hashbrown's resize-doubling transient (old+new tables coexisting; dhat showed a 717 MB
  spike). A non-reference input-size *estimate* was implemented and then **removed**: the
  measured distinct kv-mers (4.47 M) diverged from the estimate (3.25 M), giving no
  benefit and a small `dump` regression. *(Non-gain — §5.)*
- **Python: load the reference once.** `_load_fasta` in
  `scripts/benchmark_simulated_context_model.py` is memoised (`functools.lru_cache`); the
  hundreds-of-MB combined reference was being parsed 4–6× per run. (Aggregation and the h5
  cache already *stream* their TSVs, so they were not a memory problem.)

Full-scale implication: the `analyze` peak extrapolates from ~30 GB toward ~13–15 GB,
bringing the heaviest pipeline step within reach of a 16 GB laptop.

## 3. CPU design choices (and evidence)

Profiling first established that skiver is **single-threaded** (`user ≈ real`) and then
showed *where* the time goes. The biggest wins turned out to be single-threaded, not
threading.

**Function-level hotspots (`/usr/bin/sample`, leaf samples), `dump` & `analyze` agree:**

| Hotspot | share | meaning |
|---|---|---|
| `BuildHasher::hash_one` + `DefaultHasher::write` | **~32–34%** | std **SipHash** on the `u64`/`(u64,u64)` map keys |
| per-key aggregation + per-read counting loops | ~26% | themselves hash-bound |
| malloc/free/realloc/memmove/`reserve_rehash` | **~15%** | per-read `Vec`/qual allocations + map resizes |
| `fmh_seeds_masked` (the actual seeding arithmetic) | **~2%** | tiny |
| inference (`calculate_ratio` + fit), incl. bootstrap | **<1%** | negligible |

Phase breakdown (`dump`, from logs): Pass 1 seed+count 28% · build 14% · Pass 2
(re-read + **re-seed** + classify + write) 59% — i.e. dump seeds the input twice.

The design choices that followed:

- **Fast, deterministic hasher (FxHash).** ~⅓ of CPU was SipHash. The hot `u64`/
  `(u64,u64)`/neighbour/reference maps now use `rustc-hash`'s `FxHashMap` (`src/kvmer.rs`,
  `src/summary.rs`, `src/utils.rs`, `src/dump.rs`, `src/analyze.rs`). **FxHash, not
  `ahash`** — FxHash is deterministic, so the byte-identical-output guarantee holds.
  The serialised `KVmerSet.key_value_qual_map` and the `map` subcommand keep the default
  hasher (not hot; sketch-format stability).
  *Evidence:* `dump` ~1.4×, `analyze` ~1.9× alone, output identical.
- **Counting doesn't extract quality, and reuses buffers.** Counting needs only
  key/value/strand, so `add_file_impl` uses a no-quality seeder
  (`extract_kv_no_qual_into`) and reuses three seeding buffers across reads instead of
  allocating per read (`count_seeds_into` is shared). This attacks the ~15% allocation
  churn.
  *Evidence:* `analyze` ~1.4×, `dump` ~1.2× on top of the hasher change.
  **Combined:** `dump` 6.37 → 3.78 s (**1.7×**), `analyze` 13.85 → 5.15 s (**2.7×**),
  output byte-identical.

**Python multi-processing (where work is genuinely independent):**

- **Concurrent benchmark subprocesses.** The two `skiver dump`s and the `analyze` run via
  a `ThreadPoolExecutor` (each spends its time in a child process, so the GIL is released);
  the train/test BAM truth-map builds likewise. cProfile had shown 2.17 s of sequential
  `posix.waitpid`. (`scripts/benchmark_simulated_context_model.py`.)
- **Parallel per-file TSV aggregation.** `aggregate_context_length_screen_counts`
  (`scripts/lib/context_error_models.py`) aggregates each file in its own process
  (`ProcessPoolExecutor`) and **sums** the count tensors — addition is associative, so the
  result is identical to serial. Auto-enabled when there is >1 file, serial fallback
  otherwise (and serial keeps the progress callback). Each worker reads its *own* file, so
  there is no shared-reader bottleneck.
  *Evidence:* parallel vs serial produce **identical** count tensors (verified on a
  multi-prefix run). Benefit is at full scale (the real 18-file aggregation); the small
  fixture is single-file and stays serial.
- **torch is already parallel.** `torch.get_num_threads() == 8`; training shows
  `user > real`. The SVI/MLE+VI steps already use all cores, so models are *not* run in
  parallel processes (that would oversubscribe).

## 4. Correctness guardrails

- `tests/cli_golden.rs`: runs the built binary on a committed fixture and compares
  `analyze`/`dump` output to committed golden files (byte-for-byte). `summary_error_rate.csv`
  is intentionally excluded — its bootstrap CI columns use an unseeded RNG and are not
  reproducible (a pre-existing property, noted as a latent improvement).
- `tests/core.rs`: unit tests for neighbour enumeration, seeding, count-map consensus /
  tie-break, and the numeric fits.
- Python: the existing `tests/test_benchmark_simulated_context_model.py` plus a
  memoisation test; P3's parallel/serial equivalence was verified directly.

## 5. Non-gains — measured and rejected

Recording these so they are not re-attempted blindly:

- **Threading the per-read seed+count loop (crossbeam producer/consumer).** Implemented
  with per-thread maps merged at the end; `-t8` output was identical to `-t1` (counts are
  associative). But it **regressed**: `analyze` `-t1` 5.45 s → `-t8` **8.32 s** (user CPU
  4.66 → 9.29 s). Once the hasher fix made seed+count cheap, a *single* reader thread
  (decompress + per-read `seq` copy + batching) is the bottleneck and the workers starve;
  channel/copy/thread overhead dominates, and gzipped input would starve the reader
  further. **Reverted.** Real per-read CPU parallelism here would require parallel
  decompression / chunked FASTQ splitting — a much larger change. (Contrast P3, which
  works because each process reads an *independent* file.)
- **Parallelising the bootstrap CI loop** (`src/inference.rs`, default 100 experiments).
  The code-only guess flagged it as embarrassingly parallel, but it is **negligible**:
  `--num-experiments` 1 → 1000 changes `analyze` user time only 2.66 → 2.89 s. Not worth
  it.
- **Non-reference count-map pre-sizing from an input-size estimate.** The estimate
  (3.25 M) diverged from the actual distinct kv-mers (4.47 M); no benefit and a ~+47 MiB
  `dump` regression. Removed; pre-sizing kept only for the safe reference-key bound.
- **Packing the `(key,value)` map key into one `u64`.** Inapplicable at k=21/v=13
  (k+v=34 > 32); `u128` saves nothing over the tuple.

## 6. Caveats

- Hotspot percentages and speedup ratios were measured on **arm64 macOS, 8 cores,
  200K-read uncompressed FASTQ**. The real pipeline reads `.gz`, so gzip decode is a larger
  I/O share — which *lowers* the ceiling for any seed+count threading further, but does not
  change the single-threaded rankings.
- Memory full-scale figures (~30 GB → ~13–15 GB) are **linear extrapolations** from the
  small workload, not an end-to-end 16 GB-laptop run.
- The deterministic hasher (FxHash) and tie-break make output reproducible; if a future
  change needs a randomised hasher, the golden tests will (correctly) flag the output
  change.

## 7. Summary table

| Change | Kind | Status | Evidence |
|---|---|---|---|
| Flat count map + sort-based per-key grouping | memory | shipped | analyze 1219→510 MiB (−58%); 3.67 M→42 allocs |
| Deterministic consensus tie-break | correctness | shipped | reproducible output |
| Pre-size count map (reference mode only) | memory | shipped | removes resize-doubling spike |
| Load reference once (lru_cache) | memory (Py) | shipped | reference parsed 1× not 4–6× |
| FxHash fast hasher | CPU | shipped | dump 1.4×, analyze 1.9× |
| No-qual counting + reused buffers | CPU | shipped | +1.2–1.4× (combined dump 1.7×, analyze 2.7×) |
| Concurrent benchmark subprocesses / BAM splits | CPU (Py) | shipped | removes 2.17 s serial waits |
| Parallel per-file TSV aggregation | CPU (Py) | shipped | parallel == serial; scales with #files |
| Threaded seed+count | CPU | **reverted** | analyze -t8 8.3 s vs -t1 5.4 s (reader-bound) |
| Bootstrap parallelisation | CPU | **rejected** | 1→1000 experiments = +0.2 s |
| Non-ref pre-size estimate | memory | **reverted** | estimate 3.25 M vs actual 4.47 M; +47 MiB |
| Pack key into u64 | memory | **N/A** | k+v=34 > 32 |
