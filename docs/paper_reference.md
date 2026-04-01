# Skiver: Mathematical and Design Reference

Reference for Gu, Sharma, Wong & Nagarajan (2026), "Skiver: Alignment-free Estimation
of Sequencing Error Rates and Spectra using (k, v)-mer Sketches", bioRxiv 2026.02.12.705514v1.

This document describes the mathematical framework behind the code and the intention
of each component and design choice.

---

## 1. Core Concept: Survival Analysis of Sequencing Errors

Skiver reframes error-rate estimation as a **discrete survival analysis** problem. Rather
than asking "what fraction of bases are wrong?", it asks "how far can we read from a
random starting position before hitting the first error?"

Let **T** (T >= 1) be the random variable representing the number of bases from a random
starting position until the first failure (disagreement between the sequenced base and
the true underlying base).

### 1.1 Hazard Rate

**Definition.** The *hazard rate* at the t-th base from the starting position is:

    h(t) := Pr[T = t | T >= t]

This is the conditional probability that the t-th base disagrees with truth, given that
the previous (t - 1) bases all agree.

### 1.2 Survival Rate

**Definition.** The *survival rate* at the t-th base is:

    S(t) := Pr[T > t]

i.e. the probability that the first t bases are all error-free. By the chain rule of
conditional probability:

    S(k) = prod_{t=1}^{k} (1 - h(t))

The connection to error rate: S(k) also gives the probability of a k-mer in the read
being free of sequencing error. The first-base hazard gives the per-base error rate:

    h(1) = 1 - S(1)

### 1.3 Why Hazard Rate Instead of Simple Error Rate

The paper argues that hazard rate is more informative than a single error-rate number:

1. **Different alignment tools give different error counts.** Alignment scoring schemes
   introduce ambiguity (e.g. Minimap2 favours mismatches over indels), producing
   tool-dependent error estimates. Skiver is alignment-free.

2. **Hazard rate captures positional error structure.** Two reads can have the same
   overall error rate (e.g. 1/4) but very different hazard rate profiles. A read with
   clustered errors (low hazard at small t, high later) behaves differently from one
   with uniformly distributed errors. The hazard rate curve distinguishes these cases.

---

## 2. (k, v)-mer Sketches

### 2.1 Definition

A **(k, v)-mer** is a segment of DNA of length k + v, where:
- The first **k** bases are the **key**
- The last **v** bases are the **value**

Key and value must be **adjacent** in the read (unlike SPLASH anchor-target pairs which
can be separated).

### 2.2 Sketch Construction (Step by Step)

The sketch construction (implemented in `sketch.rs`, `seeding.rs`) proceeds as:

**Step 1.** Extract all (k, v)-mers from reads and their reverse complements. Optionally,
use only the forward strand (`--forward-only`).

**Step 2.** Subsample roughly 1/c of the (k, v)-mers using **FracMinHash**. Given a random
hash function H : Sigma^k -> (0, 1), a (k, v)-mer is kept if the hash of its key is
less than 1/c. Default: c = 1000.

**Step 3.** Store subsampled (k, v)-mers in a hash table mapping each key to its
associated values and per-observation metadata (`ValueInfo`: quality scores, read
position, strand).

**Step 4.** For each key, identify the **consensus** value: the most frequently appearing
value. This is the best estimate of the true genomic sequence following that key.

### 2.3 Why k is Large (default k = 21)

k is chosen large enough (default k = 21) that keys are mostly **unique** identifiers in
the sequenced genomes. This means each key corresponds to a specific genomic position,
so variation in the associated values reflects sequencing errors (not biological
variation at different loci).

### 2.4 Why Separate Key and Value (default v = 13)

The key anchors a genomic position; the value is the region where errors are detected.
By examining how agreement with the consensus value breaks down across increasing
prefix lengths of v, skiver reconstructs the hazard rate curve h(t) for
t in {k+1, ..., k+v}.

### 2.5 Subsampling Rate Auto-determination

If `-c` is not specified, skiver automatically determines c as:

    c = ceil(total_decompressed_input_size / 16 GB) * 1000

For gzipped files, the decompressed size is estimated as 4x the compressed size.
This targets roughly 16 GB of effective data regardless of input size. Implemented in
`utils.rs` (`estimate_c_from_raw_files`).

### 2.6 Consensus Validity

Given a per-base error rate epsilon, if the key appears N_key times in the read set,
then:

    N_key = Omega((1 - epsilon)^{-2v} * log(v))

ensures the highest-count value matches the true genomic sequence with high probability
(Supplementary Note S1). In practice, the default threshold is N_key >= 5 (without
reference) or N_key >= 0 (with reference, since the reference provides the consensus
directly).

---

## 3. Hazard Rate Estimation from (k, v)-mers

### 3.1 Computing Hazard Rates Per Key

For a key K, let N_{K,t} be the number of (k, v)-mers that have key K and whose value
matches the consensus up to the t-th base (i.e. the first t - k bases of the value
agree with the consensus).

The estimated hazard rate for key K at position t is:

    h_hat_K(t) = 1 - N_{K,t} / N_{K,t-1}      for k < t <= k + v

This is the fraction of values that "fail" (diverge from consensus) at exactly
position t, among those that survived up to t - 1.

### 3.2 Aggregating Across Keys

The global hazard rate estimate at position t pools across all keys:

    h_hat(t) = 1 - (sum_K N_{K,t}) / (sum_K N_{K,t-1})      for k < t <= k + v

This is equation (1) in the paper. It gives h(t) estimates in the interval
[k + 1, k + v] (a window of v values).

### 3.3 Implementation

In `summary.rs`, `ErrorSummary` accumulates per-key statistics:
- `consensus_counts[i]`: N_{K,k+v} for key i (full-length consensus matches)
- `total_counts[i]`: total observations of key i
- `consensus_up_to_v_counts[j][i]`: N_{K,k+j+1} for key i at prefix length j+1

In `inference.rs`, `ErrorAnalyzer` computes the aggregate hazard rates and fits the
Weibull model.

---

## 4. Discrete Weibull Survival Model

### 4.1 Model Assumption

Skiver assumes T follows a **discrete Weibull distribution** with parameters lambda > 0
and beta > 0. The survival function is:

    S(t) = exp(-lambda * t^beta)

And the hazard rate is:

    h(t) = 1 - exp(-lambda * (t^beta - (t-1)^beta))

### 4.2 Why Weibull (Not Constant Hazard)

A constant hazard rate (beta = 1) corresponds to a geometric distribution, meaning
errors are independently and uniformly distributed along reads. Real sequencing data
shows that:

- Hazard rates **decrease** with t (errors cluster near read ends or in bursts)
- The fitted beta is typically **less than 1**, indicating decreasing hazard

The Weibull model captures this heterogeneity. The paper's ablation study (Table 2)
shows the Weibull model consistently outperforms constant-hazard across all tested
datasets, with the gap being most pronounced for Nanopore data where errors cluster
along reads.

The constant-hazard variant is available via `--hazard-model constant`.

### 4.3 Parameter Estimation via Complementary Log-Log Transform

Taking the complementary log-log (cloglog) transformation:

    log(-log(1 - h(t))) = log(lambda * beta) + (beta - 1) * log(t)
                        approx log(lambda * beta * t^{beta-1})

More precisely, starting from:

    log(1 - h(t)) = -lambda * (t^beta - (t-1)^beta)
                   approx -lambda * beta * t^{beta-1}

So:

    log(-log(1 - h(t))) approx log(lambda * beta) + (beta - 1) * log(t)

This is a **linear relationship** between log(-log(1 - h(t))) and log(t). A line fit
to (log(t), log(-log(1 - h(t)))) for t in [k+1, k+v] yields:

    slope = beta - 1       =>  beta_hat = slope + 1
    intercept = log(lambda * beta)  =>  lambda_hat = exp(intercept) / beta_hat

### 4.4 Huber Ridge Regression

The line fit uses **Huber ridge regression** (`huber.rs`) rather than ordinary least
squares. This is a robust regression method that:

1. **Huber loss** (instead of squared loss): reduces the influence of outliers by
   switching from quadratic to linear penalty beyond a threshold delta. This is
   important because some t values may have noisy hazard estimates (especially at
   large t where fewer observations survive).

2. **Ridge penalty** (L2 regularization on slope): prevents extreme parameter estimates
   when data is sparse or noisy.

The last `ignore_last_hazard_ratios` (default 2) hazard rate estimates are excluded
from the fit, as they tend to be noisier due to fewer surviving observations.

### 4.5 Derived Quantities

From the fitted lambda_hat and beta_hat:

**Per-base error rate:**

    epsilon_perbase = 1 - exp(-lambda_hat)
                    = h_hat(1)

This is the probability that a single random base is erroneous.

**Effective error rate:**

    epsilon_eff = 1 / E[T]

where E[T] is the expected value of T under the fitted Weibull. The effective error
rate captures the rate at which errors occur when accounting for clustering. When
errors cluster (beta < 1), epsilon_eff < epsilon_perbase because the effective spacing
between error events is larger than what a uniform model would predict.

**Survival rate estimate:**

    S_hat(t) = exp(-lambda_hat * t^{beta_hat})

---

## 5. Error Spectrum Estimation

### 5.1 Decomposing Error Types

After identifying the consensus value for each key, skiver finds all **distance-1
neighbours**: values that differ from the consensus by exactly one edit operation
(substitution, insertion, or deletion).

The `EditOperation` enum in `types.rs` defines 20 operations:
- 12 substitutions: A>C, A>G, A>T, C>A, C>G, C>T, G>A, G>C, G>T, T>A, T>C, T>G
- 4 insertions: ->A, ->C, ->G, ->T
- 4 deletions: A>-, C>-, G>-, T>-

Neighbour computation is in `utils.rs` (`_get_neighbors`). If a value can be reached
from the consensus by **multiple** different single edits, it is marked `AMBIGUOUS` and
excluded from spectrum counting.

### 5.2 Constant Composition Assumption

The paper assumes that the **error composition is roughly constant and independent of
t**. That is, for a given single-base error type e (e.g. C>A), its hazard rate is:

    h_e(t) = pi_e * h(t)

where pi_e is a constant. The probability that error e happens exactly once in the
value, with no other errors, is:

    Pr[error e happens exactly once] = sum_{t=k+1}^{k+v} S(t-1) * h_e(t) * S(k+v-t)
                                     proportional to pi_e

So pi_e can be estimated from the frequency of error e among all single-error
observations. The output file `summary_error_spectrum.csv` reports these counts, and
the `summary_error_spectrum_dependence_on_v.csv` file allows verification that the
composition is indeed approximately constant across t.

### 5.3 Canonical (Bidirectional) Operations

When both forward and reverse complement strands are used (default), some edit
operations are equivalent under reverse complementation. The 20 operations collapse
to **10 canonical operations** (`ALL_OPERATIONS_CANONICAL` in `types.rs`):

- 6 substitutions: C>A, C>G, C>T, T>A, T>C, T>G
- 2 insertions: ->G, ->T
- 2 deletions: G>-, T>-

The `BASES_TO_SUBSTITUTION_CANONICAL` lookup table maps observed (from, to) pairs to
their canonical representative.

### 5.4 SBS96 Spectrum

The error spectrum also records the **trinucleotide context** (previous base, operation,
next base) for substitutions, producing the standard SBS96 mutational signature format.
The `sbs96_str` function in `types.rs` formats these as e.g. "A[C>T]G".

### 5.5 Computational Efficiency

There are at most 11v distance-1 neighbours of a consensus value of length v
(3v substitutions, 4(v+1) insertions, v deletions, approximately). The time to count
error frequencies is O(v * #keys + #values), which is much more efficient than aligning
all values to consensus (O(v^2 * #values)).

---

## 6. Outlier Filter for Repeat/Multi-Strain Regions

### 6.1 Problem

Keys from repetitive regions, multiple alleles, or co-existing strains will have
**multiple true values** in the sequenced genomes. This makes the hazard rate for those
keys artificially high (many values diverge from "consensus" not due to error but due
to biological variation).

### 6.2 Algorithm (Algorithm 1 in paper)

The outlier filter (`inference.rs`) works as follows:

```
1. Initialize: keep_key[K] = true for all K
2. For each t from k+1 to k+v:
   a. Compute h_K(t) = 1 - N_{K,t} / N_{K,t-1} for each key K
   b. Collect all h_K(t) where h_K(t) > 0 into hazard_rates
   c. Compute IQR = interquartile_range(hazard_rates)
   d. For each key K:
      if h_K(t) > median(hazard_rates) + 3 * IQR:
         keep_key[K] = false
3. Return {K : keep_key[K] is true}
```

This is a per-value-of-t outlier test using the **median + 3*IQR** rule (a robust
version of the 3-sigma rule). A key is discarded if its hazard rate is an outlier at
**any** value of t.

### 6.3 When the Filter is Disabled

- When a reference genome is provided (`-r`), keys with multiple values in the
  reference are already discarded (the reference provides the true consensus), so the
  outlier filter is not needed.
- The filter can be explicitly disabled with `--use-all`.

### 6.4 Binomial Outlier Test (Alternative)

The `--outlier-threshold` (`-e`) parameter controls a **binomial test**: a key is
removed if P(X <= observed) < threshold under the fitted Weibull hazard model. The
default threshold is 1e-9. This tests whether the observed survival count for a key
is implausibly low given the fitted error model.

---

## 7. Bootstrap Confidence Intervals

### 7.1 Procedure

Skiver estimates 5th-95th percentile confidence intervals via bootstrapping
(`inference.rs`):

1. Compute the point estimate of hazard rates from all keys
2. Repeat `num_experiments` times (default 100):
   a. Resample keys with replacement
   b. Recompute aggregate hazard rates from the resampled keys
   c. Refit the Weibull parameters (lambda, beta)
   d. Derive per-base error rate, effective error rate, and coverage
3. Report the 5th and 95th percentiles of each quantity across bootstrap replicates

### 7.2 Design Choice

Resampling is done at the **key** level (not individual observations), preserving the
correlation structure within each key's observations. This is appropriate because
keys represent independent genomic positions.

---

## 8. Coverage Estimation

### 8.1 Key Coverage

The observed (key) median coverage is simply the median of N_{K,k+v} across all keys
K that pass the outlier filter.

### 8.2 True Coverage Estimation

The observed coverage is deflated by sequencing errors (erroneous k-mers create
spurious keys). The true coverage is estimated by dividing the observed coverage by
the survival probability of the key:

    true_coverage = observed_coverage / S_hat(k)
                  = observed_coverage / exp(-lambda_hat * k^{beta_hat})

---

## 9. Additional Analyses

### 9.1 Phred Score Calibration

For each Phred quality score Q, skiver tallies:
- `num_correct`: bases with score Q that match the consensus
- `num_error`: bases with score Q that differ from consensus

The empirical error rate is compared against the theoretical rate
10^{-Q/10}. This reveals whether quality scores are well-calibrated.

Implemented in `summary.rs` (`PhredScoreSummary`). The `--first-base-only` flag
controls whether only the first value base is considered or all bases up to the
first error.

### 9.2 Read Position Error Profile

For each position index relative to the start/end of the read, skiver tallies
correct vs. error counts. This reveals positional biases (e.g. elevated error rates
at read ends in Illumina data).

Implemented in `summary.rs` (`ReadPositionSummary`).

### 9.3 Reference-Based Mode

When a reference genome is provided (`-r`):
- (k, v)-mers are extracted from the reference using the same hash/subsampling
- For each key in the reference, if it has a unique value, that value becomes the
  consensus (keys with multiple reference values indicate repeats and are discarded)
- The lower bound threshold defaults to 0 (since the reference provides ground truth
  for the consensus, even low-coverage keys are usable)

---

## 10. Evaluation Metrics (from the paper)

The paper uses two MSE metrics to evaluate survival and hazard rate estimates:

**MSE_S** (survival rate MSE):

    MSE_S = (1/n) * sum_{t=1}^{n} (S(t) - S_hat(t))^2

where n = 100 (analogous to the Brier score in survival analysis).

**MSE_h** (hazard rate MSE):

    MSE_h = (1/n) * sum_{t=1}^{n} (h(t) - h_hat(t))^2

where h_hat(t) = 1 - exp(-lambda_hat * (t^beta_hat - (t-1)^beta_hat)) for skiver,
and h_hat(t) = epsilon_hat for methods that only report a single error rate.

---

## 11. Map of Mathematics to Code

| Concept | Code location | Key function/struct |
|---------|--------------|-------------------|
| (k,v)-mer storage | `kvmer.rs` | `KVmerSet`, `key_value_qual_map` |
| Per-observation metadata | `types.rs` | `ValueInfo` |
| Subsampling (FracMinHash) | `seeding.rs` | `mm_hash64_masked` |
| Consensus identification | `kvmer.rs` | `get_stats`, `get_stats_with_reference` |
| Neighbour computation | `utils.rs` | `_get_neighbors` |
| Edit operations | `types.rs` | `EditOperation`, `NeighborInfo` |
| Per-key error accumulation | `summary.rs` | `ErrorSummary::accumulate` |
| Error spectrum accumulation | `summary.rs` | `ErrorSpectrumSummary` |
| Phred calibration | `summary.rs` | `PhredScoreSummary` |
| Read position profile | `summary.rs` | `ReadPositionSummary` |
| Aggregate hazard estimation | `inference.rs` | `ErrorAnalyzer` |
| Weibull fitting (cloglog regression) | `inference.rs` | `ErrorAnalyzer` |
| Huber ridge regression | `huber.rs` | `huber_ridge_fit_1d` |
| Outlier filter | `inference.rs` | Algorithm 1 implementation |
| Bootstrap CI | `inference.rs` | bootstrap loop |
| Coverage estimation | `inference.rs` | `ErrorSpectrum` fields |
| CLI argument parsing | `cmdline.rs` | `AnalyzeArgs`, `SketchArgs` |
| Pipeline orchestration | `analyze.rs` | `analyze()` |
| Sketch I/O | `sketch.rs`, `kvmer.rs` | `dump`, `load` (bincode) |
| Reference-based map testing | `mapping.rs` | `KmerSet`, `map()` |
| AVX2-accelerated seeding | `avx2_seeding.rs` | x86_64-only |
