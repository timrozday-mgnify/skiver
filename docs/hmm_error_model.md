# Sequencing Error Model: Design Reference

This document describes the full sequencing error characterisation pipeline
built on top of skiver's (k,v)-mer sketch outputs. The pipeline has five
complementary components:

| Component | What it measures | Output |
|-----------|-----------------|--------|
| **Weibull survival model** | Scalar per-base error rate, with CIs | `summary_error_rate.csv` |
| **Phred calibration** | Empirical error rate per Phred score vs. theoretical | `summary_phred.csv` |
| **Error spectrum** | Which error types occur and in which trinucleotide context | `summary_error_spectrum.csv` |
| **Spectrum vs. position** | How error type distribution changes along the value window | `summary_error_spectrum_dependence_on_t.csv` |
| **Read-position dependence** | How error rate changes from read start to end | `summary_read_position.csv` |
| **HMM error model** | Latent quality-regime structure within each value window | `base_observations.tsv` → trained `.pt` |

The first five are produced by `skiver analyze` and visualised by `scripts/plot_*.py`.
The HMM is trained separately by `scripts/train_hmm_error_model.py` using
`skiver dump --base` output.

This document covers all six components and explains how they relate.

---

---

## 1. Weibull survival model (recap)

The foundational output of `skiver analyze` is a scalar per-base error rate
estimated via a discrete Weibull survival model — see `docs/paper_reference.md`
for full mathematical detail. Briefly:

- For each high-coverage key, skiver observes how many value occurrences first
  disagree from consensus at position t = 1, 2, …, v.
- The resulting empirical survival curve S(t) = Pr[T > t] is fitted to a
  Weibull model S(t) = exp(−λ · t^β).
- The **per-base error rate** is λ (= h(1) when β ≈ 1), with bootstrap
  confidence intervals.
- The **effective error rate** weights by the actual distribution of read
  lengths, giving the marginal probability that a random base in a random read
  is wrong.

All other components described here either sharpen this scalar estimate (Phred
calibration, spectrum), contextualise it (read position, spectrum-vs-t), or
model its within-sequence structure (HMM).

---

## 2. Phred score calibration

### 2.1 What it measures

Each base in a FASTQ file carries a Phred quality score Q, defined by the
instrument to satisfy P(error) = 10^(−Q/10). In practice instruments are
often miscalibrated — the reported Q may systematically over- or under-estimate
the true error probability.

Skiver measures calibration by grouping bases by their reported Q score and
computing the empirical error rate in each group (using consensus disagreement
as the ground truth for "error"):

```
empirical_Q(q) = −10 · log₁₀(num_error(q) / (num_correct(q) + num_error(q)))
```

### 2.2 Output: `summary_phred.csv`

| Column | Meaning |
|--------|---------|
| `qscore` | Reported Phred score (integer) |
| `empirical_qscore` | Empirically derived Phred score at this bin |
| `num_correct` | Bases agreeing with consensus at this Q |
| `num_error` | Bases disagreeing with consensus at this Q |
| `error_rate` | `num_error / (num_correct + num_error)` |

### 2.3 Visualisation: `plot_qscore_calibration.py`

Plots empirical error rate against reported Q (solid line) with the theoretical
curve 10^(−Q/10) (dashed), plus a histogram of how many bases fall in each Q
bin. A well-calibrated instrument should show the two curves overlapping. A
curve that lies above the theoretical line means errors are more common than
reported; below means the instrument is conservative.

### 2.4 Relationship to the HMM

The HMM's observation encoding bins Phred scores into 8 coarse bins (Q0–4,
Q5–9, …, Q35+). The learned emission distributions `B_{s,c}` implicitly encode
the joint (quality bin, error type) distribution per state, capturing any
miscalibration within each state's regime. However, the HMM does not produce
a calibration curve — use `summary_phred.csv` for that diagnostic.

---

## 3. Error spectrum and substitution matrix

### 3.1 What it measures

The **error spectrum** decomposes the total error rate by *type* of error:
which base was the true consensus base, which base was observed instead, and
(for the SBS96 form) what were the flanking bases in the reference sequence.

Skiver records each 1-edit-distance value occurrence's operation type (e.g.
`C>T`, `->A`, `G>-`) and the preceding and following bases of the *key* (which
acts as the flanking context). This gives a 96-channel SBS-style spectrum
familiar from cancer genomics, plus 4 insertion and 4 deletion channels.

### 3.2 Output: `summary_error_spectrum.csv`

Long-format table with one row per (operation, prev_base, next_base) triple:

| Column | Meaning |
|--------|---------|
| `operation` | Edit type, e.g. `C>T`, `->A`, `G>-` |
| `prev_base` | Base immediately before the value in the key (5′ context) |
| `next_base` | Base immediately after the value in the key (3′ context) |
| `total` | Count across both strands |
| `forward` | Count on the forward strand only |

### 3.3 The 5×5 substitution matrix

`plot_spectrum.py` builds a 5×5 matrix (rows = true base ∈ {A,C,G,T,−},
columns = observed base ∈ {A,C,G,T,−}) by summing over all trinucleotide
contexts. Off-diagonal cells are errors; the diagonal is suppressed. The
matrix is scaled so off-diagonal entries sum to the per-base error rate
(from `summary_error_rate.csv`), giving absolute error rates per substitution
type rather than relative proportions.

Two rows are shown: one for both strands combined (`total`), one for the
forward strand only (`forward`). Comparing the two reveals strand-asymmetric
errors — e.g. oxidative damage (G→T / C→A complement) appears predominantly
on one strand.

### 3.4 The SBS96 spectrum

`plot_sbs96_spectrum.py` renders the classic SBS96 bar chart: 6 substitution
types × 16 trinucleotide contexts = 96 bars. In bidirectional mode (default),
skiver collapses to canonical pyrimidine-centred substitutions (C→\* and T→\*),
producing a standard SBS96 chart. In unidirectional mode all 12 substitution
types are shown.

The SBS96 spectrum is useful for:
- Attributing sequencing artefacts to known damage signatures
- Comparing error profiles across sequencing platforms or chemistries
- Checking that the error model is not confounded by biological mutations
  (which would show mutation-signature-like patterns)

### 3.5 Relationship to the HMM

The HMM emission categories encode `(true_base, obs_base, phred_bin)` — the
first two dimensions are exactly the row and column of the 5×5 substitution
matrix. Marginalising the learned emissions over Phred bins recovers a
per-state substitution matrix:

```
M_{s, tb, ob} = Σ_{pb} B_{s, encode(tb, ob, pb)}
```

The HMM does not model trinucleotide context (no flanking base information is
in `base_observations.tsv`). For context-sensitive error rates, use the
`summary_error_spectrum.csv` spectrum directly.

---

## 4. Error spectrum dependence on position t

### 4.1 What it measures

The error type distribution may shift along the value window. For example,
certain chemistry-induced damage (e.g. deamination of cytosine at the 3′ end)
may concentrate at later positions in the value. This component shows whether
the *relative proportions* of error types change with t.

### 4.2 Output: `summary_error_spectrum_dependence_on_t.csv`

Same (operation, prev_base, next_base) rows as the spectrum file, plus one
column per value-window position:

| Column | Meaning |
|--------|---------|
| `operation` | Edit type |
| `prev_base`, `next_base` | Context bases |
| `total` | Total count summed over all t |
| `freq_at_t{T}` | Count of this operation at value position T |

### 4.3 Interpretation

Plotting `freq_at_t{T}` across positions T reveals whether, say, C→T
transitions are enriched at position t=v (the last position) compared to
earlier positions. This is a check on whether the error model should be
position-dependent — motivating the HMM's ability to capture positional regime
changes through its transition structure.

---

## 5. Read-position dependence

### 5.1 What it measures

Error rate is expected to increase towards the end of a read as the polymerase
or chemistry degrades. Skiver computes the empirical error rate at each
absolute read position (from both the start and the end of the read), using
the `start_index` and `dist_to_read_end` fields stored in `ValueInfo`.

### 5.2 Output: `summary_read_position.csv`

| Column | Meaning |
|--------|---------|
| `index` | Absolute position (0-based from start or from end) |
| `from_start` | `true` if measured from read start, `false` from read end |
| `num_correct` | Bases agreeing with consensus at this position |
| `num_error` | Bases disagreeing at this position |
| `error_rate` | `num_error / (num_correct + num_error)` |

### 5.3 Visualisation: `plot_read_position.py`

Two panels: error rate from read start (left) and from read end (right), each
with a smoothed overlay (uniform filter, window = 10% of the plotted range)
and a base-count histogram below. The characteristic shape for Illumina short
reads is a flat plateau across most of the read with a sharp increase in the
last 10–20 bases.

### 5.4 Relationship to the HMM

The HMM operates at the scale of a single value window (~10–30 bp), not the
full read. Read-position degradation shows up as a systematic difference
between keys whose values land near the read end vs. the read start. The HMM
does not condition on `start_index` — doing so would require a covariate-
conditional emission model. The read-position plot is therefore a complementary
diagnostic: if the per-base error rate rises sharply near the read end, it is
worth training separate HMMs on early-read and late-read value windows.

---

## 6. HMM error model

The components above characterise the error distribution in aggregate — scalar
rates, calibration curves, and spectrum counts. The HMM goes further by
modelling the **sequential structure** of errors within a single value-window
occurrence: the sequence of base positions is treated as a Markov chain of
latent quality regimes rather than a bag of independent draws.

### 6.1 Data source: `skiver dump --base`

The HMM is trained on `{prefix}.base_observations.tsv`, produced by
`skiver dump --base`. Each row is one base position within one (k,v)-mer value
occurrence. Rows sharing an `obs_id` form a sequence of length `v` aligned to
the consensus value for that key.

| Field | Meaning |
|-------|---------|
| `true_base` | Consensus base at position `t` (ACGT or `-` for insertion) |
| `obs_base` | Observed base at position `t` (ACGT or `-` for deletion) |
| `phred` | Integer Phred score (qual byte − 33); −1 if unavailable |
| `t` | 1-based position within the value window |
| `passes_filter` | Whether the key passed the per-key outlier filter |

Only observations with `passes_filter=true` are used by default.

### 6.2 Observation encoding

Each base position is encoded as a single integer category:

```
category = true_base_idx × (NUM_BASES × NUM_PHRED_BINS)
         + obs_base_idx  × NUM_PHRED_BINS
         + phred_bin
```

- `NUM_BASES = 5` (A, C, G, T, gap)
- `NUM_PHRED_BINS = 8` (Q0–4, Q5–9, …, Q35+)
- `NUM_OBS_CATEGORIES = 200`

This captures error occurrence, error type, and instrument confidence jointly,
without assuming independence between them — unlike a factored model that
multiplies separate distributions for each dimension.

### 6.3 Model structure

The model has `S` hidden states (default 3). Three parameter tensors are
learned:

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| `initial_logits` | `[S]` | Log-odds of starting in each state |
| `transition_logits` | `[S, S]` | Log-odds of transitioning from state `i` to `j` |
| `emission_logits` | `[S, 200]` | Log-odds of emitting each category from state `s` |

The joint probability of an observation sequence `x = (x_1, …, x_v)` is:

```
p(x) = Σ_{z_1,…,z_v} π_{z_1} · B_{z_1, x_1} · Π_{t=2}^{v} A_{z_{t-1}, z_t} · B_{z_t, x_t}
```

computed via the forward algorithm in O(v · S²) per sequence, using Pyro's
`DiscreteHMM` distribution.

### 6.4 Training: MAP via SVI

Training maximises the marginal log-likelihood `L(θ) = Σ_n log p(x^(n); θ)`.
All parameters are `pyro.param`; the guide is empty. `Trace_ELBO` with an
empty guide reduces to the exact log-likelihood (not a lower bound). Adam
optimiser, lr = 0.005, 1000 steps by default.

**Initialisation.** The true error rate (~0.07%) means ~199 of the 200
categories are matches. Uniform logit initialisation places every state at
~60% error rate, far from the correct solution. The fix:

- Match categories: logit = phred_bin index (0–7), so higher-quality matches
  start more probable.
- Mismatch categories: logit = −8 + state_index, providing slight per-state
  offset to break symmetry.

The transition matrix is initialised as `eye(S) × 3.0` (sticky), encouraging
states to represent persistent regimes.

### 6.5 Interpreting the output

**Per-state error rate:**
```
error_rate(s) = Σ_{c : true_base(c) ≠ obs_base(c)} B_{s,c}
```

States typically differ more in their Phred-bin distributions than in raw
error rate. A state concentrated on Q35+ represents high-confidence positions;
a state with mass across Q10–Q24 represents lower-confidence stretches.

**Mean run length** of state `s`:
```
E[run] = 1 / (1 − A_{ss})
```

**Per-state substitution matrix:** marginalise over Phred bins:
```
M_{s, tb, ob} = Σ_{pb} B_{s, encode(tb, ob, pb)}
```
This gives each state's 5×5 error spectrum, complementing the global spectrum
from `summary_error_spectrum.csv`.

**Consistency check:** the HMM's marginal error rate `Σ_s π_s · error_rate(s)`
should approximate the Weibull per-base error rate from `summary_error_rate.csv`
when both are fitted to the same data.

### 6.6 HMM limitations

**Fixed value length.** All sequences must have the same length `v`. Mixing
prefixes with different `-v` introduces padding artefacts.

**No positional covariates.** Position `t` influences the hidden state only
through the Markov transition structure, not via a direct covariate. For
explicit position effects, see `summary_error_spectrum_dependence_on_t.csv`.

**No read-pair or strand conditioning.** R1 and R2 have different error
profiles; train separate models per read pair. Strand (`is_forward`) is
available in `base_observations.tsv` but unused.

**Point estimation.** SVI with an empty guide gives no posterior uncertainty.
For small datasets, add Dirichlet priors on emission and transition rows with
`pyro.sample` and an `AutoDelta` guide.

---

## 7. Profile HMM (context-dependent model)

The basic HMM (§6) uses a flat 200-category emission encoding that is
context-blind — it does not distinguish dinucleotide context, error type, or
position within the value window. The **profile HMM** extends this with
factored, context-dependent emissions inspired by PBSIM3's error-type HMM
and Hercules' profile HMM (see `docs/pbsim3_reference.md` and
`docs/hercules_reference.md`).

### 7.1 Architecture

S = 4 latent quality-regime states (configurable). Emissions are factored:

**Component A — Error type:**
```
P(error_type | state, prev_base, true_base, position_t)
```

10 error types: match, 4 substitutions (to A/C/G/T), 4 insertions (A/C/G/T),
1 deletion.

Parameter shape: `error_type_logits[S, 16, T, 10]` — indexed by state,
dinucleotide context (prev_base × true_base = 4×4 = 16), position, and
error type.

**Component B — Phred quality:**
```
P(phred_bin | error_class, state, position_t)
```

8 Phred bins (Q0–4, Q5–9, ..., Q35+), conditioned on 3 coarse error classes
(match, mismatch, indel).

Parameter shape: `phred_logits[S, 3, T, 8]`.

**Joint log-probability:**
```
log P(obs | z_t, ctx_t) = log P(error_type | z_t, prev_base, true_base, t)
                        + log P(phred_bin | error_class, z_t, t)
```

### 7.2 Dinucleotide context

The `prev_base` column in `base_observations.tsv` provides the preceding base:
- At t=1: the last base of the k-mer key
- At t>1: the previous consensus base

This is combined with `true_base` to form 16 dinucleotide contexts, enabling
the model to capture context-dependent error patterns (e.g., C→T transitions
are enriched in CpG context for oxidative damage).

### 7.3 Parameter count

With S=4, T=13, 16 contexts, 10 error types, 3 error classes, 8 Phred bins:
- Initial logits: 4
- Transition logits: 4 × 4 = 16
- Error-type logits: 4 × 16 × 13 × 10 = 8,320
- Phred logits: 4 × 3 × 13 × 8 = 1,248
- **Total: ~9,588 parameters**

### 7.4 Training

Same SVI approach as the basic HMM. Key differences:
- Uses `ClippedAdam` optimiser with gradient clipping
- Stratified subsampling keeps all error-containing sequences; subsamples
  error-free to 50:1 ratio (configurable)
- Informative initialisation: match logit ~4, error logits ~-8 with per-state
  offset for symmetry breaking
- Default: 2000 SVI steps, lr=0.005

### 7.5 Usage

```bash
# Train
python scripts/train_profile_hmm.py \
    ../skiver_run/mimicc_example/250700000051_25Nov5669-DL133_S133_L001_R1 \
    -o profile_hmm.pt --states 4 --steps 2000

# With both read pairs
python scripts/train_profile_hmm.py prefix_R1 prefix_R2 \
    -o profile_hmm.pt
```

### 7.6 Comparison with basic HMM

| Feature | Basic HMM (§6) | Profile HMM (§7) |
|---------|----------------|-------------------|
| Emissions | 200 flat categories | Factored: 10 error types × 8 Phred bins |
| Context | None | Dinucleotide (16 contexts) |
| Position dependence | Only via Markov transitions | Explicit per-position emission parameters |
| Error types | Implicit (true≠obs) | Explicit (sub/ins/del separated) |
| Parameters | ~600 (S=3) | ~9,600 (S=4) |
| Script | `train_hmm_error_model.py` | `train_profile_hmm.py` |

---

## 8. Composable context error model

The non-HMM context error model (`scripts/train_context_error_models.py`,
`scripts/lib/context_error_models.py`) models:

```
P(error_type | preceding bases, covariates)
```

as a categorical distribution whose log-probabilities are a linear combination
of a base-context table and optional additive covariate contributions. Each
model is fully described by a **component string** — a `+`-separated list of
tokens specifying which effects to include.

### 8.1 Component reference

Components are separated by `+`.  Parameterised components use `Name(N)` syntax.
Exactly one context component (`BaseContext` or `AdditiveContext`) is required;
all others are optional and may appear in any order.

#### `BaseContext(N)` — combinatorial context table

Models the full joint distribution of the N preceding consensus bases as
independent parameter rows.  Encodes 4^N context rows, each with
`NUM_ERROR_TYPES − 1` free parameters.

- **Use when:** N ≤ 4.  Four bases of context = 256 rows.  Larger N becomes
  memory-prohibitive and the table grows too sparse to fit reliably.
- **Parameters added:** 4^N × (E − 1) where E = 10 error types.
- **Example:** `BaseContext(2)` — conditions on the two preceding bases.

#### `AdditiveContext(N)` — factored additive context

Replaces the full joint table with an additive decomposition: each of the N
preceding positions contributes independently to log-odds.

```
logit(error_type | ctx) = intercept + Σ_{i=1}^{N} base_effect[i, base_i]
```

- **Use when:** N ≥ 5.  Scales linearly with N; a 12-base model uses
  `(1 + 3×12) × (E − 1)` ≈ 333 parameters rather than 4^12 × 9 ≈ 38 million.
- **Parameters added:** `(1 + (4 − 1) × N) × (E − 1)`.
- **Example:** `AdditiveContext(8)` — 8 preceding bases, ~270 free parameters.

#### `PhredContext(N)` — Phred quality covariates

Adds N preceding Phred quality scores as additive linear covariates.  The N
most-recently observed integer Phred values are averaged per context row before
the dot-product with learned weight matrices, so a context row with many
observations effectively sees the mean Phred lag at that context.

- **What it captures:** sequencer miscalibration — the model learns how much
  the instrument-reported quality deviates from the true error probability,
  conditional on base context.
- **Parameters added:** N × (E − 1).
- **Example:** `PhredContext(3)` — uses the 3 most-recent Phred scores.

#### `Position(N)` — read-position covariates

Adds N read-position features as additive covariates.  N must be 1 or 2.

- **N = 1:** `log1p(dist_to_read_end)` — captures 3′-end degradation.  The
  log transform compresses the long tail of very-long reads.
- **N = 2:** additionally adds `log1p(read_pos)` — captures both 3′ and 5′
  effects.  Use N = 2 when both ends show elevated error rates (e.g. PacBio).
- **Parameters added:** N × (E − 1).
- **Example:** `Position(1)` for Illumina (3′ degradation only).

#### `Strand` — strand-asymmetry covariate

Adds the per-context forward-strand fraction as a scalar additive covariate.
Captures strand-asymmetric damage chemistry such as oxidative G→T / C→A
paired damage or FFPE-type C→T deamination enriched on one strand.  When
strand asymmetry is absent the learned weight vector converges to zero.

- **Parameters added:** 1 × (E − 1) = 9 parameters.
- **Example:** `Strand`.

#### `FragmentOverdispersion` — Dirichlet-Multinomial likelihood

Replaces the Multinomial likelihood with a Dirichlet-Multinomial (DM).  Models
between-fragment variability in error rate: different DNA fragments may carry
different error loads (e.g. due to damage heterogeneity across the sample).

- **What it adds:** one scalar concentration parameter φ (> 0).  When
  φ → ∞ the DM recovers the Multinomial.  Requires `fragment_id` column in
  the TSV files (produced by `skiver dump`).
- **Parameters added:** 1 (the scalar φ).
- **Example:** `FragmentOverdispersion`.

#### `Homopolymer` — run-length effect *(screen aggregation not yet supported)*

Adds a learnable monotone scalar run-length transform and per-context slope to
model the elevated indel rates in homopolymer runs.  Currently only available
through the platform-level aggregation path (`aggregate_platform_counts`); raises
`NotImplementedError` when used with `aggregate_context_length_screen_counts`.

---

### 8.2 Component string examples

| String | Description |
|---|---|
| `"BaseContext(1)"` | Single-base context, no covariates. Minimal baseline. |
| `"BaseContext(3)"` | Trinucleotide context. Standard starting point for Illumina. |
| `"AdditiveContext(6)"` | 6-base additive context. Good balance for all platforms. |
| `"AdditiveContext(12)"` | 12-base additive context. Long-range context for ONT. |
| `"AdditiveContext(6)+Strand"` | Adds strand-asymmetry correction. |
| `"AdditiveContext(6)+PhredContext(3)"` | Context + 3 Phred lags. Captures calibration effects. |
| `"AdditiveContext(6)+PhredContext(3)+Position(1)"` | Adds 3′ degradation. Recommended for Illumina. |
| `"AdditiveContext(6)+PhredContext(3)+Position(2)+Strand"` | Full covariate set for short-read platforms. |
| `"AdditiveContext(8)+Position(2)+Strand+FragmentOverdispersion"` | With between-fragment overdispersion. |

---

### 8.3 Model config JSON format

Models are listed in `scripts/model_config.json`.  Each entry needs only `"id"`
and `"components"`:

```json
{
  "models": [
    {"id": "context_1",   "components": "BaseContext(1)"},
    {"id": "context_3",   "components": "BaseContext(3)"},
    {"id": "additive_6",  "components": "AdditiveContext(6)"},
    {"id": "ctx_phred",   "components": "AdditiveContext(6)+PhredContext(3)+Position(1)+Strand"}
  ]
}
```

All models in the config file are trained in one run.  Each model is
independently fitted on the same aggregated count tensors, so adding a new
model to the config does not require re-reading the TSV files.

---

### 8.4 Training

```bash
cd scripts

# Basic: all models in model_config.json, all platforms
python train_context_error_models.py --data-root ../skiver_run

# With Phred and position covariates (overrides per-model settings)
python train_context_error_models.py \
  --data-root ../skiver_run \
  --phred-lags 3 \
  --position-features 1

# With fragment overdispersion (requires fragment_id column in TSVs)
python train_context_error_models.py \
  --data-root ../skiver_run \
  --fragment-overdispersion

# Strand covariate globally for all models
python train_context_error_models.py \
  --data-root ../skiver_run \
  --strand-covariate

# Large additive models: cap training to 4096 most-observed contexts
python train_context_error_models.py \
  --data-root ../skiver_run \
  --max-contexts 4096

# Calibrate to a Weibull-estimated rate
python train_context_error_models.py \
  --data-root ../skiver_run \
  --weibull-rate 0.0072
```

Each trained model is saved as `{output-dir}/{model_id}_{platform}.pt` alongside
an AIC comparison CSV at `{output-dir}/context_model_aic.csv`.

---

### 8.5 Programmatic API

```python
from lib.context_error_models import (
    parse_model_components,
    aggregate_context_length_screen_counts,
    fit_and_test,
    log_likelihood,
    compute_marginal_error_rate,
    calibrate_to_rate,
    compute_marginal_weibull,
)

# Parse a component string
mc = parse_model_components("AdditiveContext(6)+PhredContext(3)+Strand")
# mc.context_length == 6, mc.additive_context == True,
# mc.num_phred_lags == 3, mc.use_strand == True

# Aggregate counts from TSV files
screen = aggregate_context_length_screen_counts(
    prefixes,                    # list of skiver dump prefixes
    context_lengths=[6],
    num_phred_lags=3,
    use_strand=True,
)
cc = screen.by_length[6]

# Train and evaluate (returns FitResult with params, losses, AIC)
result = fit_and_test(cc, test_cc, num_steps=1000)

# Evaluate on any ContextCounts — all evaluation functions take (cc, params)
ll    = log_likelihood(cc, result.params)
rate  = compute_marginal_error_rate(cc, result.params)
delta = calibrate_to_rate(cc, result.params, target_rate=0.0072)
wb    = compute_marginal_weibull(cc, result.params, v=13)
```

All evaluation functions (`log_likelihood`, `elbo_loss`,
`compute_marginal_error_rate`, `calibrate_to_rate`, `compute_marginal_weibull`)
take a `ContextCounts` object as their first argument.  The `ContextCounts`
carries all aggregated covariate tensors (`phred_context_sums`,
`position_context_sums`, `strand_context_sums`, `fragment_count_per_context`)
alongside the error-type counts, so there is no need to pass covariate tensors
separately.

---

### 8.6 Choosing components

| Platform | Recommended starting point |
|---|---|
| High-quality Illumina (HQ) | `AdditiveContext(6)+PhredContext(3)+Position(1)` |
| Low-quality / degraded Illumina | `AdditiveContext(6)+PhredContext(3)+Position(1)+Strand` |
| Oxford Nanopore (ONT) | `AdditiveContext(8)+Position(2)` |
| PacBio | `AdditiveContext(6)+Position(2)` |
| FFPE / ancient DNA | add `Strand` to any of the above |
| Paired-end (R1/R2 batch) | add `FragmentOverdispersion` to capture between-read variance |

Use AIC (the `aic` column in `context_model_aic.csv`) to compare model
configurations trained on the same dataset.  A lower AIC means better
predictive accuracy after penalising for parameter count.

---

## 9. How the components relate

```
skiver analyze → summary_error_rate.csv      (scalar λ, β, per-base error rate)
              → summary_phred.csv            (calibration: empirical Q vs reported Q)
              → summary_error_spectrum.csv   (which error types, in which context)
              → summary_error_spectrum_      (how error type varies with window position)
                  dependence_on_t.csv
              → summary_read_position.csv    (how error rate varies along the read)

skiver dump --base → base_observations.tsv (with prev_base column)
    → train_hmm_error_model.py  → hmm_error_model.pt
                                  (basic: latent quality regimes, per-state spectra)
    → train_profile_hmm.py      → profile_hmm.pt
                                  (context-dependent: dinucleotide, position, error type)
```

The components answer complementary questions:

| Question | Component |
|----------|-----------|
| What is the overall error rate? | Weibull model (`summary_error_rate.csv`) |
| Are quality scores accurate? | Phred calibration (`summary_phred.csv`) |
| Which errors dominate? | Error spectrum (`summary_error_spectrum.csv`) |
| Do error types shift along the window? | Spectrum vs. t |
| Does error rate increase near the read end? | Read position |
| Are errors clustered or independent within a window? | HMM |
| What is the error rate conditional on quality state? | HMM per-state error rate |
| What is the per-state substitution pattern? | HMM emission marginals |

---

## 9. Training data formats and the two-format tradeoff

### Two formats produced by `skiver dump`

After adding `read_id` to `ValueInfo` (commit adding `--windows`), skiver dump
produces two complementary training data formats:

| Format | Flag | Size (hq-illumina train) | Content |
|--------|------|--------------------------|---------|
| `{prefix}.base_observations.tsv` | `--base` | ~32 GB | One row per base position per kv-mer occurrence. Has `read_id`, `read_pos`, `prev_base`, `phred`, `edit_op`. Dense context features. |
| `{prefix}.windows.bin` | `--windows` | ~430 MB (~75× smaller) | One 13-byte record per sampled kv-mer: `(read_id: u64, start_index: u32, error_type: u8)`. No context features. |

`windows.bin` binary layout: 32-byte header (`b"SKIVRERR"`, version=1, k, v,
n_records) followed by 13-byte records, unsorted. Load with NumPy:

```python
import numpy as np, struct
dtype = np.dtype([('read_id', np.uint64), ('start_index', np.uint32), ('error_type', np.uint8)])
recs = np.fromfile(path, dtype=dtype, offset=32)
recs = recs[np.lexsort((recs['start_index'], recs['read_id']))]
```

Error type encoding matches `scripts/lib/encoding.py`:
`0`=match, `1–4`=sub_to_{A,C,G,T}, `5–8`=ins_{A,C,G,T}, `9`=deletion.
Only the error type at t=1 (position `start_index`) is recorded; multi-edit
observations are dropped.

### What each format trains

- **`base_observations.tsv`** → context-dependent emission probabilities:
  P(error_type | prev_base, phred, position, …). Used by the context error
  models in `scripts/lib/context_error_models.py`.

- **`windows.bin`** → read-level sparse error sequences, one observation per
  sampled kv-mer window. Used to train HMM latent state transition probabilities
  and per-state error rate scaling factors.

### The two-format tradeoff

Training these two components from separate datasets introduces a modeling
tension: the HMM trained on `windows.bin` sees observations without their
context features. It cannot distinguish "high error rate because prev_base=C"
from "high error rate because the read entered a degraded latent quality state."
The inferred latent states will absorb some context-driven variance, and the
transition probabilities will be slightly inflated.

**Why this is usually acceptable:** Context effects are rapidly-varying
(base-to-base scale), while latent quality states change slowly across a read
(spanning many kv-mer windows). These two timescales are mostly orthogonal. The
same two-stage training strategy is used by PBSIM3 and Hercules with acceptable
results.

**Diagnostic signal for a problem:** If inferred latent states correlate
strongly with context features (e.g., one state predominantly activates on
GC-rich keys, or on low-phred positions), the separate training is failing and
joint training is needed.

### How to do joint training if needed

`base_observations.tsv` now contains `read_id` (added alongside `--windows`).
Grouping by `read_id` and sorting by `read_pos` recovers read-level error
sequences with full context features — the same read-level structure as
`windows.bin` but enriched with `prev_base`, `phred`, `t`, etc. This single
dataset can train both the context emission model and the HMM transitions
jointly, removing the cross-dataset inconsistency. The cost is working with the
full ~32 GB TSV rather than the compact binary.

Concretely, for joint training:

1. Load `base_observations.tsv` grouped by `(read_id, read_pos)` (use the
   HDF5 row cache in `scripts/lib/context_h5_cache.py` for efficiency).
2. For each read, construct the sparse sequence of `(position, error_type,
   prev_base, phred)` tuples from accepted (non-multi-edit) rows.
3. Run the E-step of the HMM forward–backward algorithm using context-conditioned
   emission probabilities P(error_type | state, prev_base, phred) from the
   context model.
4. M-step updates both the context emission parameters and the transition matrix.

### Recommendation

Start with the two-format (separate training) approach. Use `windows.bin` for
HMM transition fitting and `base_observations.tsv` (via the H5 cache) for
context emission training. Switch to joint training from `base_observations.tsv`
only if the latent state diagnostic above shows contamination by context effects.

---

## 10. Synthetic context-model recovery benchmark

The context-model simulator and recovery benchmark are intended to answer a
specific validation question: if a trained Skiver context model is used by
genome-blender to generate reads, does the Skiver dump + retraining pipeline
recover the same error distribution?

This benchmark currently targets the non-HMM context models produced by
`scripts/train_context_error_models.py` (`context_error_models/*.pt`). It does
not generate from `scripts/train_hmm_error_model.py` artifacts.

### 10.1 One-command full-scale run

Use `scripts/benchmark_simulated_context_model.py` for the full genome-blender
generate, Skiver dump, retrain, and compare loop:

```bash
cargo build --release

python scripts/benchmark_simulated_context_model.py \
  --model context_error_models/additive_7_hq-illumina.pt \
  --reference reference.fa \
  -o synthetic_recovery/additive_7_hq_illumina \
  --skiver-bin target/release/skiver \
  --genome-blender-dir ../genome-blender \
  --genome-blender-conda-env genome_blender_dev \
  --joint-phred-calibration-json context_error_cache/hq-illumina_train_phred_calibration.json \
  --quality-calibration-model-json examples/amplicon_synthetic_recovery/source_quality_calibration_model.json \
  --quality-calibration-fit-model log-linear \
  --n-copies 1000 \
  --k 21 --v 13 --c 1 \
  --steps 1000 \
  --lr 0.05 \
  --seed 42
```

Recommended starting settings:

| Setting | Recommendation | Rationale |
|---------|----------------|-----------|
| `--model` | Use the exact `.pt` artifact being validated | The benchmark reads model type and context length from the artifact |
| `--reference` | Use a high-complexity FASTA with enough bases to cover many contexts | Repetitive short references under-sample context space |
| `--genome-blender-dir` | Point at the modified sibling genome-blender checkout | That repository applies the Skiver `.pt` artifact during read generation |
| `--genome-blender-conda-env` | Use `genome_blender_dev`, or set to `""` to use the current Python | The benchmark shells out to genome-blender for simulation |
| `--n-copies` | 1000 or more for short references; fewer for chromosome-scale references | This is the number of genome-blender single-end long amplicon reads per train/test split |
| `--k`, `--v`, `--c` | Match the intended Skiver analysis settings; start with `--c 1` | Recovery is easiest to interpret when no subsampling is introduced |
| `--steps` | 1000 for routine checks; 2000–5000 for final validation | More steps reduce optimizer error in the retrained model |
| `--joint-phred-calibration-json` | Use the empirical `P(Q \| error type)` JSON for the source platform | Samples quality scores jointly with the sampled Skiver error type |
| `--quality-calibration-model-json` | Optional source qcal artifact | Used as a `P(error \| Q)` validation target, not as the generator when joint Phred is provided |
| `--max-contexts` | Set only for very large additive models if training time is excessive | Caps retraining to the most-observed contexts |

The benchmark writes:

| Output | Meaning |
|--------|---------|
| `synthetic_train.genomes.csv`, `synthetic_test.genomes.csv` | One-genome abundance tables passed to genome-blender |
| `synthetic_train.fastq`, `synthetic_test.fastq` | Reads generated by genome-blender from the source model |
| `synthetic_train.bam`, `synthetic_test.bam` | genome-blender alignment outputs for the simulated reads |
| `dump_train.base_observations.tsv`, `dump_test.base_observations.tsv` | Skiver observations used for retraining |
| `retrained_context_model.pt` | Retrained Skiver context model artifact with MLE parameters and, by default, VI posterior summaries |
| `recovery_metrics.json` | Source vs Skiver-observed vs retrained probability comparisons |
| `vi_uncertainty_metrics.json` | Source vs retrained VI posterior standard-deviation comparisons |
| `source_phred_calibration.json` | Empirical joint-quality `P(Q \| error type)` model used during genome-blender simulation |
| `recovered_train_phred_calibration.json`, `recovered_test_phred_calibration.json` | Recovered empirical joint-quality tables from generated Skiver dumps |
| `phred_calibration_metrics.json` | Source/recovered `P(Q \| error type)` and marginal Q comparisons |
| `source_quality_calibration_model.json` | Source Q-to-error model used only as a validation target |
| `recovered_train_quality_calibration_model.json`, `recovered_test_quality_calibration_model.json` | Q-to-error calibrations refit from generated Skiver dumps for validation |
| `quality_calibration_model_metrics.json` | Source/recovered validation-only Q-to-error calibration comparisons |

### 10.2 Interpreting `recovery_metrics.json`

The benchmark compares distributions in probability space, not raw logits.
This is important because categorical logits are identifiable only up to an
additive per-context constant.

Key metrics:

| Metric | Interpretation |
|--------|----------------|
| `observed_vs_source_tv` | Difference between Skiver-observed event counts and source probabilities over the same context/base exposure |
| `retrained_vs_source_tv` | Overall recovery error from simulation, Skiver dump, and retraining |
| `observed_vs_retrained_tv` | How closely the retrained model matches the observations Skiver actually emitted |
| `*_kl` | Same comparisons as KL divergence; more sensitive to rare categories |
| `source_error_rate`, `observed_error_rate`, `retrained_error_rate` | Marginal non-match event fraction |
| `source_probs`, `observed_probs`, `retrained_probs` | Marginal probabilities for the 10 error types |

For a proper full-scale test, `observed_vs_source_tv` should be small first. If
it is not, the synthetic dataset is too small for the source error rate and
context distribution, or the Skiver dump settings are not capturing the
generated events well. Once this observation noise is small,
`retrained_vs_source_tv` and `observed_vs_retrained_tv` should decrease as
`--n-copies`, reference diversity, and training steps increase.

`vi_uncertainty_metrics.json` compares posterior standard deviations from the
source model artifact and the retrained artifact. This is only available when
the source artifact has a `variational_inference.params_stdev` section and the
benchmark is run with `--vi-steps > 0`.

`phred_calibration_metrics.json` checks the empirical joint-quality model used
for generation: `P(Q | error_type)`. It reports marginal Q overlap and
per-error-type total-variation distances. `quality_calibration_model_metrics.json`
is validation-only: it refits `P(error | Q)` from generated reads and compares it
with the source qcal curve. Qcal is not used to assign synthetic Phred scores
when `--joint-phred-calibration-json` is provided.

### 10.3 True-base masked training

Skiver context models represent substitutions as destination-base categories:
`sub_to_A`, `sub_to_C`, `sub_to_G`, and `sub_to_T`. During genome-blender
generation, one of those categories is impossible for each emitted true base.
For example, if the true base is `A`, sampling `sub_to_A` would not create a
sequencing error; it would be indistinguishable from a match. genome-blender
therefore masks the self-substitution category before normalising the model
probabilities for generation.

The training likelihood now applies the same constraint. Aggregated training
counts keep a true-base axis with five bins: `A`, `C`, `G`, `T`, and
`unknown/gap`. For A/C/G/T rows, the likelihood masks exactly the corresponding
self-substitution logit before the softmax. The `unknown/gap` bin is left
unmasked so insertion rows, where the consensus base is `-`, remain valid.

This mask is important for recovery tests. Without it, training can place
probability mass on impossible self-substitution categories because the
categorical loss sees them as ordinary unused classes. That fitted probability
mass is later removed at generation time, which makes the generation-masked
marginal error rate lower than the training-observed error rate. Masked
training keeps the fitted probability simplex aligned with the generator and
makes `source`, `Skiver-observed`, and `retrained` marginal rates directly
comparable.

The exported model artifact is unchanged: learned parameters remain context
logits over the 10 error types. The true-base axis is only part of the
aggregated count tensor and likelihood evaluation, so existing genome-blender
model loading continues to work.

### 10.4 Manual workflow

The benchmark script is just an orchestrated version of these steps. It runs
genome-blender in long-read amplicon single-end mode so each simulated read is
a complete copy of one input reference record with the Skiver error model
applied.

Create genome-blender input tables:

```bash
printf "genome_id,fasta_path,abundance\nsynthetic,%s,1.0\n" "$PWD/reference.fa" \
  > synthetic_train.genomes.csv
cp synthetic_train.genomes.csv synthetic_test.genomes.csv
```

Generate train/test reads:

```bash
conda run -n genome_blender_dev python ../genome-blender/generate_reads.py \
  --input-csv synthetic_train.genomes.csv \
  --num-reads 1000 \
  --output-prefix synthetic_train \
  --single-end --amplicon --long-read \
  --skiver-error-model context_error_models/additive_7_hq-illumina.pt \
  --skiver-phred-calibration context_error_cache/hq-illumina_train_phred_calibration.json \
  --seed 42 \
  --no-compress --no-ansi

conda run -n genome_blender_dev python ../genome-blender/generate_reads.py \
  --input-csv synthetic_test.genomes.csv \
  --num-reads 1000 \
  --output-prefix synthetic_test \
  --single-end --amplicon --long-read \
  --skiver-error-model context_error_models/additive_7_hq-illumina.pt \
  --skiver-phred-calibration context_error_cache/hq-illumina_train_phred_calibration.json \
  --seed 43 \
  --no-compress --no-ansi
```

Run Skiver dump with reference guidance:

```bash
skiver dump synthetic_train.fastq \
  -r reference.fa \
  -o dump_train \
  --base --use-all \
  -l 0 \
  -k 21 -v 13 -c 1

skiver dump synthetic_test.fastq \
  -r reference.fa \
  -o dump_test \
  --base --use-all \
  -l 0 \
  -k 21 -v 13 -c 1
```

Then retrain the matching model family and compare probabilities. The benchmark
script uses the same aggregation and fitting functions as
`scripts/train_context_error_models.py`, but avoids requiring the train/test
directory layout used for platform-wide model training.

### 10.4 Practical cautions

- Use a reference long and diverse enough to expose the model's context space.
  A very short repeating sequence can make marginal recovery look good while
  leaving most contexts untested.
- Use reference-guided `skiver dump` for the first recovery tests. This isolates
  model generation and retraining from consensus-calling uncertainty.
- Keep `--use-all -l 0` in the dump step for synthetic recovery tests. Outlier
  filtering is useful for real data but can hide generated errors when the goal
  is parameter recovery.
- Match `--forward-only` between generation/dump/retraining experiments if
  strand handling is being tested.
- Low-error Illumina-like models need large synthetic datasets. If no errors are
  sampled in a small smoke test, the pipeline can still run, but recovery metrics
  are not meaningful.

---

## 11. Future model extensions: considered approaches and decisions

Four extensions to the context error model and HMM have been evaluated for
their training data suitability, modeling complexity, and added value. This
section records what was considered and what was decided, so that future
implementation work starts from a known position rather than re-deriving the
same conclusions.

---

### 11.1 Phred score context in emission probabilities

**What it captures.** The Phred score Q assigned to a base is a soft indicator
of error probability. Conditioning emission probabilities on Q lets the model
say "given a C at this position with Q=10, substitutions are 50× more likely
than at Q=35."

**Training data suitability.** Excellent. `base_observations.tsv` has `read_id`,
`read_pos`, `phred`, and encoded error information. The quality-calibration
trainer de-duplicates by `(read_id, read_pos)` and fits genome-blender-style
`P(error | Q)` models from the resulting per-base rows.

**Decided approach.** Use `scripts/fit_phred_calibration.py` to fit empirical
`P(Q | error_type)` tables from Skiver dump output, then pass that table to
genome-blender with `--skiver-phred-calibration` through the benchmark's
`--joint-phred-calibration-json` option. Use
`scripts/fit_quality_calibration_model.py` only to fit/refit qcal curves as a
validation target for `P(error | Q)`.
**Status: implemented.**

**Notes on integration into model training.** The calibration tables could
alternatively be used in the forward pass of the HMM as a likelihood weight
per observation using either the fitted `P(error | Q)` curve or a factored
quality-emission term. This has not been implemented; the current models encode
Q as a binned covariate in the emission directly (§6.2, §7.1).

---

### 11.2 HMM latent quality states across the full read

**What it captures.** Sequencing quality varies slowly along a read — an
instrument may enter a "degraded" regime for hundreds of bases before recovering.
This is beyond the scope of the within-window HMM (§6, §7) which only spans
`v` bases. A read-level HMM, modelled on PBSIM3's ERRHMM, captures these
slow-varying regimes: each state has a different error rate scaling factor, and
state transitions occur between kv-mer windows.

**Training data suitability.** Good with the new `--windows` output. Each
`windows.bin` record is `(read_id, start_index, error_type)`. After sorting by
`(read_id, start_index)`, grouping by `read_id` gives the sparse per-read error
sequence needed for gap-aware HMM forward–backward. Gaps between sampled
positions (not covered by any kv-mer at the chosen `-c`) are marginalised over
in the forward algorithm. Coverage analysis: at `c=10` and read length 150 bp
(k=15, v=15), ~12 windows per read, sufficient to estimate 2–3 latent states
reliably.

**Decided approach.** Two-stage training (see §9 for the full tradeoff
discussion):

1. **Stage 1 — Context emission model:** train P(error_type | context) from
   `base_observations.tsv` using `scripts/lib/context_error_models.py`. This is
   the existing additive/combinatorial context model pipeline.

2. **Stage 2 — HMM transition model:** train HMM latent state transitions and
   per-state error rate scaling factors from `windows.bin`. Each observation's
   "is_error" indicator (error_type > 0) feeds the forward algorithm. The
   emission probability for each state is a scaled version of the marginal error
   rate: `P(error | state_s) = scale_s × P_marginal`, where `scale_s` is a
   per-state learned scalar. This keeps the number of HMM-specific parameters
   small (S scaling factors + S×S transition matrix) while delegating the
   detailed error type distribution to the context model.

3. **Number of states:** start with S=2 (good/bad regime). Add states only if
   the log-likelihood improvement justifies the extra parameters. PBSIM3 uses
   S=2 for most platforms.

**Implementation note.** The gap-aware forward algorithm for sparse sequences:
for a gap of `g` unobserved windows between observations at positions `a` and
`b`, the transition matrix contribution is `A^g` (matrix power). For typical
gap sizes (≤20 windows), this can be computed by repeated squaring or via
eigendecomposition.

**Data volume warning.** With `--windows-reads N`, limit the reads used to cap
memory. At `c=10`, 1 M reads → ~12 M records → ~156 MB — loadable in one go.
At `c=1` with no read cap, 33 M observations → 429 MB; feasible but chunked
loading is recommended. **Status: data infrastructure implemented
(`skiver dump --windows`). Training script not yet written.**

---

### 11.3 Position along read as a covariate

**What it captures.** Error rate is not uniform along a read. For short-read
Illumina, the last 10–20 bases show a sharp increase. For ONT, the start and
end are noisier than the middle. Conditioning the emission on absolute position
lets the model assign different expected error rates to a window depending on
where in the read it falls.

**Training data suitability.** Excellent. `base_observations.tsv` has
`read_pos` (absolute 0-based position) and `dist_to_read_end`. `windows.bin`
has `start_index` which serves the same role. The `summary_read_position.csv`
output from `skiver analyze` already characterises the empirical
position–error-rate curve, confirming the effect is present and worth modelling.

**Decided approach.** Add position as an additive covariate to the context
model (§6, §7), not as a separate model. Two position features per observation:

- `from_start = start_index` (or `read_pos` for base-level models)
- `from_end = dist_to_read_end`

Represent each as a smooth function using a small B-spline basis (4–8 knots) or
a learnable embedding over binned positions (10–20 bins). Additive in log space:

```
log P(error_type | context, pos) = log P(error_type | context)
                                 + f_start(from_start)
                                 + f_end(from_end)
```

where `f_start` and `f_end` are learned scalar offsets (one per error type, or
shared across types if data is sparse at extreme positions).

**Complexity.** Low. Binned position already works: the `base_observations.tsv`
has sufficient data to fill 20 position bins even for rare error types. The
additive structure means no change to the context model's core architecture —
position is simply an extra level in the additive decomposition already present
in `context_error_models.py`.

**Caution.** Use `dist_to_read_end` rather than `from_start` as the primary
covariate if read lengths vary substantially across the dataset. The degradation
signal is anchored to the read end, so `dist_to_read_end` aligns reads of
different lengths correctly. **Status: not yet implemented.**

---

### 11.4 Strand (forward/reverse complement) as a covariate

**What it captures.** Certain damage types are strand-asymmetric. Oxidative
damage (8-oxoG) causes G→T on the template strand and C→A on the coding strand,
which appear as different substitutions depending on which strand the kv-mer was
extracted from. In bidirectional mode skiver records `is_forward` per
observation, enabling strand-conditional error rates.

**Training data suitability.** Good. `is_forward` is present in all TSV outputs
and is already stored in `ValueInfo`. The SBS96 spectrum output (`summary_error_
spectrum.csv`) already computes forward and total counts separately, giving a
direct read of how much strand asymmetry exists before committing to modelling it.

**Decided approach.** Include `is_forward` as a binary covariate in the additive
context model, equivalent to training two sub-models (forward / RC) that share
all parameters except a per-(error_type, strand) offset vector. This is
preferable to fully separate models when data is limited, and degrades gracefully
to no effect when strand asymmetry is absent (the offset vector converges to
zero).

In practice: add a `strand_logits[2, 10]` parameter tensor (2 strands × 10
error types), initialised to zero. The forward pass adds `strand_logits[is_forward]`
to the context logits before softmax. With sufficient data this will learn any
asymmetric error signature automatically.

**Read pair vs. strand.** The `is_forward` field in skiver is strand direction
of the kv-mer, not the R1/R2 read pair index. For paired-end Illumina, R1 and R2
have different error profiles (R2 is typically worse) regardless of strand. True
read pair membership is not stored in `ValueInfo` and cannot be recovered from
`base_observations.tsv` alone. If R1/R2 differentiation is needed, the input
reads must be labelled before sketching (e.g., by file or by FASTQ header), and
a separate `read_pair_id` field added to `ValueInfo`. **Status: not yet
implemented. Strand covariate is straightforward; R1/R2 requires upstream
labelling.**

---

### 11.5 Priority and dependency order

| Extension | Depends on | Complexity | Value | Recommended order |
|-----------|-----------|------------|-------|-------------------|
| Phred calibration in training | Nothing | Low | Medium | Done (simulation only); integrate into model loss next |
| Position covariate | Nothing | Low | High (for ONT/PacBio) | 1st |
| Strand covariate | Nothing | Low | Medium (Illumina) | 2nd |
| Read-level HMM (latent states) | Context model trained | Medium | High | 3rd (after context model is stable) |
| Joint context+HMM training | Both above | High | Low (see §9) | Only if latent state diagnostic fails |
| R1/R2 differentiation | Upstream labelling change | Medium | Medium | Defer until paired data is a priority |

---

## 12. File index

| File | Role |
|------|------|
| `scripts/train_hmm_error_model.py` | CLI: train basic HMM from `base_observations.tsv` |
| `scripts/train_profile_hmm.py` | CLI: train profile HMM with context-dependent emissions |
| `scripts/simulate_errors.py` | Legacy/local context-model simulator and probability helper |
| `scripts/benchmark_simulated_context_model.py` | CLI: generate synthetic reads with genome-blender, dump observations, retrain, and compare recovery |
| `scripts/lib/encoding.py` | Observation encoding: error types, context indexing |
| `scripts/lib/data_loading.py` | TSV loader, stratified subsampling, tensor construction |
| `scripts/lib/profile_hmm.py` | Pyro model: factored emission distribution, training |
| `scripts/lib/validation.py` | Comparison routines against summary CSVs |
| `notebooks/hmm_error_model.ipynb` | Interactive training, data exploration, plots |
| `scripts/plot_qscore_calibration.py` | Plot empirical vs. theoretical Phred calibration |
| `scripts/plot_spectrum.py` | Plot 5×5 substitution matrix (scaled to error rate) |
| `scripts/plot_sbs96_spectrum.py` | Plot SBS96 trinucleotide substitution spectrum |
| `scripts/plot_read_position.py` | Plot error rate vs. position from read start/end |
| `docs/hmm_error_model.md` | This document |
| `docs/pbsim3_reference.md` | PBSIM3 error model design reference |
| `docs/hercules_reference.md` | Hercules profile HMM design reference |
| `docs/paper_reference.md` | Mathematical reference: Weibull survival model |
