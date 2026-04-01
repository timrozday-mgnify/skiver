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

## 6. HMM motivation

Skiver's core output is a scalar per-base error rate (estimated via a Weibull
survival model — see `docs/paper_reference.md`). This is useful for comparing
libraries or calibrating downstream tools, but it discards structure that may
be present *within* a read:

- Error probability may vary with **position within the value window** (e.g.
  the last base of a k-mer is harder to sequence than the first).
- Errors may cluster — a "high-error" stretch followed by a "low-error"
  stretch — rather than occurring independently at a fixed rate.
- The **Phred quality score** reported by the instrument may be miscalibrated;
  different quality regimes may have genuinely different underlying error rates.

An HMM can capture all of this by modelling each position in a value-window
sequence as an emission from a latent state, where transitions between states
represent shifts between error-propensity regimes.

---

## 2. Data source: `skiver dump --base`

The HMM is trained on `{prefix}.base_observations.tsv`, produced by
`skiver dump --base`. Each row is one base position within one (k,v)-mer value
occurrence. Rows sharing an `obs_id` form a sequence of length `v` — the value
length — aligned to the consensus value for that key.

Each row contains:

| Field | Meaning |
|-------|---------|
| `true_base` | Consensus base at position `t` (ACGT or `-` for insertion) |
| `obs_base` | Observed base at position `t` (ACGT or `-` for deletion) |
| `phred` | Integer Phred quality score (qual byte − 33); −1 if unavailable |
| `t` | 1-based position within the value window |
| `passes_filter` | Whether the key passed the per-key outlier filter |

Only observations with `passes_filter=true` are used by default, excluding
keys whose hazard rates are statistical outliers (likely contamination or
repeat elements).

---

## 3. Observation encoding

Each base position is encoded as a single integer category:

```
category = true_base_idx × (NUM_BASES × NUM_PHRED_BINS)
         + obs_base_idx  × NUM_PHRED_BINS
         + phred_bin
```

where:

- `NUM_BASES = 5` — A, C, G, T, gap (`-`)
- `NUM_PHRED_BINS = 8` — Phred scores binned into 5-unit intervals:
  Q0–4, Q5–9, Q10–14, Q15–19, Q20–24, Q25–29, Q30–34, Q35+
- `NUM_OBS_CATEGORIES = 5 × 5 × 8 = 200`

This encoding captures three distinct pieces of information simultaneously:

1. **Whether an error occurred** — `true_base != obs_base`
2. **The type of error** — which substitution or indel
3. **The instrument's confidence** — Phred bin

A flat categorical distribution over 200 categories can therefore represent
joint distributions over error type and quality, without requiring independence
assumptions between these dimensions.

---

## 4. HMM structure

### 4.1 States

The model has `S` hidden states (default `S = 3`). States are **latent** — they
are never observed directly. The intention is that different states capture
different error-propensity regimes, such as:

- A "high-quality" state where most emissions are high-Phred matches
- A "low-quality" or "error-prone" state with more mismatches and lower Phred
- A transitional or "context-sensitive" state capturing position-dependent effects

The model does not prescribe what each state means; the semantics emerge from
training.

### 4.2 Parameters

Three parameter tensors are learned:

| Parameter | Shape | Meaning |
|-----------|-------|---------|
| `initial_logits` | `[S]` | Log-odds of starting in each state |
| `transition_logits` | `[S, S]` | Log-odds of transitioning from state `i` to state `j` |
| `emission_logits` | `[S, 200]` | Log-odds of emitting each of the 200 categories from state `s` |

Probabilities are obtained via softmax:
- `π_s = softmax(initial_logits)_s`
- `A_{ij} = softmax(transition_logits[i])_j`
- `B_{s,c} = softmax(emission_logits[s])_c`

### 4.3 Sequence model

Each value-window occurrence is an observation sequence `x = (x_1, …, x_v)`,
where `x_t ∈ {0, …, 199}` is the encoded category at position `t`. The joint
probability under the HMM is:

```
p(x) = Σ_{z_1,…,z_v} π_{z_1} · B_{z_1, x_1} · Π_{t=2}^{v} A_{z_{t-1}, z_t} · B_{z_t, x_t}
```

This is computed efficiently by Pyro's `DiscreteHMM` distribution using the
forward algorithm, which runs in O(v · S²) per sequence.

---

## 5. Training: MAP estimation via SVI

### 5.1 Objective

Training maximises the marginal log-likelihood summed over all sequences:

```
L(θ) = Σ_n log p(x^(n) ; θ)
```

This is equivalent to MAP estimation (Baum-Welch / EM) since all parameters
are `pyro.param` and the guide is empty (no latent samples, no variational
approximation).

Pyro's `Trace_ELBO` with an empty guide reduces to the exact log-likelihood
evaluated by the forward algorithm, so each SVI step is a gradient step on the
exact objective — not a lower bound.

### 5.2 Optimiser

Adam with learning rate 0.005 (default). SVI runs for 1000 steps by default.
Loss should decrease monotonically; plateau indicates convergence.

### 5.3 Symmetry and initialisation

The primary training difficulty is the **extreme class imbalance**: the true
per-base error rate is ~0.07%, meaning roughly 1 in 1500 categories is an
error. If all 200 emission logits start at zero (uniform), every state begins
predicting ~60% error rate, and the gradient landscape is flat in the region
of interest — the optimiser cannot find the correct solution in a reasonable
number of steps.

The fix is **informative initialisation**:

- **Match categories** (`true_base == obs_base`): logit = phred_bin index
  (0–7). This encodes the prior that high-quality matches are the most common
  observations, and higher Phred bins should be more probable.
- **Mismatch categories** (`true_base != obs_base`): logit = −8 + state_index.
  The slight per-state offset breaks the symmetry that would otherwise cause
  all states to learn identical distributions.

This initialisation places each state near a ~0.03% error rate at the start of
training, close enough to the true ~0.07% that gradient descent converges
reliably.

The **transition matrix** is initialised as `eye(S) × 3.0` — a sticky prior
that favours self-transitions. This encourages states to represent persistent
regimes (long runs of similar behaviour) rather than rapid switching.

---

## 6. Interpreting the output

### 6.1 Per-state error rate

```
error_rate(s) = Σ_{c : true_base(c) ≠ obs_base(c)} B_{s,c}
```

States will typically differ more in their **Phred-bin distributions** than
in their raw error rates, because quality score variation accounts for most of
the between-position variance. A state with concentrated Q35+ mass represents
high-confidence sequencing; a state with mass spread across Q10–Q24 represents
lower-confidence positions.

### 6.2 Mean run length

The self-transition probability `A_{ss}` implies an expected run length (number
of consecutive positions in state `s`) of:

```
E[run length | state s] = 1 / (1 − A_{ss})
```

For a sticky state with `A_{ss} = 0.99`, the mean run length is 100 bases —
longer than a typical value window, so it rarely transitions within a single
sequence. States with low self-transition probabilities are transient and
capture brief positional effects.

### 6.3 Relationship to Weibull model

The Weibull model (see `docs/paper_reference.md`) estimates a *marginal*
per-base error rate averaged over all positions and all reads. The HMM
generalises this in two ways:

1. It models **within-sequence structure** — error propensity can change
   along the value window rather than being constant.
2. It models **quality-conditional error rates** — states capture different
   (quality, error) joint distributions, making the model sensitive to
   instrument calibration.

The overall marginal error rate implied by the HMM is:

```
error_rate = Σ_s π_s · error_rate(s)
```

which should approximate the Weibull-estimated per-base error rate when both
models are fitted to the same data.

---

## 7. Limitations

**Fixed value length.** The model currently assumes all sequences have the
same length `v`. If multiple prefixes with different `-v` settings are
combined, sequences will be padded, and the padded positions (encoded as
category 0, i.e. `A→A at Q0–4`) will be treated as real observations. Always
use prefixes generated with the same `-k`/`-v` parameters.

**No positional covariates.** Position `t` within the window is not explicitly
modelled — it influences the hidden state only indirectly via the Markov
transition structure. An extension would condition emissions on `t` directly.

**No read-pair information.** `ValueInfo` does not store read-pair membership.
R1 and R2 have different error profiles (R2 typically degrades faster), so
training separate models per read pair (or including read pair as a covariate)
is preferable.

**No strand-specific effects.** Is-forward (`is_forward`) is available in the
dump output but not used as a covariate. Strand-specific error patterns (e.g.
oxidative damage) would require a conditional model.

**MAP not Bayesian.** SVI with an empty guide performs point estimation.
Posterior uncertainty over parameters is not quantified. For small datasets,
a Dirichlet prior on emission and transition rows would regularise the model;
this can be added by placing `pyro.sample` statements with Dirichlet priors in
the model and using an `AutoDelta` or `AutoDiagonalNormal` guide.

---

## 8. Files

| File | Role |
|------|------|
| `scripts/train_hmm_error_model.py` | CLI training script |
| `notebooks/hmm_error_model.ipynb` | Interactive training with data exploration and plots |
| `docs/hmm_error_model.md` | This document |
| `docs/paper_reference.md` | Mathematical reference for the underlying Weibull survival model |
