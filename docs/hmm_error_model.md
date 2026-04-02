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

## 8. How the components relate

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

## 8. Files

| File | Role |
|------|------|
| `scripts/train_hmm_error_model.py` | CLI: train basic HMM from `base_observations.tsv` |
| `scripts/train_profile_hmm.py` | CLI: train profile HMM with context-dependent emissions |
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
