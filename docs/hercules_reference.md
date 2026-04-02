# Hercules: Profile HMM Error Correction Reference

Reference for Firtina, Bar-Joseph & Alkan (2018), "Hercules: a profile HMM-based
hybrid error correction algorithm for long reads", Nucleic Acids Research
46(21):e125.

Paper: https://academic.oup.com/nar/article/46/21/e125/5075030
Code: https://github.com/BilkentCompGen/Hercules

---

## 1. Overview

Hercules is a **hybrid error correction** tool that uses a profile HMM to
correct long reads (PacBio/ONT) using aligned short reads (Illumina). It is
not a simulator — it does not generate reads — but its profile HMM structure
is instructive for error model design.

Key differences from PBSIM3:
- Per-read model (one profile HMM per long read, not per accuracy level)
- Match/insert/delete topology (not latent error-propensity states)
- Trained via Forward-Backward on aligned short reads
- Used for correction (Viterbi decoding), not simulation

---

## 2. Profile HMM structure

### 2.1 States per position

For each position t in the long read:

- **1 match state** M_t: emits the base at position t
- **l insertion states** I_t^1 ... I_t^l (default l=3): model consecutive
  insertions without self-loops
- **Deletion transitions** (not states): skip from M_t to M_{t+x} directly

This differs from the standard profile HMM (which has self-looping insertion
states) by using **multiple insertion states in sequence** instead. This
avoids the geometric distribution over insertion lengths that self-loops
impose, allowing more flexible indel length distributions.

### 2.2 Transition probabilities

From a match state M_t, three types of transitions:

| Transition | Target | Probability |
|-----------|--------|-------------|
| Match | M_{t+1} | alpha_M (default 0.75) |
| Insertion | I_t^1 | alpha_I (default 0.20) |
| Deletion of x bases | M_{t+1+x} | alpha_D^x |

Between insertion states:
- I_t^j → I_t^{j+1}: probability alpha_I
- I_t^l (last) → M_{t+1}: probability alpha_M + alpha_I (absorbs leftover)

Deletion probability formula:
```
alpha_D^x = f^(k-x) × alpha_del / Σ_{j=0}^{k-1} f^j
```
where alpha_del = 1 − alpha_M − alpha_I (default 0.05), f is a distribution
factor (default 2.5), and k is the maximum deletion length (default 10).
Shorter deletions are more probable (geometric decay).

### 2.3 Emission distributions

**Match states**: emit {A, C, G, T} with:
- P(observed base) = beta (default 0.97)
- P(each other base) = delta = (1 − beta) / 3 ≈ 0.01

**Insertion states**: emit {A, C, G, T} with:
- P(X) = 0 if X is the most likely base at M_{t+1} (the next match state)
- P(each remaining base) = 1/3

This design biases insertions to occur at the **end of homopolymer runs**
rather than the beginning.

**Deletion transitions**: no emission (silent).

---

## 3. Training via Forward-Backward

### 3.1 Input

Aligned short reads to the long read. Only the **start position** of each
alignment is used — CIGAR strings are ignored. This reduces dependency on
aligner artefacts.

### 3.2 Procedure

For each long read:
1. Construct the full profile HMM G(V, E)
2. For each aligned short read s at position q:
   - Extract subgraph G_s covering positions [q-1, q+m+r) where m is the
     short read length and r = ceil(m/3)
   - Run Forward-Backward on G_s to compute posterior state probabilities
3. Average posteriors across overlapping subgraphs
4. Regions with no short read coverage retain prior probabilities

### 3.3 Consensus via Viterbi

After training, Viterbi decoding finds the highest-probability path through
the profile HMM. The sequence of emissions along this path is the corrected
long read.

---

## 4. Homopolymer handling

### 4.1 Run-length encoding preprocessing

Before building the profile HMM, long reads are compressed using run-length
encoding: `AAACTGGGAC → ACTGAC`. This collapses homopolymer errors into
single positions, reducing the number of states.

### 4.2 Insertion state design

The constraint that insertion states cannot emit the next match-state base
ensures that homopolymer expansions are treated as insertions at the
**boundary** of the run, not internally.

---

## 5. Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| beta | 0.97 | Match emission probability for the observed base |
| alpha_M | 0.75 | Match-to-match transition probability |
| alpha_I | 0.20 | Match-to-insertion transition probability |
| alpha_del | 0.05 | Total deletion probability (1 − alpha_M − alpha_I) |
| f | 2.5 | Deletion length distribution decay factor |
| k | 10 | Maximum deletion length |
| l | 3 | Number of insertion states per position |

---

## 6. Relevance to skiver profile HMM

### What we adopt from Hercules
- The concept of position-specific error modeling (each position has its own
  emission characteristics)
- The separation of match, insertion, and deletion as distinct event types

### Where our model differs
- We do NOT build a per-read profile HMM (too expensive, and skiver works
  on aggregated (k,v)-mer statistics)
- We use **latent quality-regime states** (like PBSIM3) rather than
  explicit match/insert/delete state topology
- We add **dinucleotide context** conditioning on emissions
- We jointly model **Phred quality scores**
- We train on skiver dump output (consensus vs. observed), not aligned
  short reads vs. long reads

---

## 7. Code structure

The implementation is in a single file: `src/main.cpp`.

| Symbol | Role |
|--------|------|
| `struct HMMParameters` | Global transition/emission priors (beta, alpha_M, alpha_I, etc.) |
| `struct Node` | Profile HMM state: position, is_match flag, emission/transition methods |
| `Node::getEmissionProb()` | Returns P(base) given whether state is match or insertion |
| `Node::transitionProbFromThisNode()` | Returns transition probability to a target state |
| `struct TransitionInfoNode` | Stores forward/backward algorithm intermediate values |
| Forward-Backward | Implemented inline in main processing loop |
| Viterbi | Finds highest-probability path for consensus output |
