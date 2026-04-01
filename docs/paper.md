# Skiver: Alignment-free Estimation of Sequencing Error Rates and Spectra using (k, v)-mer Sketches

Zhenhao Gu, Puru Sharma, Limsoon Wong, and Niranjan Nagarajan

bioRxiv preprint doi: 10.64898/2026.02.12.705514; February 13, 2026.

---

## Abstract

**Background.** Quality control of sequencing datasets is an important first step in
numerous bioinformatics pipelines such as mapping, variant calling, and assembly.
Existing methods typically rely on alignment results or quality scores. However, the
reference genome is not always available for mapping, and uncalibrated quality scores
may yield biased estimates of error rates.

**Results.** We present *skiver*, a reference-free and alignment-free framework that
estimates sequencing errors using (k, v)-mer sketches. By identifying the consensus
through the sketched (k, v)-mers, skiver estimates survival and hazard rates that
capture positional information of sequencing errors. Across simulated and real datasets
from various sequencing platforms, skiver accurately recovers error rates and spectra.
It also reliably handles complex datasets containing multiple strains, alleles, and
repetitive regions through an outlier filtering strategy. Skiver is computationally
efficient and provides a lightweight solution for error profiling in high-throughput
sequencing.

**Availability and Implementation.** https://github.com/GZHoffie/skiver

**Key words:** sequencing error profiling, survival analysis, alignment-free methods,
high-throughput sequencing

---

## 1. Introduction

Given a set of sequenced reads, accurately estimating the overall sequencing error rate
and the spectrum of error types (substitutions, insertions, and deletions) is a
fundamental task in computational biology. These statistics are central to routine
*quality control*. For example, they are used to decide whether a run is usable, to
compare runs across different flow cells, chemistries, or library preparations, and to
detect systematic failure modes. They also directly affect downstream inference. In
particular, error profiles influence alignment scoring and chaining heuristics in read
mapping, and serve as key priors or likelihood components in sensitive variant calling
and strain-aware analysis such as phasing. Because many pipelines implicitly assume a
particular error model, biased error-rate estimates can propagate and lead to avoidable
miscalls, unstable parameter tuning, and misleading biological conclusions.

The challenge is amplified in *metagenomic* settings. Real samples often contain a
mixture of organisms with highly uneven abundance, including low coverage genomes and
closely related coexisting strains. These properties can confound error-rate estimation
in two ways: (i) true biological variation and strain heterogeneity can be mistaken for
sequencing errors, and (ii) reference genomes may be incomplete or unavailable because
organisms are missing from databases or differ substantially from available references.
Consequently, methods that rely on a single known reference, or assume a single
homogeneous genome, can produce biased estimates precisely in the scenarios where robust
error profiling is most needed.

### Previous work

#### Reference-based methods

A common and often effective strategy is to map the reads to a reference genome and
compute error statistics from the resulting alignments (e.g., via CIGAR operations).
When a high-quality, closely matching reference is available, this approach can provide
accurate estimates. However, it has two major drawbacks. First, mapping and alignment
can be computationally expensive for large read sets and for long references, making it
costly as a routine QC step. Second, and more critically for metagenomics, suitable
references may be missing, incomplete, or diverge from the sequenced genomes because
organisms are not represented in databases or because strains differ substantially from
available references. In such cases, true biological differences are conflated with
sequencing errors, inflating apparent mismatch and indel rates and biasing the estimated
sequencing error rates upward. One possible mitigation is to assemble the reads to
obtain a sample-specific consensus and then compare the reads to this consensus, but
assembly can be computationally intensive and may be unstable for low-coverage
components and complex mixtures.

#### Reference-free methods

To avoid dependence on an external reference, many approaches estimate error rates
directly from read-derived statistics. For example, *shadow regression* and *SequencErr*
utilize overlapping (paired-end) short reads to quantify disagreements between reads.
Another prominent family of methods leverages k-mer frequency information to infer
error-related quantities without explicit alignment. While scalable, these methods are
often tailored to specific regimes (e.g., single-genome assumptions or specific
sequencing platforms) and can struggle to recover the *full error spectrum* (i.e.,
distribution of substitutions vs. insertions vs. deletions). Moreover, the severe
coverage imbalance typical of metagenomic samples introduces a key pitfall:
low-coverage genomes yield sparse and highly variable k-mer counts, making
frequency-based inference unstable and causing rare true k-mers to be difficult to
distinguish from erroneous k-mers induced by sequencing noise.

Another class of reference-free methods uses Phred quality scores, which theoretically
satisfy Q = -10 log10 Pr[error] and are used by many downstream tools. However, in
practice, quality scores can be miscalibrated and vary with sequencing technology and
library preparation. Consistent with these observations, the paper's experiments show
that uncalibrated quality scores can substantially underestimate or overestimate true
error rates. Moreover, they do not directly provide reliable estimates of the full
substitution/insertion/deletion spectrum.

### Our contribution

Skiver uses (k, v)-mer sketches for reference-free profiling of sequencing error rates
and spectra. The core idea is to construct sketches that group (k, v)-mers sharing the
same *key* and then detect structured variation patterns in the associated *values*;
these patterns provide statistical signals for substitutions and indels without
requiring alignment to a reference genome. Rather than relying on a single reference or
trusting potentially miscalibrated quality scores, we aggregate evidence across
key-sharing groups to infer error processes from observable variation events in the read
set. Across simulated and real datasets from multiple sequencing platforms, our method
yields accurate error-rate estimates while remaining robust in settings where reference
genomes are unavailable or incomplete, including metagenomic regimes with uneven
coverage and strain heterogeneity.

---

## 2. Methods

### 2.1 Problem Formulation

The definition of sequencing error rate varies across the literature. It is usually
defined as the number of errors divided by the alignment length. In this paper, we
borrow concepts from survival analysis. Let T (T >= 1) be the random variable that is
the number of bases from a random starting position until the first failure
(disagreement between the sequenced base and the true underlying base).

**Definition 1.** The *hazard rate* at the t-th base from the starting position, denoted
by h(t) := Pr[T = t | T >= t], is the conditional probability of the t-th base
disagreeing with the underlying genome given that the previous (t - 1) bases agree.

**Definition 2.** The *survival rate* at the t-th base, denoted S(t) := Pr[T > t], is
the probability of the first t bases from the point of observation being free of
sequencing errors.

Note that the survival rate S(k) = prod_{t=1}^{k} (1 - h(t)) also indicates the
probability of a k-mer in the read being free of sequencing error, and
h(1) = 1 - S(1) represents the probability of a random base being erroneous. In this
work, we aim to propose an efficient and accurate estimator for hazard rate and survival
rate using (k, v)-mer sketches, and show its ability to estimate the frequency of each
error type (substitutions, insertions, and deletions) at the same time.

### 2.2 (k, v)-mer Sketches

**Definition 3.** A (k, v)-mer is a segment of DNA of length k + v, with the first k
bases being the *key* and the last v bases being the *value*.

The structure of (k, v)-mer is similar to the idea of anchor-target k-mer pairs in
SPLASH, but different in that our key and value must be adjacent for accurate error
profiling, whereas the anchor-target pair can be separated in the reads. In this paper,
we focus on using this structure for sequencing error rate and spectra estimation.

A (k, v)-mer sketch of a set of sequenced reads is created with the following steps:

- **Step 1.** We extract all the (k, v)-mers from the reads and their reverse
  complement. Optionally, only the forward strand of the reads is used.

- **Step 2.** We subsample roughly 1/c (k, v)-mers with FracMinHash. In particular,
  given a random hash function H : Sigma^k -> (0, 1), we subsample a (k, v)-mer if
  the hash value of its key is less than 1/c (c = 1000 be default).

- **Step 3.** All the subsampled (k, v)-mers are stored in a hash table that maps a
  key to the set of associated values along with the number of times they appear in
  the read set. The most frequently appearing value is identified as the *consensus*.

Here, k is chosen to be large (k = 21 by default) such that the keys are mostly unique
in the set of sequenced genomes. The keys are used as positional identifiers in the
genomes. We then identify variation in the associated values to determine possible
sequencing errors or mutations.

It is possible to show that given a per-base error rate of epsilon, if the number of
times the keys appear in the read set
N_key = Omega((1 - epsilon)^{-2v} log v), the value with the highest count
(consensus) matches the true value from the sequenced genome with high probability
(Supplementary Note S1). In practice, we simply choose a default threshold N_key = 5,
use the keys above this coverage for subsequent tasks, and show that this is sufficient
for the evaluated datasets.

If a reference genome is provided, we also extract the set of (k, v)-mers from the
genome using the same hash function. If a key K is associated with multiple different
values in the reference genome, which indicates a repeat sequence, the key is
discarded. Otherwise, the unique value associated with the key is regarded as the
*consensus*. Since the consensus is obtained from the reference, a lower threshold for
key multiplicity is set (N_key = 1) to also allow profiling of lower coverage read
sets.

### 2.3 Estimating Hazard Rate and Survival Rate

In this work, we assume that T follows a **discrete Weibull distribution** with
parameters lambda and beta, which is often used for discrete-time survival analysis.
This assumption comes from the empirical observation that hazard rates h(t) in real
sequencing datasets decrease with t, and the survival rate S(t) fits well to the curve:

    S(t) = exp(-lambda * t^beta)

Our model differs from existing error models which typically assume a constant error
rate, essentially assuming beta = 1. In real datasets, especially Nanopore and
Illumina, the best fitted parameter beta_hat is much smaller than 1. This indicates a
decreasing hazard rate, and is often interpreted as evidence for heterogeneity in the
failure process, or a heterogeneous sequencing error rate across reads in our case.

Given a (k, v)-mer sketch of the sequenced reads, we count N_{K,t}, the number of
(k, v)-mers that have key K and match with the consensus up to the t-th base. For
example, in Figure 1.B, 4 (k, v)-mers share the key K = TTACATTGGCAG. The consensus
value is taken as the value with the highest count, in this case, AGCG. Two of the
(k, v)-mers differ from the consensus in the 14th and 15th base, respectively. We
therefore have N_{K,12} = N_{K,13} = 4, N_{K,14} = 3, N_{K,15} = 2,
N_{K,16} = 2.

This process is repeated for all keys. Assuming the hazard rate at the t-th base is
h(t), we should have N_{K,t} ~ Binomial(N_{K,t-1}, 1 - h(t)). We take the maximum
likelihood estimate:

    h_hat(t) = 1 - (sum_K N_{K,t}) / (sum_K N_{K,t-1})    for k < t <= k + v   (1)

This allows us to estimate h(t) in a small interval between k and k + v. Under the
assumption that T follows a discrete Weibull distribution, we have:

    h(t) = 1 - exp(-lambda * (t^beta - (t-1)^beta))

    => log(1 - h(t)) = -lambda * (t^beta - (t-1)^beta)
                      approx -lambda * beta * t^{beta-1}

    => log(-log(1 - h(t))) approx log(lambda * beta) + (beta - 1) * log(t)

This transformation (complementary log-log) is widely used in discrete survival
analysis. We can then perform a ridge regression with Huber loss of
log(-log(1 - h_hat(t))) vs. log(t) in the range [k + 1, k + v]. Let a and b be the
estimated slope and intercept. Then, we have:

    beta_hat = a + 1,       lambda_hat = exp(b) / beta_hat

The survival rate can then be estimated by directly plugging the estimated parameters:
S_hat(t) = exp(-lambda_hat * t^{beta_hat}), and the sequencing error rate is estimated
to be h_hat(1) = 1 - S_hat(1) = 1 - exp(-lambda_hat).

This process takes time and space that are linear in v and the number of keys in the
sketch.

### 2.4 Estimating Error Spectra

To estimate the composition of errors (types of substitutions, insertions, and
deletions), we make an additional assumption that the error composition is roughly
constant and is independent of t. In other words, for a given single base error type e
(such as substitution C -> A), the hazard rate of this type of error happening can be
expressed as:

    h_e(t) = pi_e * h(t)

where pi_e is a constant. The probability of the type of error e happening exactly once
in the value, while no other error is present, is:

    Pr[error e happens exactly once in value]
        = sum_{t=k+1}^{k+v} S(t-1) * h_e(t) * S(k+v-t)
        proportional to pi_e

given that k and v are fixed. Under this assumption, we can estimate pi_e by the
frequency of error e happening exactly once in the value.

After identifying the consensus value for each key, we find the set of all pairs
(value, edit_type), where the value can be obtained from the consensus via one edit of
edit_type (such as A -> C). An example of the neighbour set can be found in
Supplementary Table S1. If a neighbour can be reached via multiple types of edit, we
mark the edit_type to be Ambiguous.

We then count the number of times each distance-1 neighbour appears in the (k, v)-mer
sketch. In the example in Figure 1.E, the neighbours corresponding to the single base
substitutions C -> A and G -> C appear once respectively. This process is repeated for
all keys, which gives us the total number of times each edit_type has appeared. The
frequency of an edit_type is estimated to be its count divided by the sum of counts of
all the profiled error types. The Ambiguous neighbours are not counted.

There are at most 11v distance-1 neighbours given a consensus value. As a result, the
time needed to count the error frequencies is only O(v * #keys + #values), where #keys
and #values are the total number of keys and values in the (k, v)-mer sketch. This is
much more efficient than the case in which we align all values to the consensus, which
can take time O(v^2 * #values).

### 2.5 Dealing with Repeats, Multiple Alleles or Strains

In real datasets, it is common that the sequenced genomes contain long repetitive
regions and heterozygous sites in the case of the human genome, or multiple co-existing
strains of the same species in the case of metagenomic samples. In these cases, a key
can be associated with multiple values in the sequenced genome, causing overestimation
of the hazard rate.

A key observation is that if a key K is associated with multiple values that have high
counts, the estimated hazard rate h_K(t) = 1 - N_{K,t}/N_{K,t-1} of that key at one
of k < t <= k + v is going to be significantly higher. We therefore use a simple
outlier filter as shown in Algorithm 1 that filters all K that have a significantly
higher h_K(t).

**Algorithm 1: Outlier filter in hazard rate estimation**

```
Input: N_{K,t} for all keys K and k < t <= k + v.
Output: A set of keys that pass the filter.

/* Initialization of the frontier */
1  keep_key <- {};
2  foreach key K do
3    keep_key[K] <- true;
4  end

/* Exclude outliers */
5  for t <- k + 1 to k + v do
6    hazard_rates <- [];
7    foreach key K do
8      h_K(t) <- 1 - N_{K,t} / N_{K,t-1};
9      if h_K(t) > 0 then
10       hazard_rates.append(h_K(t));
11     end
12   end
13   IQR <- interquartile_range(hazard_rates);
14   foreach key K do
15     if h_K(t) > median(hazard_rates) + 3 * IQR then
16       keep_key[K] <- false;
17     end
18   end
19 end
20 return {K : keep_key[K] is true};
```

If the reference genome is provided, this filter is disabled by default as keys that
are associated with multiple values are already discarded.

---

## 3. Results

### 3.1 Baselines and Datasets

For mapping-based tools, the authors used BEST to profile the BAM output of Minimap2,
assuming that the correct references are known. They used the field `matches_per_kbp`
to infer the chance of observing a match in the read:

    epsilon_BEST+Minimap2 = 1 - matches_per_kbp / 1000

For reference-free methods, they used GenomeScope2.0 to fit the k-mer histogram output
of KMC 3.2.4, and used the Read Error Rate field in the summary file. For
quality-score-based methods, they used seqtk and the ErrQ field:

    epsilon_seqtk = 10^{-ErrQ/10}

Datasets tested include mock bacterial communities, bacterial isolates, and human reads
from Nanopore (GridION, R10.4, R9.4), PacBio (HiFi, RSII), and Illumina platforms.

### 3.2 Evaluation Metrics

**MSE_S** (survival rate MSE):

    MSE_S := (1/n) * sum_{t=1}^{n} (S(t) - S_hat(t))^2

where n = 100 (similar to the Brier score in survival estimation).

**MSE_h** (hazard rate MSE):

    MSE_h := (1/n) * sum_{t=1}^{n} (h(t) - h_hat(t))^2

where h_hat(t) = 1 - exp(-lambda_hat * (t^beta_hat - (t-1)^beta_hat)) for skiver, and
h_hat(t) = epsilon_hat, S_hat(t) = exp(-epsilon_hat * t) for other tools that only
report a single error rate.

### 3.3 Key Results

1. **Mapping-based methods are biased by incomplete references.** Using the Zymo Log
   mock community, error rates estimated from individual reference genomes are
   consistently biased upward compared to using the complete reference set. Minimap2
   assigns reads from unknown species to known genomes, inflating apparent error rates.

2. **Even modest strain-level divergence biases mapping.** Using simulated E. coli reads
   with references at varying ANI (95-100%), the estimated error rate almost doubled
   as ANI decreased to 96%.

3. **Skiver accurately estimates hazard and survival rates.** At 100x coverage with
   simulated data, the survival curve S_hat(t) closely matches ground truth across
   read identities from 90% to 100%.

4. **Skiver recovers the full error spectrum.** With a random error model (equal
   probability of substitution, insertion, deletion), skiver correctly profiles
   approximately equal frequencies across all SBS types and ~1/3 each for the three
   error categories. BEST + Minimap2 overestimates substitutions relative to indels
   due to alignment scoring bias.

5. **The outlier filter handles multi-strain mixtures.** In mixtures of E. coli K-12
   MG1655 and O157:H7 (98% ANI), skiver with filtering maintains accurate error
   estimates across all mixture proportions, while skiver without filtering shows
   elevated rates as mixture proportion approaches 0.5.

6. **Skiver generalizes across platforms.** Across Nanopore (GridION, R10.4, R9.4),
   PacBio (HiFi, RSII), and Illumina datasets, skiver achieves the lowest or
   near-lowest MSE_S. The inferred error spectra match known platform characteristics:
   Nanopore is dominated by G>A and A>G transitions; PacBio by insertions and
   deletions; Illumina by substitution errors.

7. **Weibull outperforms constant hazard.** The ablation study (Table 2) shows the full
   Weibull model (beta free) consistently achieves similar or better MSE_S compared to
   the constant-hazard variant (beta = 1), with the gap most pronounced for Nanopore
   data where errors cluster along reads.

8. **Computational efficiency.** Skiver is the fastest tool tested and uses the least
   memory (0.23 GB for the Zymo Gut Microbiome PacBio HiFi dataset), running in
   single-threaded mode.

### 3.4 Operating Requirements

Skiver produces generally reliable survival and hazard rate estimates when at least one
genome in the read set has coverage exceeding 20x, and the read error rate is below 6%.
For lower coverage or higher error rates, a reference genome is required for an accurate
estimate.

---

## 4. Discussion

The paper notes several directions for improvement:

- Performance on datasets with high error rates or low sequencing coverage could be
  enhanced through adaptive selection of parameters k and v.
- The results of skiver can be applied in existing quality control pipelines, aiding
  quality score calibration and sequencing bias detection.
- Beyond sequencing reads, the framework may extend to collections of genomes where
  the hazard rate is reinterpreted as the probability that the next genomic position
  differs due to mutation, enabling mutational spectra analysis.

---

## References

1. Allison PD. Discrete-time methods for the analysis of event histories. *Sociological methodology*, 13:61-98, 1982.
2. Bethune J, Kleppe A, Besenbacher S. A method to build extended point context models of point mutations and indels. *Nature Communications*, 13(1):7884, 2022.
3. Blanca A, Harris RS, Koslicki D, Medvedev P. The statistics of k-mers from a sequence undergoing a simple mutation process without spurious matches. *Journal of Computational Biology*, 29(2):155-168, 2022.
4. Chaung K, Baharav TZ, Henderson G, Zheludev IN, Wang PL, Salzman J. Splash: A statistical, reference-free genomic algorithm unifies biological discovery. *Cell*, 186(25):5440-5456, 2023.
5. Chen S, Zhou Y, Chen Y, Gu J. fastp: an ultra-fast all-in-one fastq preprocessor. *Bioinformatics*, 34(17):i884-i890, 2018.
6. Davis EM, Sun Y, Liu Y, Kolekar P, Shao Y, Szlachta K, et al. Sequencerr: measuring and suppressing sequencer errors in next-generation sequencing data. *Genome Biology*, 22(1):37, 2021.
7. Delahaye C, Nicolas J. Sequencing dna with nanopores: Troubles and biases. *PLoS one*, 16(10):e0257521, 2021.
8. Dohm JC, Peters P, Stralis-Pavese N, Himmelbauer H. Benchmarking of long-read correction methods. *NAR Genomics and Bioinformatics*, 2(2):lqaa037, 2020.
9. Greenfield P, Duesing K, Papanicolaou A, Bauer DC. Blue: correcting sequencing errors using consensus and context. *Bioinformatics*, 30(19):2723-2732, 2014.
10. Hera MR, Medvedev P, Koslicki D, Blanca A. Estimation of substitution and indel rates via k-mer statistics. *bioRxiv*, 2025-05, 2025.
11. Irber L, Brooks PT, Reiter T, Pierce-Ward N, Hera MR, Koslicki D, Brown CT. Lightweight compositional analysis of metagenomes with fracminhash and minimum metagenome covers. *BioRxiv*, 2022-01, 2022.
12. Kaplinski L, Mols T, Puuand T, Remm M. Docestfast and accurate estimator of human ngs sequencing depth and error rate. *Bioinformatics Advances*, 3(1):vbad084, 2023.
13. Kokot M, Dehghannasiri R, Baharav TZ, Salzman J, Deorowicz S. Scalable and unsupervised discovery from raw sequencing reads using splash2. *Nature Biotechnology*, 43(7):1084-1090, 2025.
14. Kokot M, Dlugosz M, Deorowicz S. Kmc 3: counting and manipulating k-mer statistics. *Bioinformatics*, 33(17):2759-2761, 2017.
15. Li H. Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18):3094-3100, 2018.
16. Li H. seqtk: Toolkit for processing sequences in FASTA/Q formats. https://github.com/lh3/seqtk. 2025.
17. Liao P, Satten GA, Hu YJ. Phredem: a phred-score-informed genotype-calling approach for next-generation sequencing studies. *Genetic epidemiology*, 41(5):375-387, 2017.
18. Liu D, Belyaeva A, Shafin K, Chang AC, Cook DE. Best: A tool for characterizing sequencing errors. *bioRxiv*, 2022-12, 2022.
19. Liu Y, Li Y, Chen E, Xu W, Zhang X, Zeng X, Luo X. Repeat and haplotype aware error correction in nanopore sequencing reads with dechat. *Communications Biology*, 7(1):1678, 2024.
20. McIntyre AB, Alexander N, Grigorev K, Bezdan D, Sichtig H, Chiu CY, Mason CE. Single-molecule sequencing detection of n 6-methyladenine in microbial reference materials. *Nature communications*, 10(1):579, 2019.
21. Melsted P, Halldorsson BV. Kmerstream: streaming algorithms for k-mer abundance estimation. *Bioinformatics*, 30(24):3541-3547, 2014.
22. Minoche AE, Dohm JC, Himmelbauer H. Evaluation of genomic high-throughput sequencing data generated on illumina hiseq and genome analyzer systems. *Genome biology*, 12(11):R112, 2011.
23. Nicholls SJ, Quick JC, Tang S, Loman NJ. Ultra-deep, long-read nanopore sequencing of mock microbial community standards. *Gigascience*, 8(5):giz043, 2019.
24. Parks DH, Chuvochina M, Rinke C, Mussig AJ, Chaumeil PA, Hugenholtz P. Gtdb: an open census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. *Nucleic acids research*, 50(D1):D785-D794, 2022.
25. Ranallo-Benavidez TR, Jaron KS, Schatz MC. Genomescope 2.0 and smudgeplot for reference-free profiling of polyploid genomes. *Nature communications*, 11(1):1432, 2020.
26. Ross MG, Russ C, Costello M, Hollinger A, Lennon NJ, Hegarty R, Nusbaum C, Jaffe DB. Characterizing and measuring bias in sequence data. *Genome biology*, 14(5):R51, 2013.
27. Sahlin K, Medvedev P. Error correction enables use of oxford nanopore technology for reference-free transcriptome analysis. *Nature communications*, 12(1):2, 2021.
28. Schirmer M, Ijaz UZ, D'Amore R, Hall N, Sloan WT, Quince C. Insight into biases and sequencing errors for amplicon sequencing with the illumina miseq platform. *Nucleic acids research*, 43(6):e37-e37, 2015.
29. Shafin K, Pesout T, Lorig-Roach R, Haukness M, Olsen HE, Bosworth C, et al. Efficient de novo assembly of eleven human genomes using promethion sequencing and a novel nanopore toolkit. *BioRxiv*, page 715722, 2019.
30. Shaw J, Gounot JS, Chen H, Nagarajan N, Yu YW. Floria: fast and accurate strain haplotyping in metagenomes. *Bioinformatics*, 40(Supplement_1):i30-i38, 2024.
31. Shaw J, Yu YW. Fast and robust metagenomic sequence comparison through sparse chaining with skani. *Nature Methods*, 20(11):1661-1665, 2023.
32. Suresh K, Severn C, Ghosh D. Survival prediction models: an introduction to discrete-time modeling. *BMC medical research methodology*, 22(1):207, 2022.
33. Vaupel JW, Manton KG, Stallard E. The impact of heterogeneity in individual frailty on the dynamics of mortality. *Demography*, 16(3):439-454, 1979.
34. Wang XV, Blades N, Ding J, Sultana R, Parmigiani G. Estimation of sequencing error rates in short reads. *BMC bioinformatics*, 13(1):185, 2012.
35. Wick RR. Badread: simulation of error-prone long reads. *Journal of Open Source Software*, 4(36):1316, 2019.
36. Wilm A, Kim PA, Bertrand D, Hui Ting Yeo G, Ong SH, Wong CK, et al. Lofreq: a sequence-quality aware, ultra-sensitive variant caller for uncovering cell-population heterogeneity from high-throughput sequencing datasets. *Nucleic acids research*, 40(22):11189-11201, 2012.
37. Wu H, Blanca A, Medvedev P. A k-mer-based estimator of the substitution rate between repetitive sequences. *bioRxiv*, 2025-06, 2025.
38. Yu YW, Yorukoglu D, Peng J, Berger B. Quality score compression improves genotyping accuracy. *Nature biotechnology*, 33(3):240-243, 2015.
