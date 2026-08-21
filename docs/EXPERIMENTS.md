# Experiments: five questions, five answers, and the negative results

Every number below regenerates from scratch — census enumeration through
statistics — by running the named notebook (each finishes in under about two
minutes) or `scripts/generate_demo_data.py`, which writes the curated CSVs in
[`data/curated_results/`](../data/curated_results/). Results are reported at
three levels that are **not interchangeable**: *representational* (the
embeddings differ), *linear* (a frozen exact-state probe reads a target out),
and *statistical* (the same probe still works from a finite shot budget).

---

## Q1. What structural information organizes the embedding?

**Experiment** ([notebook 03](../notebooks/03_structural_hierarchy.ipynb)):
complete census of connected cubic graphs at $n = 14$ (509 graphs, no
sampling); Mantel-style permutation tests on pairwise embedding distances,
globally and within fixed-motif strata; held-out ridge probes per cycle
count; graph-blind ablation controls.

**Result.** A layered hierarchy $C_3 \to C_4 \to C_5$. Globally, embedding
distance tracks triangle differences (Mantel $r = 0.933$, $p < 0.001$).
With $C_3$ fixed, $C_4$ organizes every stratum ($r = 0.94$–$0.98$). With
$(C_3, C_4)$ fixed, $C_5$ still organizes most strata, unevenly
(size-weighted mean $r = 0.46$, range $0.11$–$0.93$). Linear readout decays
monotonically: held-out $R^2 = 1.000$ ($C_3$), $0.997$ ($C_4$), $0.894$
($C_5$), $0.337$ ($C_6$); shuffled-target control $\approx 0$. Both ablated
circuits (no entangler; no mixer) are graph-blind to machine precision —
the signal is entirely the noncommuting combination.

**Limitation.** Mantel correlations measure geometric organization, not
decodability; the ridge probe measures decodability under one frozen linear
protocol, not information content. All statements are specific to this
census and operating point; regularity removes the degree dimension by
construction. Cross-order transfer is not tested in this repository.

---

## Q2. Is the representation purely spectral?

**Experiment** ([notebook 04](../notebooks/04_cospectral_witness.ipynb)):
exact integer cospectrality censuses (trace tuples — no tolerance) over the
complete cubic populations at $n = 14$ (3 pairs) and $n = 16$ (41 groups,
83 graphs, one triple); non-isomorphism and identical integer characteristic
polynomials verified per group; embedding separation measured against the
isomorphism null (distance between a graph and a relabeled copy of itself).

**Result.** Every one of the 44 exact cospectral groups is separated:
minimum within-group separation $1.9\times 10^{-8}$ ($n=16$) and
$5.9\times 10^{-8}$ ($n=14$) against a null of $\sim 10^{-15}$ — seven to
ten orders of magnitude. Separation persists across the entire entangler
angle sweep $\gamma \in [0.4, 3.0]$. The embedding is not a function of the
adjacency spectrum.

**Limitation.** These are complete censuses at two small orders, and 1-WL is
trivially blind on regular graphs (all cubic graphs of equal order are
1-WL-equivalent), so no comparison against higher WL levels is made or
implied. No universal cospectral-separation claim follows.

---

## Q3. Do the non-spectral differences predict anything?

**Experiment** (notebook 04, section 7): within exact cospectral groups at
$n = 16$, predict which member has the larger invariant from the difference
of truncated embeddings ($k = 100$). Leave-one-group-out; antisymmetric
intercept-free logistic ranker, both orientations; inner group-wise CV for
regularization; exact binomial significance with the threshold printed
before the score; the adjacency spectrum runs as a chance-by-construction
control (its within-group feature differences are eigensolver noise,
$\sim 6\times 10^{-15}$).

**Result.**

| target | QuIC $k=100$ | threshold ($p<0.05$) | spectrum control |
|---|---|---|---|
| automorphism count ($\log_2$) | **14/14** ($p = 10^{-4}$) | ≥ 11/14 | 10/14 ($p = 0.09$) |
| Wiener index | **18/26** ($p = 0.038$) | ≥ 18/26 | 6/26 ($p = 0.999$) |

**Limitation.** The Wiener result sits exactly at its pre-stated threshold —
real but thin, and the power is capped by the population itself (26 varying
pairs exist at $n = 16$; exhaustive enumeration, not sampling, sets that
number). Within-class *ranking* on these censuses is the claim; prediction
beyond them is not.

---

## Q4. What changes on irregular graphs?

**Experiment** ([notebook 05](../notebooks/05_readouts_and_finite_shots.ipynb)):
200 sampled graphs with fixed bimodal degree sequence $2^{\times7}4^{\times7}$;
target = the stratum's single joint-degree mixing coordinate (cross-degree
edge count); three readouts — R0 global sort (top 500), R1 Hamming-weight
pooling (15 dims), R2 degree-sector pooling (64 dims) — under three
preparations.

**Result.** The degree dimension changes the readout question entirely, and
the answer is preparation-dependent — measured, not assumed:

| prep | R0 exact ceiling | R1 exact ceiling | R2 exact ceiling |
|---|---|---|---|
| flat | 1.000 | 1.000 | 1.000 |
| hadamard | **0.413** | 0.995 | 1.000 |
| degree | 0.988 | 1.000 | 1.000 |

Under the Hadamard preparation, global sorting discards most of the mixing
target *before sampling starts*. Degree-sector pooling is the only readout
of the three that works in every regime.

**Limitation.** One degree sequence, one target, one probe family. The
preparation-dependence table is itself the argument against transferring
any readout conclusion across arms without rerunning it.

---

## Q5. Does exact accessibility survive finite sampling?

**Experiment** (notebook 05): same population; empirical distributions
sampled at $2^{10}$–$2^{18}$ shots per graph, 6 seeded replicates, one
sample feeding all readouts (paired design); held-out ridge recovery vs the
exact-state ceilings; bootstrap CIs on paired differences.

**Result.** Not necessarily — and the readout decides. Under the flat
preparation all three readouts have exact ceilings $\approx 1.0$, yet at
$2^{14}$ shots R2 recovers $R^2 = 0.49$ while R0 is at $0.09$ and R1 at
zero; at $2^{18}$, R0 and R2 reach $\approx 0.95$ while **R1 — the easiest
readout to estimate, with the smallest total-variation error at every
budget — has climbed only to $0.23$, never reaching half its ceiling.** Its
failure is semantic (Hamming pooling sums over the degree-class split that
carries the signal), not statistical-efficiency. The paired R2−R1 advantage
is $0.7$–$0.8$ across the top of the ladder; R2−R0 peaks mid-ladder at
$+0.41$ and is honestly $-0.01$ at $2^{18}$, where both sit at ceiling.

**Limitation.** No sample-complexity bound; no extrapolation beyond
$2^{18}$; "never recovers" means at these budgets under this estimator.

---

## Negative results and limitations, collected

Reported here because they shape what the method is, not despite it.

$C_6$ is the weakest linear target on the cubic census ($R^2 = 0.337$) —
consistent with theory: on cubic graphs the spectrum pins $C_6 + D_\diamond$
through $\operatorname{tr}(A^6)$, and six-cycle structure enters only at
third order. The conditional $C_5$ geometry is heterogeneous — four strata
sit below $r = 0.15$, so "the hierarchy weakens with depth" is the accurate
summary, not "$C_5$ is organized." The Wiener ranking is significant exactly
at threshold (18/26); one flipped pair would lose significance, and that
sensitivity is stated rather than smoothed. Under the Hadamard preparation
the default global-sort readout is representationally crippled for degree
mixing (exact ceiling $0.41$) — the flagship readout of the QCE paper is not
universally the right one, which is precisely the finding that motivates
degree-sector pooling. R1's total exact-vs-statistical gap is a negative
result about a plausible readout that a smaller study would likely have
shipped. Finally, the second-order motif closure does not explain
operating-point magnitudes (the $\beta^3$ term dominates at $\beta = 0.1$
by a factor of eleven) — the theory earns its keep as structure
identification, not magnitude prediction.

Hardware execution, noise models, and larger orders are outside this
repository's scope; for hardware evaluation of the embedding see the QCE
2026 paper ([arXiv:2604.18841](https://arxiv.org/abs/2604.18841)).
