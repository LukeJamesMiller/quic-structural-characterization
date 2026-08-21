# The mathematics of the QuIC representation

*A technical note accompanying the notebooks. Everything stated here is
verified by executable identity checks in
[notebook 02](../notebooks/02_boundary_and_motif_formalism.ipynb) and in
`quic.selfcheck()`; equation-by-equation derivations live in the notebook.
This note is the narrative.*

## Setting

A graph $G$ on $n$ vertices with adjacency matrix $A$ is mapped to a quantum
state by a fixed-parameter circuit — no training, no ansatz search:

$$
|\psi_G\rangle \;=\;
\Big[\textstyle\prod_i R_X^{(i)}(\beta)\Big]
\Big[\textstyle\prod_{uv \in E} R_{ZZ}^{(uv)}(\gamma)\Big]
\Big[\textstyle\prod_i R_X^{(i)}(\eta\, d_i/\Delta)\Big]\,
|0\rangle^{\otimes n},
$$

with canonical angles $(\eta, \gamma, \beta) = (2.875,\, 2.0,\, 0.1)$. The
representation is the measured outcome distribution
$p_G(x) = |\langle x|\psi_G\rangle|^2$, reduced to a graph invariant by a
permutation-invariant readout — by default the globally sorted vector
$p_G^\downarrow$. Bit $i$ of the outcome $x$ records vertex $i$; spins are
$z_i(x) = (-1)^{x_i}$.

The theory answers, in order: what the entangler writes into the state, how
the mixer makes it measurable, which combinatorial structure appears at each
order of the response, where that structure sits in the sorted vector, and
why the choice of readout is a separate question from any of the above.

## 1. The entangler writes the cut function

The commuting entangler layer acts diagonally, and up to a global phase it
multiplies each basis state by a phase determined by one combinatorial
quantity — the cut size $c_G(x)$ of the vertex bipartition encoded by $x$:

$$
\prod_{uv\in E} e^{-i\gamma z_u z_v/2}
= e^{-i\gamma m/2}\, e^{i\gamma\, c_G(x)}, \qquad m = |E|.
$$

So the graph enters the circuit as a **phase potential in the cut function**.
Read thermodynamically, $e^{i\gamma c_G(x)}$ is an Ising Boltzmann weight at
imaginary coupling $K = -i\gamma/2$: the whole construction is an
imaginary-temperature Ising model on $G$, evaluated coherently.

## 2. The pre-mixer state is a boundary transform

Expanding each edge factor turns the Walsh (Fourier) coefficients of the
pre-mixer state into generating functions over edge subsets with prescribed
odd-degree boundary $\partial F$:

$$
Z_{G,T}(z) = \sum_{F \subseteq E,\ \partial F = T} z^{|F|},
\qquad
\widehat f_G(T) \propto Z_{G,T}\!\big(-i\tan(\gamma/2)\big).
$$

This is the high-temperature expansion of the Ising model above, and it
makes the encoded combinatorics explicit. The **closed sector**
($T = \varnothing$) counts even subgraphs by size; on a cubic graph these
are vertex-disjoint cycle packings, so its low-order coefficients are
exactly $C_3$, $C_4$, $C_5$, and then $C_6 + \binom{C_3}{2} - D_\diamond$ —
cycle counts, with the first composite and diamond corrections appearing at
order six. The **open sectors** ($T = \{u,v\}$) are unnormalized
spin-correlation numerators whose valuation is the graph distance:
$\operatorname{val} Z_{G,\{u,v\}} = d_G(u,v)$. Degree-weighted encoders
generalize the transform verbatim, with joint-degree mixing entering at the
first edge order — the seed of everything in notebook 05.

## 3. The weak mixer measures the cut gradient

At $\beta = 0$ the cut phase is invisible to measurement. The mixer
$H = \sum_i X_i$ interferes each outcome with its single-bit-flip
neighbors, and the first derivative of every labeled probability is exact
and local:

$$
p'_G(x; 0) = p_0(x) \sum_i \big[x_i t_i^{-1} - (1-x_i) t_i\big]
\cos\big(\gamma\, h_i(x)\big),
\qquad
h_i(x) = \sum_{j \in N(i)} z_j(x),\quad t_i = \tan(\eta_i/2).
$$

Since the discrete gradient of the cut function is
$\Delta_i c_G(x) = z_i(x) h_i(x)$, the measured first-order response is a
**cosine-filtered discrete gradient of the cut potential**. Root flips are
the probes: flipping bit $i$ interrogates the incidence structure of vertex
$i$'s neighborhood, which is why local incidence statistics — not raw
spectra — are the natural language for what becomes measurable.

## 4. Second order closes on pair incidence: triangles, four-cycles, diamonds

The scalar probe of the deformation is the purity
$M_2(\beta) = \sum_x p(x;\beta)^2$. On a $d$-regular family its first
derivative at $\beta=0$ is graph-independent; the first graph-dependent
coefficient is $M_2''(0)$, and on cubic graphs it satisfies an exact
structure theorem:

$$
M_2''(G; 0) = A_n + b_3\, C_3(G) + b_4\, C_4(G) + b_D\, D_\diamond(G).
$$

The mechanism is **pair locality**: each vertex pair contributes through its
radius-one environment, classified by adjacency $a_{ij}$ and codegree
$\kappa_{ij} = |N(i) \cap N(j)|$. Kernels $F_{a,\kappa}$ computed on small
rooted local graphs, weighted by the pair census $N_{a,\kappa}(G)$,
reproduce $M_2''$ exactly (verified exhaustively at $n = 8, 10$); the
closure to $(C_3, C_4, D_\diamond)$ rests on an exact cancellation — the
vanishing third difference of the nonadjacent kernels — with a closed-form
Laurent proof. At the canonical angles
$b_3 \approx 3.73\times 10^{-5}$ dominates $b_4, b_D \sim 4\times 10^{-8}$
by three orders of magnitude: triangles are the leading structural
coordinate, but the exact identity is never triangle-only.

**Scope, stated plainly:** this is an infinitesimal theorem. At the
operating point $\beta = 0.1$ the third-order term of the prism/$K_{3,3}$
purity difference exceeds the second-order term by a factor of eleven. The
closure identifies *which* structure enters first; it does not predict
finite-$\beta$ magnitudes.

## 5. Higher order reaches deeper incidence

The third derivative is affine in the census of rooted incidence
configurations of up to three roots. Three-root patterns see strictly more
than pairs — alternating-root configurations expose $C_5$ and $C_6$ through
first-order score histograms — but the exact third-order coefficient does
**not** close on the cycle-count basis $(C_3, C_4, C_5, C_6, D_\diamond)$
alone; the full radius-one three-root census is required. This is the
precise sense in which the representation is "cycles first, then more":
each order of the weak-mixer response adds rooted incidence structure, of
which cycle counts are the leading shadows.

## 6. Defect layers: where each structure sits in the sorted vector

Near the canonical encoder, sorted probabilities organize by **defect
number** — how many bits deviate from the dominant all-ones outcome. The
first-order score of a defect pattern depends on the graph only through two
integers $(A, B)$ of its induced neighborhood statistics; over two-defect
patterns, $(A, B) = (n - 8 + 2a + \kappa,\ 2(1-a))$, so the two-defect
score histogram *is* the pair census, containing $C_3$, $C_4$, and
$D_\diamond$ in closed form.

At the actual angles this layer picture is certified, not assumed: analytic
shell bounds prove $p_{\ell=0} > p_{\ell=1} > p_{\ell=2} > p_{\ell=3}$ for
every simple cubic graph at $n = 14, 16$, so the first
$1 + n + \binom{n}{2}$ sorted coordinates (106 and 137 respectively) are
exactly the $\le 2$-defect outcomes, with smallest margin
$\approx 1.2\times 10^{-5}$. The three-defect layer is not certified to
remain internally ordered, and the ideal $C_5$/$C_6$ score-histogram
identities are deliberately not restated as finite-$\beta$ or finite-shot
decoding claims.

## 7. Readout is a separate question from content

Between the labeled distribution and any usable feature vector stands a
quotient: labeled $\to$ degree-sector-sorted $\to$ globally sorted. Sorting
is nonexpansive in every $\ell_p$ norm, so perturbation bounds survive it —
but each quotient can *discard* structure, and which structure survives
which readout is an empirical, budget-dependent question. Two facts frame
the repository's experiments. There is no universal injectivity theorem
after Born measurement and sorting: the honest statement is a dichotomy
(for each pair, collision parameters have measure zero unless the
probability multisets coincide identically as functions of the angles). And
on irregular graphs, scalar defect counts split into degree-class
occupancy sectors that global sorting merges — untyped cycle counts pool
mathematically different signals, which is why degree-sector pooling (R2 in
notebook 05) is the readout aligned with what the circuit actually writes.

## What is deliberately not claimed

No universal injectivity; no triangle-only identity; no
second-order-explains-the-operating-point claim; no internal ordering of the
three-defect layer at finite $\beta$; no finite-shot decodability of the
ideal $C_5$/$C_6$ identities; no transfer of regular-census conclusions to
irregular graphs without degree-typed targets. Each boundary is load-bearing
somewhere in notebooks 03–05.
