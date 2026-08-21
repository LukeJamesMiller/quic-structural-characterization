# Reproducibility

Every result in this repository regenerates from an empty environment. There
are no stored intermediate artifacts to trust: notebooks and scripts
recompute censuses, embeddings, and statistics from scratch, and the curated
files in `data/` are the committed output of one deterministic script.

## Environment

Python ≥ 3.10 with `numpy`, `networkx ≥ 3.1` (undirected
`simple_cycles(length_bound=...)`), `scipy`, `scikit-learn`, `qiskit ≥ 1.0`,
and for the notebooks `matplotlib` and `pandas`:

```
pip install -e ".[notebooks]"
```

Census enumeration shells out to `nauty-geng`
(Debian/Ubuntu: `apt install nauty`; macOS: `brew install nauty`).

## The verification gate

Run first, always:

```python
import quic
quic.selfcheck()
```

This executes the package's mathematical audit: Qiskit vs the pure-NumPy
simulator on every family arm and both controls (independent
implementations, agreement ~1e-15), normalization, coordinatewise
relabeling invariance of R0/R1/R2 under full re-embedding, dual independent
R2 implementations with binomial sector cardinalities, the trace identities
for C3/C4 and the cubic C6+diamond identity, pair-census cycle recovery,
and boundary-polynomial checks. Every notebook calls it in its first code
cell.

## Determinism inventory

Graph populations are identified positionally in deterministic `nauty-geng`
output order and by order-sensitive SHA-256 fingerprints of the graph6
lists:

| population | count | sha256 of graph6 list |
|---|---|---|
| connected cubic, n=10 | 19 | `b32ebf2b90ad583789d93569948aa71be5f0e112d4fdb09cedb7fb27ef05b17c` |
| connected cubic, n=14 | 509 | `94ddc5d34ad98bfe1b584d22fa19df798f22ee31d111ec8db63c8a0fa2db9ad9` |
| connected cubic, n=16 | 4060 | `66eebea498e459377c3b90d3a0c5102a1dc320b62d720338b3ff1e78f2fcc3dc` |

Census sizes are additionally asserted against the known counts of connected
cubic graphs (OEIS A002851), so a broken nauty install fails loudly. The
bimodal stratum is drawn by `quic.datasets.degree_stratum(count=200, seed=7)`
— a seeded pairing-model sampler with isomorphism-duplicate rejection — and
its 200 graph6 strings are committed in
`data/demo_graphs/bimodal_stratum_n14.json`. All sampling in the finite-shot
experiments flows through `numpy.random.default_rng` seeded per
(budget, replicate), so recovery curves reproduce bit-for-bit.

## Regeneration

| artifact | command | runtime (laptop-scale) |
|---|---|---|
| all of `data/` | `python scripts/generate_demo_data.py` | ~1 min |
| notebook results + `figures/*.png` | run notebooks 01–05 top to bottom | ~1–2 min each |
| examples | `python examples/<name>.py` | seconds to ~1 min |

## Numerical caveats

Exact-integer paths (trace tuples, characteristic polynomials, cycle counts,
sector cardinalities) are platform-independent by construction. Floating
point results (embeddings, probe scores) can differ in trailing digits
across BLAS builds; all notebook assertions use explicit tolerances
(typically 1e-12 to 1e-14 for identities, far coarser for statistics), and
no conclusion in the repository depends on digits beyond those tolerances.
Cross-validated scores fix fold seeds; scikit-learn version changes may
still shift third decimals of probe R² without affecting any stated
ordering or significance call. The analytic head-separation margins are
evaluated in ordinary floating point; a certificate-grade evaluation would
re-run them in interval arithmetic (margins are ~1e-5 against ~1e-16
evaluation error).
