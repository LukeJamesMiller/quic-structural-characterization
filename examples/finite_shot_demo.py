#!/usr/bin/env python3
"""Minimal finite-shot readout comparison on a small bimodal stratum.

Reproduces the shape of notebook 05 in about a minute at reduced scale:
40 graphs, three readouts, three budgets.

    python examples/finite_shot_demo.py
"""
import warnings

import networkx as nx
import numpy as np

from quic import circuit_probabilities
from quic.datasets import degree_stratum
from quic.finite_shots import recovery_curves
from quic.graph_features import cross_degree_edge_count
from quic.readouts import sector_structure

warnings.filterwarnings("ignore")

graphs = degree_stratum([2] * 7 + [4] * 7, count=40, seed=7)
adjacency = [nx.to_numpy_array(g) for g in graphs]
target = np.array([cross_degree_edge_count(g) for g in graphs], dtype=float)
structures = [sector_structure(a.sum(axis=1).astype(int)) for a in adjacency]
probs = [circuit_probabilities(a, prep="flat") for a in adjacency]

result = recovery_curves(probs, target, structures=structures,
                         shot_ladder=(1 << 12, 1 << 15, 1 << 18),
                         n_replicates=4, seed=0)

print("exact-state ceilings:",
      {k: round(v, 3) for k, v in result["ceilings"].items()})
print("\nmean held-out R^2 by shots:")
print(f"{'shots':>8} {'R0':>8} {'R1':>8} {'R2':>8}")
for shots in result["shot_ladder"]:
    row = [np.mean(result["curves"][k][shots]) for k in ("R0", "R1", "R2")]
    print(f"{shots:>8} " + " ".join(f"{v:>8.3f}" for v in row))
print("\nR2 (degree sectors) recovers with the fewest samples; "
      "R1 (Hamming pooling) trails despite a near-perfect exact ceiling.")
