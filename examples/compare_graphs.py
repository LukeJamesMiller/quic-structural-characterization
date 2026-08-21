#!/usr/bin/env python3
"""Compare two graphs through the QuIC embedding, with the honest null.

    python examples/compare_graphs.py                    # prism vs K33
    python examples/compare_graphs.py G6STRING G6STRING  # any pair
"""
import sys

import networkx as nx
import numpy as np

from quic import circuit_probabilities
from quic.datasets import named_graphs
from quic.graph_features import cycle_counts, trace_tuple
from quic.readouts import readout_R0

if len(sys.argv) == 3:
    A = nx.from_graph6_bytes(sys.argv[1].encode())
    B = nx.from_graph6_bytes(sys.argv[2].encode())
    names = sys.argv[1:3]
else:
    graphs = named_graphs()
    A, B, names = graphs["prism"], graphs["K33"], ["prism", "K33"]

adjA, adjB = nx.to_numpy_array(A), nx.to_numpy_array(B)
embA = readout_R0(circuit_probabilities(adjA))
embB = readout_R0(circuit_probabilities(adjB))
size = max(len(embA), len(embB))
embA = np.pad(embA, (0, size - len(embA)))
embB = np.pad(embB, (0, size - len(embB)))

# isomorphism null: distance between A and a relabeled copy of itself
perm = np.random.default_rng(0).permutation(adjA.shape[0])
adjA_relabeled = adjA[np.ix_(np.argsort(perm), np.argsort(perm))]
null = np.abs(readout_R0(circuit_probabilities(adjA_relabeled))
              - readout_R0(circuit_probabilities(adjA))).sum()

print(f"{names[0]}: cycles {cycle_counts(A, 5)}")
print(f"{names[1]}: cycles {cycle_counts(B, 5)}")
print(f"isomorphic: {nx.is_isomorphic(A, B)}")
print(f"cospectral (exact): {trace_tuple(adjA) == trace_tuple(adjB)}"
      if adjA.shape == adjB.shape else "cospectral: n/a (different orders)")
print(f"\nQuIC L1 separation:  {np.abs(embA - embB).sum():.6e}")
print(f"isomorphism null:    {null:.6e}")
