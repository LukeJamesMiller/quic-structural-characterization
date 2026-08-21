#!/usr/bin/env python3
"""Compute the QuIC embedding of one graph and print its head.

    python examples/compute_embedding.py                 # Petersen graph
    python examples/compute_embedding.py "IsP@OkWHG"     # any graph6 string
"""
import sys

import networkx as nx
import numpy as np

from quic import QuICCircuit
from quic.datasets import named_graphs

if len(sys.argv) > 1:
    graph = nx.from_graph6_bytes(sys.argv[1].encode())
    name = sys.argv[1]
else:
    graph, name = named_graphs()["petersen"], "petersen"

circuit = QuICCircuit(nx.to_numpy_array(graph))
embedding = circuit.embedding()

print(f"graph: {name}  (n={graph.number_of_nodes()}, m={graph.number_of_edges()})")
print(f"embedding dimension: {len(embedding)}  (sums to {embedding.sum():.12f})")
print("top 10 sorted probabilities:")
for rank, p in enumerate(embedding[:10], 1):
    print(f"  {rank:2d}  {p:.6e}")
print("\ncircuit:")
print(circuit.circuit(measure=False).draw(output="text", fold=100))
