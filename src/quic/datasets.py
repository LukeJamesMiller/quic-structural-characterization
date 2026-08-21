"""Graph populations: exhaustive censuses, named examples, and strata.

Census generation shells out to ``nauty-geng``; the known counts of
connected cubic graphs (OEIS A002851) are asserted so a broken install
or changed flag fails loudly instead of silently biasing a study.
Everything downstream identifies graphs by their position in the
deterministic ``geng`` output order and by the SHA-256 of the graph6
list, so a census regenerated anywhere is byte-comparable.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess

import networkx as nx
import numpy as np

__all__ = [
    "EXPECTED_CUBIC_COUNTS", "cubic_census", "graph6_sha256",
    "named_graphs", "degree_stratum",
]

#: Connected cubic graphs on n vertices (OEIS A002851).
EXPECTED_CUBIC_COUNTS = {4: 1, 6: 2, 8: 5, 10: 19, 12: 85, 14: 509, 16: 4060}


def cubic_census(n: int) -> tuple[list[nx.Graph], list[str]]:
    """Every connected 3-regular graph on n vertices, in geng order."""
    if shutil.which("nauty-geng") is None:
        raise RuntimeError("nauty-geng not found; install the 'nauty' package")
    out = subprocess.run(["nauty-geng", "-c", "-d3", "-D3", str(n)],
                         capture_output=True, text=True, check=True)
    g6_strings = out.stdout.split()
    graphs = [nx.from_graph6_bytes(s.encode()) for s in g6_strings]
    if n in EXPECTED_CUBIC_COUNTS:
        assert len(graphs) == EXPECTED_CUBIC_COUNTS[n], (
            f"expected {EXPECTED_CUBIC_COUNTS[n]} cubic graphs at n={n}, "
            f"got {len(graphs)}")
    return graphs, g6_strings


def graph6_sha256(g6_strings) -> str:
    """Order-sensitive digest of a graph6 list: the census fingerprint."""
    return hashlib.sha256("|".join(g6_strings).encode()).hexdigest()


def named_graphs() -> dict[str, nx.Graph]:
    """The small pedagogical graphs used across the notebooks."""
    prism = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                      (0, 3), (1, 4), (2, 5)])
    return {
        "path_6": nx.path_graph(6),
        "cycle_6": nx.cycle_graph(6),
        "prism": prism,
        "K33": nx.complete_bipartite_graph(3, 3),
        "K4": nx.complete_graph(4),
        "petersen": nx.petersen_graph(),
    }


def degree_stratum(degree_sequence, count: int, seed: int,
                   max_attempts_per_graph: int = 4000) -> list[nx.Graph]:
    """Distinct connected simple graphs with an exact degree sequence.

    Deterministic under ``seed``: pairing-model sampling with rejection
    of multi-edges, self-loops, disconnected results, and isomorphic
    duplicates of already-accepted graphs (certificate: sorted trace
    tuple + graph6 canonical form via nx.weisfeiler_lehman would be
    overkill; exact duplicates are rejected by canonical graph6).
    """
    rng = np.random.default_rng(seed)
    degree_sequence = list(map(int, degree_sequence))
    assert sum(degree_sequence) % 2 == 0
    n = len(degree_sequence)
    stubs_template = np.repeat(np.arange(n), degree_sequence)
    graphs: list[nx.Graph] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(graphs) < count:
        attempts += 1
        if attempts > max_attempts_per_graph * count:
            raise RuntimeError(
                f"sampled only {len(graphs)}/{count} graphs for degree "
                f"sequence {sorted(degree_sequence)}")
        stubs = stubs_template.copy()
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for j in range(0, len(stubs), 2):
            u, v = int(stubs[j]), int(stubs[j + 1])
            if u == v or (min(u, v), max(u, v)) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        if not valid:
            continue
        graph = nx.Graph(sorted(edges))
        graph.add_nodes_from(range(n))
        if not nx.is_connected(graph):
            continue
        cert = _canonical_certificate(graph)
        if cert in seen:
            continue
        seen.add(cert)
        graphs.append(graph)
    return graphs


def _canonical_certificate(graph: nx.Graph) -> bytes:
    """Isomorphism-invariant certificate (WL hash + sorted trace tuple).

    Collision-resistant enough for duplicate rejection in modest strata;
    a WL-hash collision between non-isomorphic sampled graphs would only
    make the stratum slightly smaller than requested, never wrong.
    """
    import warnings

    from .graph_features import trace_tuple
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # nx 3.5 hash-change notice
        wl = nx.weisfeiler_lehman_graph_hash(graph, iterations=4)
    tt = trace_tuple(nx.to_numpy_array(graph))
    return (wl + "|" + ",".join(map(str, tt))).encode()
