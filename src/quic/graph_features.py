"""Classical graph structure: cycles, pair census, spectra, invariants.

Everything the notebooks predict *from* the quantum representation is
computed here, classically and where possible exactly over the integers.
Key design rule inherited from the research program: every counting
routine that admits an independent algebraic identity is validated
against it (trace identities for C3/C4, the cubic C6+diamond identity,
the pair-census cycle recovery), so no motif count enters an analysis
unchecked.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import numpy as np

__all__ = [
    "cycle_counts", "diamond_count", "validate_cycle_identities",
    "pair_profile", "cycles_from_pair_profile",
    "trace_tuple", "cospectral_groups", "characteristic_polynomial_int",
    "graph_invariants", "automorphism_group_order",
    "joint_degree_edge_counts", "cross_degree_edge_count",
    "closed_boundary_counts", "boundary_valuation",
]


# ----------------------------------------------------------------- cycles

def cycle_counts(graph: nx.Graph, max_len: int = 6) -> dict[int, int]:
    """Simple-cycle counts by length 3..max_len, each cycle counted once.

    Uses ``nx.simple_cycles(length_bound=...)`` (networkx >= 3.1 supports
    undirected graphs): no closed-walk corrections, no double counting.
    """
    counts = {k: 0 for k in range(3, max_len + 1)}
    for cyc in nx.simple_cycles(graph, length_bound=max_len):
        counts[len(cyc)] += 1
    return counts


def diamond_count(graph: nx.Graph) -> int:
    """Number of diamonds (K4 minus an edge): edges uv with codegree >= 2."""
    A = nx.to_numpy_array(graph).astype(np.int64)
    n = A.shape[0]
    return int(sum(
        math.comb(int(A[u] @ A[v]), 2)
        for u, v in combinations(range(n), 2) if A[u, v]
    ))


def validate_cycle_identities(graph: nx.Graph) -> dict[str, int]:
    """Check the exact trace identities that pin C3, C4, and (cubic) C6+D.

    * tr(A^3) = 6 C3                          (any graph)
    * tr(A^4) = 8 C4 + 2 sum d^2 - sum d      (any graph)
    * tr(A^6) = 12 (C6 + D) + 87 n + 6 C3 + 96 C4   (cubic graphs)

    Raises AssertionError on any violation; returns the counts used.
    """
    A = nx.to_numpy_array(graph).astype(np.int64)
    degrees = A.sum(axis=1)
    cc = cycle_counts(graph, max_len=6)
    A2 = A @ A
    A3 = A2 @ A
    assert int(np.trace(A3)) == 6 * cc[3], "triangle trace identity failed"
    assert int(np.trace(A3 @ A)) == 8 * cc[4] + 2 * int((degrees ** 2).sum()) - int(degrees.sum()), \
        "4-cycle trace identity failed"
    if set(degrees.tolist()) == {3}:
        n = A.shape[0]
        D = diamond_count(graph)
        assert int(np.trace(A3 @ A3)) == 12 * (cc[6] + D) + 87 * n + 6 * cc[3] + 96 * cc[4], \
            "cubic C6+diamond trace identity failed"
    return {**{f"C{k}": v for k, v in cc.items()}, "D_diamond": diamond_count(graph)}


# ----------------------------------------------------- pair (a, kappa) census

def pair_profile(graph: nx.Graph, maximum_codegree: int | None = None) -> np.ndarray:
    """The pair-incidence census N[a, k].

    ``N[a, k]`` counts vertex pairs with adjacency indicator ``a`` and
    codegree (number of common neighbors) ``k``.  This census is exactly
    the two-defect layer of the QuIC probability ranking (THEORY.md
    section 5), which is what makes it the natural classical target.
    """
    A = nx.to_numpy_array(graph).astype(np.int64)
    n = A.shape[0]
    if maximum_codegree is None:
        maximum_codegree = int(A.sum(axis=1).max())
    profile = np.zeros((2, maximum_codegree + 1), dtype=np.int64)
    for u, v in combinations(range(n), 2):
        profile[int(A[u, v]), int(A[u] @ A[v])] += 1
    return profile


def cycles_from_pair_profile(profile: np.ndarray) -> tuple[int, int, int]:
    """Recover (C3, C4, D_diamond) from the pair census alone.

    C3 = (1/3) sum_k k N[1,k];  C4 = (1/2) sum_{a,k} C(k,2) N[a,k];
    D  = sum_k C(k,2) N[1,k].
    """
    kmax = profile.shape[1]
    C3_times_3 = int(sum(k * profile[1, k] for k in range(kmax)))
    C4_times_2 = int(sum(math.comb(k, 2) * profile[a, k]
                         for a in (0, 1) for k in range(kmax)))
    D = int(sum(math.comb(k, 2) * profile[1, k] for k in range(kmax)))
    assert C3_times_3 % 3 == 0 and C4_times_2 % 2 == 0
    return C3_times_3 // 3, C4_times_2 // 2, D


# ------------------------------------------------- exact cospectrality

def trace_tuple(adj_mat) -> tuple[int, ...]:
    """(tr A, tr A^2, ..., tr A^n) as exact int64 walk counts.

    Two graphs are adjacency-cospectral iff their trace tuples agree
    (Newton's identities make the tuple equivalent to the characteristic
    polynomial), so grouping by this tuple decides cospectrality exactly,
    with no floating-point tolerance anywhere.
    """
    A = np.asarray(adj_mat).astype(np.int64)
    n = A.shape[0]
    traces = []
    P = A.copy()
    traces.append(int(np.trace(P)))
    for _ in range(n - 1):
        P = P @ A
        traces.append(int(np.trace(P)))
    return tuple(traces)


def cospectral_groups(adj_mats) -> list[list[int]]:
    """Indices of exactly-cospectral groups (size >= 2) in a list of graphs."""
    groups = defaultdict(list)
    for index, A in enumerate(adj_mats):
        groups[trace_tuple(A)].append(index)
    return sorted(sorted(v) for v in groups.values() if len(v) > 1)


def characteristic_polynomial_int(adj_mat) -> tuple[int, ...]:
    """Integer coefficients of det(tI - A) via Faddeev-LeVerrier.

    Exact integer arithmetic (Python ints), so equality of characteristic
    polynomials is decided with no tolerance.  Coefficients are returned
    from t^n down to t^0.
    """
    A = np.asarray(adj_mat).astype(object)
    n = A.shape[0]
    coeffs = [1]
    M = np.zeros_like(A)
    I = np.eye(n, dtype=object)
    for k in range(1, n + 1):
        M = A @ M + coeffs[-1] * I
        AM = A @ M
        c = -sum(AM[i, i] for i in range(n)) // k
        coeffs.append(c)
    return tuple(int(c) for c in coeffs)


# ------------------------------------------------------ global invariants

def automorphism_group_order(graph: nx.Graph) -> int:
    """|Aut(G)| by explicit enumeration of self-isomorphisms."""
    matcher = nx.isomorphism.GraphMatcher(graph, graph)
    return sum(1 for _ in matcher.isomorphisms_iter())


def graph_invariants(graph: nx.Graph, max_cycle_len: int = 7) -> dict:
    """The invariant panel used in the cospectral audit."""
    cc = cycle_counts(graph, max_len=max_cycle_len)
    spl = dict(nx.all_pairs_shortest_path_length(graph))
    dists = [d for u in spl for d in spl[u].values()]
    ecc = {u: max(spl[u].values()) for u in spl}
    return {
        **{f"C{k}": v for k, v in cc.items()},
        "girth": nx.girth(graph),
        "diameter": max(ecc.values()),
        "radius": min(ecc.values()),
        "wiener": sum(dists) // 2,
        "node_connectivity": nx.node_connectivity(graph),
        "edge_connectivity": nx.edge_connectivity(graph),
        "matching_number": len(nx.max_weight_matching(graph, maxcardinality=True)),
        "clique_number": max(len(c) for c in nx.find_cliques(graph)),
        "aut_order": automorphism_group_order(graph),
    }


# ---------------------------------------------- degree mixing (irregular)

def joint_degree_edge_counts(graph: nx.Graph) -> dict[tuple[int, int], int]:
    """Edge counts by sorted endpoint-degree pair (d_u, d_v)."""
    degrees = dict(graph.degree())
    counts: dict[tuple[int, int], int] = {}
    for u, v in graph.edges():
        key = tuple(sorted((degrees[u], degrees[v])))
        counts[key] = counts.get(key, 0) + 1
    return counts


def cross_degree_edge_count(graph: nx.Graph) -> int:
    """Edges whose endpoints have different degrees.

    On a two-class (bimodal) degree sequence this single number carries
    the entire joint-degree mixing freedom: with the degree sequence
    fixed, all of m_{aa}, m_{ab}, m_{bb} are affine in it.
    """
    degrees = dict(graph.degree())
    return sum(1 for u, v in graph.edges() if degrees[u] != degrees[v])


# ---------------------------------- boundary polynomial (small graphs only)

def _boundary_of_edge_mask(edge_mask: int, edges) -> int:
    boundary = 0
    for edge_index, (u, v) in enumerate(edges):
        if (edge_mask >> edge_index) & 1:
            boundary ^= (1 << u) ^ (1 << v)
    return boundary


def closed_boundary_counts(graph: nx.Graph, max_edges: int = 20) -> Counter:
    """Coefficients of Z_{G,emptyset}(z): even subgraphs counted by size.

    Enumerates all 2^m edge subsets, so it is restricted to small graphs
    (m <= max_edges).  On cubic graphs the coefficients at z^3..z^5 are
    exactly C3, C4, C5, and at z^6 they equal C6 + C(C3,2) - D.
    """
    edges = sorted(tuple(sorted(e)) for e in graph.edges())
    m = len(edges)
    if m > max_edges:
        raise ValueError(f"{m} edges: exhaustive boundary enumeration refused")
    counts: Counter = Counter()
    for edge_mask in range(1 << m):
        if _boundary_of_edge_mask(edge_mask, edges) == 0:
            counts[edge_mask.bit_count()] += 1
    return counts


def boundary_valuation(graph: nx.Graph, u: int, v: int, max_edges: int = 20) -> int:
    """Valuation of the open sector Z_{G,{u,v}}: equals graph distance d(u,v)."""
    edges = sorted(tuple(sorted(e)) for e in graph.edges())
    m = len(edges)
    if m > max_edges:
        raise ValueError(f"{m} edges: exhaustive boundary enumeration refused")
    target = (1 << u) ^ (1 << v)
    return min(edge_mask.bit_count() for edge_mask in range(1 << m)
               if _boundary_of_edge_mask(edge_mask, edges) == target)
