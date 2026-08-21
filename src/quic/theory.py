"""Exact theory of the QuIC circuit: identities, derivatives, defect layers.

This module carries the theorem-level machinery that
``notebooks/02_boundary_and_motif_formalism.ipynb`` demonstrates:

* the cut-phase identity (the entangler is a phase potential in the
  graph cut function);
* the weak-mixer response (the first probability derivative is a
  cosine-filtered discrete cut gradient);
* the exact second-order purity closure on cubic graphs,
  ``M2''(0) = A_n + b3*C3 + b4*C4 + bD*D_diamond``, with its
  coefficients computed from rooted local graphs (pair kernels);
* the defect-layer scores whose histogram over two-defect outcomes is
  the pair census;
* the finite-beta head-separation certificate at the canonical angles.

Everything here is exact or machine-precision and is asserted against
direct statevector computation in the notebook and in ``quic.selfcheck``
extensions.  Conventions follow :mod:`quic.statevector`.
"""

from __future__ import annotations

import cmath
import math
from itertools import combinations

import numpy as np

from .statevector import (
    CANONICAL_BETA, CANONICAL_ETA, CANONICAL_GAMMA, basis_bits, circuit_state,
)

__all__ = [
    "cut_size", "verify_cut_phase", "apply_H", "pre_mixer_state",
    "probability_first_response", "local_first_response",
    "purity_derivatives", "regular_first_purity_derivative",
    "pair_kernel", "pair_kernel_table", "motif_coefficients",
    "defect_statistics", "cubic_defect_score", "outcome_from_defects",
    "head_separation_margins",
]


# ------------------------------------------------------------ cut phase

def cut_size(x: int, edges) -> int:
    """Number of edges cut by the bipartition encoded in bitstring x."""
    return sum(((x >> u) & 1) != ((x >> v) & 1) for u, v in edges)


def verify_cut_phase(n: int, edges, gamma: float = CANONICAL_GAMMA) -> float:
    """Max error of  prod RZZ = e^{-i gamma m/2} e^{i gamma cut(x)}."""
    m = len(edges)
    worst = 0.0
    for x in range(1 << n):
        z = [1 - 2 * ((x >> i) & 1) for i in range(n)]
        lhs = cmath.exp(-0.5j * gamma * sum(z[u] * z[v] for u, v in edges))
        rhs = cmath.exp(-0.5j * gamma * m) * cmath.exp(1j * gamma * cut_size(x, edges))
        worst = max(worst, abs(lhs - rhs))
    return worst


# ------------------------------------------------- mixer generator action

def apply_H(state: np.ndarray, n: int) -> np.ndarray:
    """Apply H = sum_i X_i (the mixer generator) to a statevector."""
    basis = np.arange(1 << n)
    return sum(state[basis ^ (1 << i)] for i in range(n))


def pre_mixer_state(adj_mat, *, prep: str = "flat", eta: float = CANONICAL_ETA,
                    gamma: float = CANONICAL_GAMMA) -> np.ndarray:
    """State after preparation and entangler, before the mixer."""
    return circuit_state(adj_mat, prep=prep, mixer=None, eta=eta, gamma=gamma)


def probability_first_response(phi: np.ndarray, n: int) -> np.ndarray:
    """Exact d p(x)/d beta at beta = 0:  Im(conj(phi) * H phi)."""
    return np.imag(np.conjugate(phi) * apply_H(phi, n))


def local_first_response(adj_mat, *, eta: float = CANONICAL_ETA,
                         gamma: float = CANONICAL_GAMMA) -> np.ndarray:
    """The same derivative from the local formula.

    p'(x; 0) = p0(x) * sum_i [x_i / t_i - (1 - x_i) t_i] cos(gamma h_i(x)),
    with t_i = tan(eta_i / 2) and h_i(x) the neighbor spin field.  The
    cosine of the field is why the measured response is a *filtered*
    discrete gradient of the cut function.
    """
    adj_mat = np.asarray(adj_mat)
    n = adj_mat.shape[0]
    degrees = adj_mat.sum(axis=1)
    angles = eta * degrees / degrees.max()
    neighbors = [np.flatnonzero(adj_mat[i]).tolist() for i in range(n)]
    _, bits = basis_bits(n)
    p0 = np.abs(pre_mixer_state(adj_mat, prep="degree", eta=eta, gamma=gamma)) ** 2
    tangents = np.tan(angles / 2)
    result = np.zeros(1 << n)
    for x in range(1 << n):
        z = 1 - 2 * bits[x].astype(int)
        score = 0.0
        for i in range(n):
            local_weight = 1 / tangents[i] if bits[x, i] else -tangents[i]
            field = int(sum(z[j] for j in neighbors[i]))
            score += local_weight * math.cos(gamma * field)
        result[x] = p0[x] * score
    return result


def purity_derivatives(adj_mat, *, prep: str = "flat", eta: float = CANONICAL_ETA,
                       gamma: float = CANONICAL_GAMMA) -> tuple[float, float]:
    """Exact (M2'(0), M2''(0)) by direct statevector differentiation."""
    adj_mat = np.asarray(adj_mat)
    n = adj_mat.shape[0]
    phi = pre_mixer_state(adj_mat, prep=prep, eta=eta, gamma=gamma)
    H1 = apply_H(phi, n)
    H2 = apply_H(H1, n)
    p0 = np.abs(phi) ** 2
    p1 = np.imag(np.conjugate(phi) * H1)
    p2 = 0.5 * (np.abs(H1) ** 2 - np.real(np.conjugate(phi) * H2))
    return (2 * float(np.sum(p0 * p1)),
            2 * float(np.sum(p1 ** 2 + p0 * p2)))


def regular_first_purity_derivative(n: int, d: int, eta: float = CANONICAL_ETA,
                                    gamma: float = CANONICAL_GAMMA) -> float:
    """Closed form of M2'(0) on any d-regular graph: graph-independent."""
    c, s = math.cos(eta / 2), math.sin(eta / 2)
    r, u = c ** 2, s ** 2
    L = r ** 2 + u ** 2
    Q = r ** 2 * np.exp(1j * gamma) + u ** 2 * np.exp(-1j * gamma)
    return float(-2 * n * c * s * (r - u) * L ** (n - d - 1) * np.real(Q ** d))


# --------------------------------------------- pair kernels and motif closure

def _local_pair_graph(d: int, adjacent: int, common: int):
    """Adjacency matrix of the rooted local graph of a pair type (a, k)."""
    left, right, next_node = 0, 1, 2
    edges = []
    if adjacent:
        edges.append((left, right))
    for _ in range(common):
        edges.extend(((left, next_node), (right, next_node)))
        next_node += 1
    for _ in range(d - adjacent - common):
        edges.append((left, next_node)); next_node += 1
    for _ in range(d - adjacent - common):
        edges.append((right, next_node)); next_node += 1
    A = np.zeros((next_node, next_node))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    return A, next_node


def pair_kernel(d: int, adjacent: int, common: int, total_n: int, *,
                eta: float = CANONICAL_ETA, gamma: float = CANONICAL_GAMMA) -> float:
    """Contribution F_{a,k} of one pair type to M2''(0) on a d-regular graph.

    Computed on the rooted local graph of the pair (its radius-1
    environment determines everything), times the universal spectator
    factor L^(total_n - local_n).  This locality is the content of the
    pair-adjacency/codegree theorem.
    """
    if d - adjacent - common < 0:
        return 0.0
    A, local_n = _local_pair_graph(d, adjacent, common)
    phi = circuit_state(A, prep="flat", mixer=None, eta=eta, gamma=gamma)
    basis = np.arange(1 << local_n)
    left_state = phi[basis ^ 1]            # flip root 0
    right_state = phi[basis ^ 2]           # flip root 1
    double_state = phi[basis ^ 3]          # flip both
    probability = np.abs(phi) ** 2
    left_first = np.imag(np.conjugate(phi) * left_state)
    right_first = np.imag(np.conjugate(phi) * right_state)
    local_value = float(np.sum(
        4 * left_first * right_first
        + 2 * probability * (np.real(left_state * np.conjugate(right_state))
                             - np.real(np.conjugate(phi) * double_state))
    ))
    c, s = math.cos(eta / 2), math.sin(eta / 2)
    spectator = (c ** 4 + s ** 4) ** (total_n - local_n)
    return local_value * spectator


def pair_kernel_table(n: int, d: int = 3, *, eta: float = CANONICAL_ETA,
                      gamma: float = CANONICAL_GAMMA) -> dict[tuple[int, int], float]:
    return {(a, k): pair_kernel(d, a, k, n, eta=eta, gamma=gamma)
            for a in (0, 1) for k in range(d + 1) if d - a - k >= 0}


def motif_coefficients(n: int, *, eta: float = CANONICAL_ETA,
                       gamma: float = CANONICAL_GAMMA) -> dict[str, float]:
    """Coefficients in M2''(0) = A_n + b3*C3 + b4*C4 + bD*D_diamond (cubic).

    Also returns the nonadjacent third difference, whose exact vanishing
    is what closes the basis at (C3, C4, D): without it an independent
    K_{2,3}-type coordinate could survive.
    """
    F = pair_kernel_table(n, 3, eta=eta, gamma=gamma)
    v0 = F[0, 1] - F[0, 0]
    v1 = F[1, 1] - F[1, 0]
    delta0 = F[0, 2] - 2 * F[0, 1] + F[0, 0]
    delta1 = F[1, 2] - 2 * F[1, 1] + F[1, 0]
    third = F[0, 3] - 3 * F[0, 2] + 3 * F[0, 1] - F[0, 0]
    return {"b3_C3": 3 * (v1 - v0), "b4_C4": 2 * delta0,
            "bD_diamond": delta1 - delta0, "nonadjacent_third_difference": third}


# ----------------------------------------------------------- defect layers

def defect_statistics(n: int, adj_mat, defects) -> tuple[int, int]:
    """(A, B): non-defect and defect vertices with 0 or 3 defect neighbors.

    These two integers determine the first-order score of a defect
    pattern on a cubic graph; over two-defect patterns their histogram
    is exactly the pair census N[a, k] via A = n - 8 + 2a + k,
    B = 2(1 - a).
    """
    adj_mat = np.asarray(adj_mat)
    defects = set(defects)
    k = [int(sum(adj_mat[i, j] for j in defects)) for i in range(n)]
    A = sum(1 for i in range(n) if i not in defects and k[i] in (0, 3))
    B = sum(1 for i in range(n) if i in defects and k[i] in (0, 3))
    return A, B


def cubic_defect_score(n: int, defects, A: int, B: int, *,
                       eta: float = CANONICAL_ETA,
                       gamma: float = CANONICAL_GAMMA) -> float:
    """First-order score p'(x)/p0(x) of a defect pattern on a cubic graph."""
    ell = len(set(defects))
    T = math.tan(eta / 2)
    delta_gamma = math.cos(3 * gamma) - math.cos(gamma)
    return (math.cos(gamma) * ((n - ell) / T - T * ell)
            + delta_gamma * (A / T - T * B))


def outcome_from_defects(n: int, defects) -> int:
    """Bitstring with 1 everywhere except the defect vertices."""
    defects = set(defects)
    return sum(1 << i for i in range(n) if i not in defects)


# -------------------------------------- finite-beta head separation bounds

def head_separation_margins(n: int, *, eta: float = CANONICAL_ETA,
                            gamma: float = CANONICAL_GAMMA,
                            beta: float = CANONICAL_BETA) -> tuple[float, float, float]:
    """Analytic margins certifying p(0-defect) > p(1) > p(2) > p(3-defect).

    Returns (margin_01, margin_12, margin_23); all positive means the
    first 1 + n + C(n,2) sorted coordinates of every cubic graph at this
    operating point are exactly the outcomes with at most two defects.
    Floating-point evaluation of the analytic bounds (a certified
    version would use interval arithmetic; margins are ~1e-5 or larger).
    """
    ce, se = math.cos(eta / 2), math.sin(eta / 2)
    te = se / ce
    cb, sb = math.cos(beta / 2), math.sin(beta / 2)
    edge_corr = 2 * abs(math.sin(gamma))

    def shell_mass(defect_count, minimum_distance):
        total = 0.0
        for a in range(defect_count + 1):
            for b in range(n - defect_count + 1):
                dist = a + b
                if dist < minimum_distance:
                    continue
                resulting = defect_count - a + b
                total += (math.comb(defect_count, a) * math.comb(n - defect_count, b)
                          * cb ** (n - dist) * sb ** dist
                          * ce ** resulting * se ** (n - resulting))
        return total

    def all_shell_interval(defect_count):
        diagonal = cb ** n * ce ** defect_count * se ** (n - defect_count)
        off = shell_mass(defect_count, 1)
        return max(0.0, diagonal - off) ** 2, (diagonal + off) ** 2

    def flip_ratio(is_defect, defect_neighbor_count):
        field = 2 * defect_neighbor_count - 3
        if is_defect:
            return -1j * te * cmath.exp(1j * gamma * field)
        return 1j / te * cmath.exp(-1j * gamma * field)

    def second_shell_interval(defect_count, root_degrees, internal_edges_count,
                              outside_counts):
        ratios = [flip_ratio(True, deg) for deg in root_degrees]
        for neighbor_count, multiplicity in enumerate(outside_counts):
            ratios.extend([flip_ratio(False, neighbor_count)] * multiplicity)
        first_sum = sum(ratios)
        second_known = 0.5 * (first_sum ** 2 - sum(r ** 2 for r in ratios))
        if defect_count == 2:
            internal = [] if internal_edges_count == 0 else [(0, 1)]
        elif internal_edges_count == 0:
            internal = []
        elif internal_edges_count == 1:
            internal = [(0, 1)]
        elif internal_edges_count == 2:
            internal = [(1, 0), (1, 2)]
        else:
            internal = [(0, 1), (1, 2), (2, 0)]
        for a, b in internal:
            second_known += ratios[a] * ratios[b] * (cmath.exp(-2j * gamma) - 1)
        unknown_bound = edge_corr * (
            3 * defect_count - 2 * internal_edges_count
            + (1.5 * n - 3 * defect_count + internal_edges_count) / te ** 2)
        zero_mag = ce ** defect_count * se ** (n - defect_count)
        known_amp = zero_mag * cb ** (n - 2) * abs(
            cb ** 2 - 1j * sb * cb * first_sum - sb ** 2 * second_known)
        unknown_amp = zero_mag * cb ** (n - 2) * sb ** 2 * unknown_bound
        remainder = shell_mass(defect_count, 3)
        return (max(0.0, known_amp - unknown_amp - remainder),
                known_amp + unknown_amp + remainder)

    def two_defect_intervals():
        for adjacent in (0, 1):
            for common in range(4):
                q2 = common
                q1 = 6 - 2 * adjacent - 2 * common
                q0 = n - 2 - q1 - q2
                if min(q0, q1) < 0:
                    continue
                yield second_shell_interval(2, (adjacent, adjacent), adjacent,
                                            (q0, q1, q2, 0))

    def three_defect_intervals():
        rooted_types = (((0, 0, 0), 0), ((1, 1, 0), 1), ((1, 2, 1), 2),
                        ((2, 2, 2), 3))
        for root_degrees, internal_count in rooted_types:
            stubs = 9 - 2 * internal_count
            for q3 in range(4):
                for q2 in range(10):
                    q1 = stubs - 2 * q2 - 3 * q3
                    q0 = n - 3 - q1 - q2 - q3
                    if min(q0, q1) < 0:
                        continue
                    yield second_shell_interval(3, root_degrees, internal_count,
                                                (q0, q1, q2, q3))

    intervals = [all_shell_interval(ell) for ell in range(3)]
    margin_01 = intervals[0][0] - intervals[1][1]
    margin_12 = intervals[1][0] - intervals[2][1]
    two_lower = min(lo for lo, _ in two_defect_intervals())
    three_upper = max(hi for _, hi in three_defect_intervals())
    margin_23 = two_lower ** 2 - three_upper ** 2
    return margin_01, margin_12, margin_23
