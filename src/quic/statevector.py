"""Exact statevector simulation of QuIC circuits, in pure NumPy.

This module is the numerical workhorse of the package. It reproduces the
Qiskit circuit of :mod:`quic.circuits` exactly (elementwise, to machine
precision) but runs one to two orders of magnitude faster on the census
sizes used in the notebooks, and it exposes the Walsh (Fourier) picture
that the theory in ``docs/THEORY.md`` is written in.

Conventions
-----------
Computational-basis states are indexed by integers ``x`` with bit ``i``
of ``x`` recording the state of vertex/qubit ``i`` (little-endian, the
same convention Qiskit's ``Statevector.probabilities`` uses).  Spins are
``z_i(x) = (-1)^{x_i}``.

The circuit family is::

    prep  ->  entangler (RZZ(gamma) on every edge)  ->  mixer

with preparations ``"degree"`` (RX(eta * d_i / max_d) per vertex),
``"flat"`` (RX(eta) on every vertex), ``"hadamard"``; and mixers ``"x"``
(RX(beta) on every vertex), ``"y"``, ``"h"``, or ``None``.  On a regular
graph the degree and flat preparations coincide.  The canonical QuIC
operating point is ``(eta, gamma, beta) = (2.875, 2.0, 0.1)`` with one
entangler+mixer repetition.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CANONICAL_ETA", "CANONICAL_GAMMA", "CANONICAL_BETA",
    "basis_bits", "encoder_angles", "prep_state", "apply_entangler",
    "apply_single_qubit_gate", "apply_mixer", "circuit_state",
    "circuit_probabilities", "walsh", "inverse_walsh", "purity",
]

CANONICAL_ETA = 2.875
CANONICAL_GAMMA = 2.0
CANONICAL_BETA = 0.1

_BIT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def basis_bits(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(basis, bits)`` with ``bits[x, i]`` = bit ``i`` of ``x``."""
    if n not in _BIT_CACHE:
        basis = np.arange(1 << n, dtype=np.int64)
        bits = ((basis[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)
        _BIT_CACHE[n] = (basis, bits)
    return _BIT_CACHE[n]


def encoder_angles(degrees, prep: str = "degree", eta: float = CANONICAL_ETA) -> np.ndarray | None:
    """Per-vertex RX angles for the given preparation (None for hadamard)."""
    degrees = np.asarray(degrees, dtype=float)
    if prep == "degree":
        return eta * degrees / degrees.max()
    if prep == "flat":
        return np.full(len(degrees), float(eta))
    if prep == "hadamard":
        return None
    raise ValueError(f"unknown prep {prep!r}")


def prep_state(n: int, degrees=None, prep: str = "degree", eta: float = CANONICAL_ETA) -> np.ndarray:
    """Product state after the preparation layer."""
    if prep == "hadamard":
        return np.full(1 << n, 2.0 ** (-n / 2), dtype=np.complex128)
    angles = encoder_angles(degrees, prep, eta)
    _, bits = basis_bits(n)
    state = np.ones(1 << n, dtype=np.complex128)
    for i, angle in enumerate(angles):
        state *= np.where(bits[:, i] == 0, math.cos(angle / 2), -1j * math.sin(angle / 2))
    return state


def apply_entangler(state: np.ndarray, n: int, edges, gamma: float) -> np.ndarray:
    """Apply ``prod_{uv in E} RZZ(gamma)`` = ``exp(-i gamma/2 sum z_u z_v)``."""
    _, bits = basis_bits(n)
    signs = 1 - 2 * bits.astype(np.int16)
    edge_sum = np.zeros(1 << n, dtype=np.int16)
    for u, v in edges:
        edge_sum += signs[:, u] * signs[:, v]
    return state * np.exp(-0.5j * gamma * edge_sum.astype(np.float64))


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, n: int) -> np.ndarray:
    """Apply the same 2x2 gate to every qubit."""
    result = np.asarray(state, dtype=np.complex128).copy()
    for qubit in range(n):
        result = np.einsum(
            "ab,xbk->xak", gate,
            result.reshape((-1, 2, 1 << qubit)), optimize=True,
        ).reshape(-1)
    return result


def _mixer_gate(mixer: str, beta: float) -> np.ndarray:
    c, s = math.cos(beta / 2), math.sin(beta / 2)
    if mixer == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if mixer == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if mixer == "h":
        r = 1 / math.sqrt(2)
        return np.array([[r, r], [r, -r]], dtype=np.complex128)
    raise ValueError(f"unknown mixer {mixer!r}")


def apply_mixer(state: np.ndarray, n: int, mixer: str | None, beta: float) -> np.ndarray:
    if mixer is None:
        return state
    return apply_single_qubit_gate(state, _mixer_gate(mixer, beta), n)


def circuit_state(
    adj_mat,
    *,
    prep: str = "degree",
    mixer: str | None = "x",
    include_edges: bool = True,
    eta: float = CANONICAL_ETA,
    gamma: float = CANONICAL_GAMMA,
    beta: float = CANONICAL_BETA,
    reps: int = 1,
) -> np.ndarray:
    """Exact final statevector of the QuIC family circuit."""
    adj_mat = np.asarray(adj_mat)
    n = adj_mat.shape[0]
    degrees = adj_mat.sum(axis=1)
    edge_u, edge_v = np.nonzero(np.triu(adj_mat, k=1))
    edges = list(zip(edge_u.tolist(), edge_v.tolist()))
    state = prep_state(n, degrees, prep, eta)
    for _ in range(reps):
        if include_edges:
            state = apply_entangler(state, n, edges, gamma)
        state = apply_mixer(state, n, mixer, beta)
    return state


def circuit_probabilities(adj_mat, **kwargs) -> np.ndarray:
    """Labeled outcome probabilities ``p_G(x)`` of the family circuit."""
    state = circuit_state(adj_mat, **kwargs)
    return np.abs(state) ** 2


def _fwht_raw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128).copy()
    stride = 1
    while stride < len(values):
        for start in range(0, len(values), 2 * stride):
            left = values[start:start + stride].copy()
            right = values[start + stride:start + 2 * stride].copy()
            values[start:start + stride] = left + right
            values[start + stride:start + 2 * stride] = left - right
        stride *= 2
    return values


def walsh(values: np.ndarray) -> np.ndarray:
    """Normalized Walsh transform ``2^{-n} sum_x chi_T(x) h(x)``."""
    return _fwht_raw(values) / len(values)


def inverse_walsh(coefficients: np.ndarray) -> np.ndarray:
    return _fwht_raw(coefficients)


def purity(probabilities: np.ndarray) -> float:
    """Second moment ``M_2 = sum_x p(x)^2`` of a probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    return float(probabilities @ probabilities)
