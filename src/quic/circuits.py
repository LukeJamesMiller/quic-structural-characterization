"""Qiskit construction of QuIC family circuits.

The circuits here are the presentation-layer twins of the fast NumPy
simulator in :mod:`quic.statevector`: same gates, same conventions, same
probabilities to machine precision (``verify_against_statevector`` checks
this explicitly).  Use this module when you want a drawable
``QuantumCircuit`` or a hardware-facing object; use ``quic.statevector``
for bulk exact computation.

The canonical QuIC circuit is::

    RX(eta * d_i / max_d) on each vertex     # degree encoder
    RZZ(gamma) on each edge                  # cut-phase entangler
    RX(beta) on each vertex                  # weak mixer
    measure all

at ``(eta, gamma, beta) = (2.875, 2.0, 0.1)`` with one repetition.
"""

from __future__ import annotations

import numpy as np

from .statevector import (
    CANONICAL_BETA, CANONICAL_ETA, CANONICAL_GAMMA, encoder_angles,
)

__all__ = ["FAMILY_SPECS", "CONTROL_SPECS", "build_circuit", "QuICCircuit",
           "verify_against_statevector"]

#: Active family arms: preparation x mixer, entangler always on.
FAMILY_SPECS = {
    "degree-x": {"prep": "degree", "mixer": "x", "include_edges": True},
    "flat-x": {"prep": "flat", "mixer": "x", "include_edges": True},
    "hadamard-x": {"prep": "hadamard", "mixer": "x", "include_edges": True},
    "flat-y": {"prep": "flat", "mixer": "y", "include_edges": True},
    "hadamard-h": {"prep": "hadamard", "mixer": "h", "include_edges": True},
}

#: Mechanism controls: remove the entangler, or remove the mixer.  Each is
#: graph-blind on a regular census and their blindness is what makes the
#: incremental-gain reading of the active arms meaningful.
CONTROL_SPECS = {
    "no-edge": {"prep": "flat", "mixer": "x", "include_edges": False},
    "no-mixer": {"prep": "flat", "mixer": None, "include_edges": True},
}


def build_circuit(
    adj_mat,
    *,
    prep: str = "degree",
    mixer: str | None = "x",
    include_edges: bool = True,
    eta: float = CANONICAL_ETA,
    gamma: float = CANONICAL_GAMMA,
    beta: float = CANONICAL_BETA,
    reps: int = 1,
    measure: bool = False,
):
    """Build the QuIC family circuit for an adjacency matrix."""
    from qiskit import QuantumCircuit  # deferred: keep import cost out of bulk paths

    adj_mat = np.asarray(adj_mat)
    n = adj_mat.shape[0]
    degrees = adj_mat.sum(axis=1)
    qc = QuantumCircuit(n)

    if prep == "hadamard":
        qc.h(range(n))
    else:
        for i, angle in enumerate(encoder_angles(degrees, prep, eta)):
            qc.rx(float(angle), i)

    edge_u, edge_v = np.nonzero(np.triu(adj_mat, k=1))
    for _ in range(reps):
        if include_edges:
            for u, v in zip(edge_u.tolist(), edge_v.tolist()):
                qc.rzz(float(gamma), u, v)
        if mixer == "x":
            qc.rx(float(beta), range(n))
        elif mixer == "y":
            qc.ry(float(beta), range(n))
        elif mixer == "h":
            qc.h(range(n))
        elif mixer is not None:
            raise ValueError(f"unknown mixer {mixer!r}")

    if measure:
        qc.measure_all()
    return qc


class QuICCircuit:
    """Canonical QuIC circuit for one graph (degree encoder, X mixer).

    Thin convenience wrapper: holds the adjacency matrix and angle
    configuration, builds the Qiskit circuit lazily, and computes exact
    probabilities through the fast simulator.
    """

    def __init__(self, adj_mat, *, eta=CANONICAL_ETA, gamma=CANONICAL_GAMMA,
                 beta=CANONICAL_BETA, reps=1):
        self.adj_mat = np.asarray(adj_mat)
        self.degrees = self.adj_mat.sum(axis=1)
        self.num_qubits = self.adj_mat.shape[0]
        self.eta, self.gamma, self.beta, self.reps = eta, gamma, beta, reps

    def circuit(self, measure: bool = True):
        return build_circuit(self.adj_mat, prep="degree", mixer="x",
                             eta=self.eta, gamma=self.gamma, beta=self.beta,
                             reps=self.reps, measure=measure)

    def probabilities(self) -> np.ndarray:
        from .statevector import circuit_probabilities
        return circuit_probabilities(self.adj_mat, prep="degree", mixer="x",
                                     eta=self.eta, gamma=self.gamma,
                                     beta=self.beta, reps=self.reps)

    def embedding(self) -> np.ndarray:
        """Sorted (non-increasing) probability vector: the QuIC feature map."""
        return np.sort(self.probabilities())[::-1]


def verify_against_statevector(adj_mat, atol: float = 1e-12, **kwargs) -> float:
    """Max |p_qiskit - p_numpy| over labeled outcomes; asserts < atol.

    The two implementations share no code beyond NumPy, so agreement is a
    strong end-to-end audit of gate conventions and bit ordering.
    """
    from qiskit.quantum_info import Statevector

    from .statevector import circuit_probabilities

    qc = build_circuit(adj_mat, measure=False, **kwargs)
    p_qiskit = Statevector.from_instruction(qc).probabilities()
    p_numpy = circuit_probabilities(adj_mat, **kwargs)
    err = float(np.max(np.abs(p_qiskit - p_numpy)))
    assert err < atol, f"qiskit/numpy disagreement {err:.3e}"
    return err
