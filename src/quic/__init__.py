"""QuIC structural characterization: a phase-encoded quantum graph feature map.

The pipeline this package implements and studies::

    G  ->  phase-encoded circuit U_G  ->  p_G  ->  invariant readout  ->
           recoverable graph structure

with the canonical operating point ``(eta, gamma, beta) = (2.875, 2.0, 0.1)``,
one entangler+mixer repetition, and the globally sorted probability vector
as the default embedding.

Modules
-------
circuits        Qiskit circuit construction (family arms + mechanism controls)
statevector     exact pure-NumPy simulation and Walsh-transform utilities
readouts        R0 global sort / R1 Hamming pooling / R2 degree sectors
graph_features  cycles, pair census, exact cospectrality, invariants
finite_shots    seeded sampling, ridge probes, recovery curves
datasets        nauty censuses, named graphs, degree strata

``quic.selfcheck()`` runs the package's mathematical audit: circuit
implementations against Qiskit, readout invariances, and the exact
counting identities.  The notebooks call it before any analysis.
"""

from . import circuits, datasets, finite_shots, graph_features, readouts, statevector
from .circuits import QuICCircuit, build_circuit
from .statevector import (
    CANONICAL_BETA, CANONICAL_ETA, CANONICAL_GAMMA,
    circuit_probabilities, circuit_state,
)
from .readouts import readout_R0, readout_R1, readout_R2, sector_structure

__version__ = "0.1.0"

__all__ = [
    "circuits", "statevector", "readouts", "graph_features", "finite_shots",
    "datasets", "QuICCircuit", "build_circuit", "circuit_probabilities",
    "circuit_state", "readout_R0", "readout_R1", "readout_R2",
    "sector_structure", "CANONICAL_ETA", "CANONICAL_GAMMA", "CANONICAL_BETA",
    "selfcheck",
]


def selfcheck(verbose: bool = True) -> dict:
    """Run the package-wide mathematical audit.  Raises on any failure.

    Checks (all exact or to 1e-12):
      1. Qiskit vs pure-NumPy probabilities, every family arm and control.
      2. Probabilities sum to one; sorted embedding invariant under
         vertex relabeling (full re-embedding, not vector permutation).
      3. The two independent R2 implementations agree; sector
         cardinalities are the predicted binomial products.
      4. Trace identities for C3/C4 and the cubic C6+diamond identity.
      5. Pair-census recovery of (C3, C4, D_diamond) against direct
         cycle enumeration.
      6. Boundary-polynomial closed-sector coefficients equal C3..C5 on
         cubic examples, and open-sector valuation equals graph distance.
    """
    import networkx as nx
    import numpy as np

    from .circuits import CONTROL_SPECS, FAMILY_SPECS, verify_against_statevector
    from .datasets import named_graphs
    from .graph_features import (
        boundary_valuation, closed_boundary_counts, cycle_counts,
        cycles_from_pair_profile, pair_profile, validate_cycle_identities,
    )
    from .readouts import validate_R2_implementations, validate_relabeling_invariance
    from .statevector import circuit_probabilities

    report: dict = {}
    graphs = named_graphs()
    prism, k33, petersen = graphs["prism"], graphs["K33"], graphs["petersen"]

    # 1. implementation agreement, all arms, on an irregular graph too
    irregular = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    for name, spec in {**FAMILY_SPECS, **CONTROL_SPECS}.items():
        for graph in (prism, irregular):
            err = verify_against_statevector(nx.to_numpy_array(graph), **spec)
        report[f"qiskit_agreement[{name}]"] = err

    # 2. normalization + relabeling invariance of the full pipeline
    rng = np.random.default_rng(2026)
    for name, graph in [("prism", prism), ("irregular", irregular), ("petersen", petersen)]:
        A = nx.to_numpy_array(graph)
        p = circuit_probabilities(A)
        assert abs(p.sum() - 1.0) < 1e-12, f"probabilities of {name} do not sum to 1"
        report[f"relabeling[{name}]"] = validate_relabeling_invariance(A, rng=rng)

    # 3. dual R2 + sector cardinalities on an irregular graph
    A = nx.to_numpy_array(irregular)
    report["R2_dual"] = validate_R2_implementations(
        circuit_probabilities(A), A.sum(axis=1).astype(int))

    # 4. trace identities
    for name, graph in [("prism", prism), ("K33", k33), ("petersen", petersen)]:
        report[f"identities[{name}]"] = validate_cycle_identities(graph)

    # 5. pair census recovers C3, C4, diamonds
    for name, graph in [("prism", prism), ("K33", k33), ("petersen", petersen)]:
        C3, C4, D = cycles_from_pair_profile(pair_profile(graph))
        cc = cycle_counts(graph, max_len=4)
        assert (C3, C4) == (cc[3], cc[4]), f"pair census wrong on {name}"

    # 6. boundary polynomial: closed sectors are cycle counts on cubic
    for name, graph in [("prism", prism), ("K33", k33)]:
        counts = closed_boundary_counts(graph)
        cc = cycle_counts(graph, max_len=5)
        assert all(counts[k] == cc[k] for k in (3, 4, 5)), f"closed sectors wrong on {name}"
    assert boundary_valuation(prism, 0, 4) == nx.shortest_path_length(prism, 0, 4)

    if verbose:
        worst = max(v for k, v in report.items() if k.startswith("qiskit_agreement"))
        print(f"selfcheck passed: {len(report)} check groups, "
              f"worst qiskit/numpy disagreement {worst:.2e}")
    return report
