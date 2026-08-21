"""Measurement readouts: what survives of the labeled distribution.

The circuit fixes what information exists; the readout fixes what a user
of the representation can see.  Three permutation-invariant readouts are
studied here, in decreasing order of resolution:

R0  global sort
    The full probability vector sorted non-increasing.  Discards all
    bitstring labels.  This is the QuIC embedding of the QCE paper.

R1  Hamming-weight pooling
    Total probability mass at each Hamming weight (n+1 numbers).

R2  degree-sector pooling
    Total mass on each joint occupancy of the degree classes: outcomes
    are binned by how many excited vertices they contain in each degree
    class.  On a regular graph R2 coincides with R1; on an irregular
    graph it retains exactly the degree semantics that R1 destroys.

All three are invariant under vertex relabeling *coordinatewise* (not
merely as multisets); ``validate_relabeling_invariance`` checks this.
Two independent R2 implementations are provided and cross-checked, so a
bug in the mixed-radix indexing cannot pass silently.
"""

from __future__ import annotations

from itertools import product
from math import comb

import numpy as np

from .statevector import basis_bits

__all__ = [
    "readout_R0", "readout_R1", "sector_structure", "readout_R2",
    "readout_R2_independent", "permute_probabilities",
    "validate_relabeling_invariance", "validate_R2_implementations",
]


def readout_R0(probabilities: np.ndarray) -> np.ndarray:
    """Global sort, non-increasing: the full-resolution invariant readout."""
    return np.sort(probabilities)[::-1]


def readout_R1(probabilities: np.ndarray) -> np.ndarray:
    """Hamming-weight pooling: mass at each weight 0..n."""
    n = int(np.log2(len(probabilities)))
    _, bits = basis_bits(n)
    weights = bits.sum(axis=1).astype(np.int64)
    return np.bincount(weights, weights=probabilities, minlength=n + 1)


def sector_structure(degrees) -> dict:
    """Canonical degree-sector metadata for R2.

    Degree classes are ordered by increasing degree; sectors are joint
    occupancy tuples of the classes, in mixed-radix (C-order) layout with
    the lowest-degree class most significant.  The sector count is
    ``prod_d (|V_d| + 1)`` and sector cardinalities are products of
    binomials (checked in ``validate_R2_implementations``).
    """
    degrees = np.asarray(degrees, dtype=int)
    n = len(degrees)
    _, bits = basis_bits(n)
    unique_degrees = sorted(set(degrees.tolist()))
    class_vertex_sets = [[i for i in range(n) if degrees[i] == d] for d in unique_degrees]
    class_sizes = [len(vset) for vset in class_vertex_sets]
    occupancy = np.stack([
        bits[:, vset].sum(axis=1).astype(np.int64) if vset else np.zeros(1 << n, dtype=np.int64)
        for vset in class_vertex_sets
    ])
    sector_id = np.zeros(1 << n, dtype=np.int64)
    radix = 1
    for class_position in reversed(range(len(class_sizes))):
        sector_id += occupancy[class_position] * radix
        radix *= class_sizes[class_position] + 1
    n_sectors = int(np.prod([size + 1 for size in class_sizes]))
    canonical_tuples = list(product(*[range(size + 1) for size in class_sizes]))
    return {
        "degrees": degrees, "unique_degrees": unique_degrees,
        "class_vertex_sets": class_vertex_sets, "class_sizes": class_sizes,
        "occupancy": occupancy, "sector_id": sector_id,
        "n_sectors": n_sectors, "canonical_tuples": canonical_tuples,
    }


def readout_R2(probabilities: np.ndarray, structure: dict) -> np.ndarray:
    """Degree-sector pooling (primary implementation: mixed-radix bincount)."""
    return np.bincount(structure["sector_id"], weights=probabilities,
                       minlength=structure["n_sectors"])


def readout_R2_independent(probabilities: np.ndarray, structure: dict) -> np.ndarray:
    """Second, independent R2 (matrix occupancy -> ravel_multi_index -> add.at)."""
    n = len(structure["degrees"])
    _, bits = basis_bits(n)
    class_matrix = np.zeros((len(structure["class_sizes"]), n))
    for class_position, vset in enumerate(structure["class_vertex_sets"]):
        class_matrix[class_position, vset] = 1.0
    occupancy = (class_matrix @ bits.T.astype(float)).round().astype(np.int64)
    flat_index = np.ravel_multi_index(
        tuple(occupancy), dims=[s + 1 for s in structure["class_sizes"]])
    out = np.zeros(structure["n_sectors"])
    np.add.at(out, flat_index, probabilities)
    return out


def permute_probabilities(probabilities: np.ndarray, permutation) -> np.ndarray:
    """Outcome probabilities of the same state after relabeling vertices.

    ``permutation[i]`` is the new label of old vertex ``i``.
    """
    n = int(np.log2(len(probabilities)))
    _, bits = basis_bits(n)
    permutation = np.asarray(permutation, dtype=np.int64)
    permuted_index = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        permuted_index += bits[:, i].astype(np.int64) << permutation[i]
    relabeled = np.empty_like(probabilities)
    relabeled[permuted_index] = probabilities
    return relabeled


def validate_relabeling_invariance(adj_mat, rng=None, atol=1e-12, **circuit_kwargs) -> dict:
    """Coordinatewise invariance of R0/R1/R2 under a random vertex relabeling.

    The relabeled *graph* is re-embedded (not just the probability vector
    permuted), so this checks the full pipeline: circuit, simulator,
    sector construction, and pooling.
    """
    from .statevector import circuit_probabilities

    rng = np.random.default_rng(0) if rng is None else rng
    adj_mat = np.asarray(adj_mat)
    n = adj_mat.shape[0]
    perm = rng.permutation(n)
    permuted_adj = adj_mat[np.ix_(np.argsort(perm), np.argsort(perm))]

    p = circuit_probabilities(adj_mat, **circuit_kwargs)
    p_perm = circuit_probabilities(permuted_adj, **circuit_kwargs)

    s = sector_structure(adj_mat.sum(axis=1))
    s_perm = sector_structure(permuted_adj.sum(axis=1))

    errors = {
        "R0": float(np.max(np.abs(readout_R0(p) - readout_R0(p_perm)))),
        "R1": float(np.max(np.abs(readout_R1(p) - readout_R1(p_perm)))),
        "R2": float(np.max(np.abs(readout_R2(p, s) - readout_R2(p_perm, s_perm)))),
    }
    assert max(errors.values()) < atol, f"relabeling invariance broken: {errors}"
    return errors


def validate_R2_implementations(probabilities: np.ndarray, degrees, atol=1e-12) -> float:
    """Cross-check the two R2 implementations and the sector cardinalities."""
    structure = sector_structure(degrees)
    primary = readout_R2(probabilities, structure)
    independent = readout_R2_independent(probabilities, structure)
    err = float(np.max(np.abs(primary - independent)))
    assert err < atol, f"R2 implementations disagree: {err:.3e}"
    observed = np.bincount(structure["sector_id"], minlength=structure["n_sectors"])
    expected = np.array([
        int(np.prod([comb(structure["class_sizes"][j], signature[j])
                     for j in range(len(structure["class_sizes"]))]))
        for signature in structure["canonical_tuples"]
    ])
    assert np.array_equal(observed, expected), "sector cardinalities are wrong"
    assert abs(primary.sum() - probabilities.sum()) < atol
    return err
