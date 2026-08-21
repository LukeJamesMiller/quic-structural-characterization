"""Finite-shot estimation: sampling, empirical readouts, recovery curves.

The exact-state analyses answer what information the representation
contains; this module answers what survives a shot budget.  The
estimator is deliberately the naive one a hardware user would run:
sample the circuit ``shots`` times, form the empirical distribution,
apply the readout to it, regress.  All randomness flows through seeded
``numpy`` generators so every curve in the notebooks is reproducible
bit-for-bit.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from .readouts import readout_R0, readout_R1, readout_R2

__all__ = [
    "sample_empirical_distribution", "empirical_readout",
    "ridge_probe_r2", "recovery_curves", "total_variation",
]


def sample_empirical_distribution(probabilities: np.ndarray, shots: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Empirical outcome distribution from ``shots`` measurement samples."""
    counts = rng.multinomial(shots, probabilities / probabilities.sum())
    return counts / shots


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def empirical_readout(empirical: np.ndarray, kind: str, structure=None,
                      k_head: int | None = None) -> np.ndarray:
    """Apply a readout (R0/R1/R2) to an empirical distribution."""
    if kind == "R0":
        vec = readout_R0(empirical)
        return vec[:k_head] if k_head else vec
    if kind == "R1":
        return readout_R1(empirical)
    if kind == "R2":
        if structure is None:
            raise ValueError("R2 needs the sector structure")
        return readout_R2(empirical, structure)
    raise ValueError(f"unknown readout {kind!r}")


def ridge_probe_r2(features: np.ndarray, targets: np.ndarray, *,
                   alphas=tuple(np.logspace(-10, 3, 14)),
                   n_folds: int = 5, seed: int = 0) -> float:
    """Held-out R^2 of a ridge probe; alpha chosen within each training fold.

    The probe is intentionally linear: the question throughout is linear
    accessibility of a target from a readout, not what a flexible model
    could extract.  Features are rescaled by one scalar (their training-fold
    RMS) so the wide alpha grid is meaningful across readouts of very
    different magnitudes; alpha is selected by efficient leave-one-out
    generalized CV (RidgeCV) inside the training fold only.  Returns the
    mean held-out R^2 across outer folds.
    """
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    outer = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train_index, test_index in outer.split(features):
        X_train, y_train = features[train_index], targets[train_index]
        X_test, y_test = features[test_index], targets[test_index]
        scale = np.sqrt((X_train ** 2).mean()) or 1.0
        model = RidgeCV(alphas=alphas)
        model.fit(X_train / scale, y_train)
        scores.append(model.score(X_test / scale, y_test))
    return float(np.mean(scores))


def recovery_curves(exact_probabilities, targets, *, structures=None,
                    shot_ladder=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
                    readouts=("R0", "R1", "R2"), n_replicates: int = 6,
                    k_head: int = 500, seed: int = 0,
                    probe_kwargs: dict | None = None) -> dict:
    """Finite-shot recovery of a target under each readout, paired design.

    One empirical distribution is sampled per (graph, budget, replicate) and
    every readout is applied to the *same* sample, so per-replicate scores
    are paired across readouts and their differences are meaningful.  For
    every readout an exact-state ceiling (probe on exact readout vectors) is
    reported alongside.  All randomness is seeded; identical calls reproduce
    bit-for-bit.

    Returns ``{"shot_ladder", "ceilings": {readout: r2},
    "curves": {readout: {shots: [r2 per replicate]}},
    "tv": {shots: [mean total-variation error per replicate]}}``.
    """
    probe_kwargs = probe_kwargs or {}
    n_graphs = len(exact_probabilities)
    result: dict = {"shot_ladder": list(shot_ladder), "ceilings": {},
                    "curves": {kind: {int(s): [] for s in shot_ladder}
                               for kind in readouts},
                    "tv": {int(s): [] for s in shot_ladder}}

    def features_of(distribution, index, kind):
        return empirical_readout(
            distribution, kind,
            None if structures is None else structures[index],
            k_head=k_head if kind == "R0" else None)

    for kind in readouts:
        exact_features = np.vstack([features_of(exact_probabilities[i], i, kind)
                                    for i in range(n_graphs)])
        result["ceilings"][kind] = ridge_probe_r2(exact_features, targets,
                                                  seed=seed, **probe_kwargs)

    for shots in shot_ladder:
        for replicate in range(n_replicates):
            rng = np.random.default_rng((seed, int(shots), replicate))
            empirical = [sample_empirical_distribution(exact_probabilities[i],
                                                       int(shots), rng)
                         for i in range(n_graphs)]
            result["tv"][int(shots)].append(float(np.mean(
                [total_variation(empirical[i], exact_probabilities[i])
                 for i in range(n_graphs)])))
            for kind in readouts:
                features = np.vstack([features_of(empirical[i], i, kind)
                                      for i in range(n_graphs)])
                result["curves"][kind][int(shots)].append(
                    ridge_probe_r2(features, targets, seed=seed + replicate,
                                   **probe_kwargs))
    return result
