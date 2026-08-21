#!/usr/bin/env python3
"""Regenerate every file in data/ from scratch, deterministically.

The committed data files are the output of this script.  Nothing in them is
hand-edited; rerunning the script reproduces them bit-for-bit (fingerprints
are printed so a rerun can be compared at a glance).  Runtime: about two
minutes on a laptop.

    python scripts/generate_demo_data.py
"""
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

from quic import circuit_probabilities
from quic.datasets import cubic_census, degree_stratum, graph6_sha256
from quic.finite_shots import recovery_curves, ridge_probe_r2
from quic.graph_features import (
    cospectral_groups, cross_degree_edge_count, cycle_counts, graph_invariants,
)
from quic.readouts import readout_R0, sector_structure

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "data" / "demo_graphs"
RESULTS = ROOT / "data" / "curated_results"
DEMO.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def embedding(adj):
    return readout_R0(circuit_probabilities(np.asarray(adj, dtype=float)))


# ---------------------------------------------------------------- demo graphs
print("== data/demo_graphs ==")

graphs10, g6_10 = cubic_census(10)
(DEMO / "cubic_n10.graph6").write_text("\n".join(g6_10) + "\n")
print(f"cubic_n10.graph6: {len(g6_10)} graphs, census sha256 "
      f"{graph6_sha256(g6_10)[:16]}")

witnesses = {}
for n in (14, 16):
    graphs, g6 = cubic_census(n)
    adj = [nx.to_numpy_array(g) for g in graphs]
    groups = cospectral_groups(adj)
    group_records = []
    for grp in groups:
        invariants = {i: graph_invariants(graphs[i]) for i in grp}
        names = list(next(iter(invariants.values())))
        differing = [nm for nm in names
                     if len({invariants[i][nm] for i in grp}) > 1]
        embs = {i: embedding(adj[i]) for i in grp}
        min_sep = min(float(np.abs(embs[i] - embs[j]).sum())
                      for i, j in combinations(grp, 2))
        group_records.append({
            "census_indices": grp,
            "graph6": [g6[i] for i in grp],
            "differing_invariants": differing,
            "invariants": {str(i): invariants[i] for i in grp},
            "min_quic_l1_separation": min_sep,
        })
    witnesses[str(n)] = {
        "population": f"connected cubic graphs, n={n}, nauty-geng -c -d3 -D3 {n}",
        "census_size": len(graphs),
        "census_graph6_sha256": graph6_sha256(g6),
        "cospectrality": "exact integer trace tuples (no tolerance)",
        "groups": group_records,
    }
(DEMO / "cospectral_witnesses.json").write_text(json.dumps(witnesses, indent=1))
n_groups = {n: len(witnesses[n]["groups"]) for n in witnesses}
print(f"cospectral_witnesses.json: groups {n_groups}  [{time.time()-t0:.0f}s]")

DEGSEQ = [2] * 7 + [4] * 7
stratum = degree_stratum(DEGSEQ, count=200, seed=7)
stratum_records = {
    "population": "connected simple graphs with degree sequence 2^7 4^7, n=14",
    "sampler": "quic.datasets.degree_stratum(count=200, seed=7)",
    "graph6": [nx.to_graph6_bytes(g, header=False).decode().strip()
               for g in stratum],
    "mixing_target_m24": [cross_degree_edge_count(g) for g in stratum],
}
(DEMO / "bimodal_stratum_n14.json").write_text(json.dumps(stratum_records, indent=1))
print(f"bimodal_stratum_n14.json: 200 graphs  [{time.time()-t0:.0f}s]")

# ------------------------------------------------------------ curated results
print("\n== data/curated_results ==")

# hierarchy_summary.csv — the notebook-03 analyses in long form
graphs14, g6_14 = cubic_census(14)
EMB = np.vstack([embedding(nx.to_numpy_array(g)) for g in graphs14])
counts = [cycle_counts(g, max_len=6) for g in graphs14]
CYC = {k: np.array([c[k] for c in counts], dtype=float) for k in (3, 4, 5, 6)}

from scipy import stats
from scipy.spatial.distance import pdist

rows = []
D_emb = pdist(EMB, metric="cityblock")
rows.append(("global", "all", 509,
             "mantel_r_vs_dC3",
             float(stats.pearsonr(D_emb, pdist(CYC[3][:, None], "cityblock"))[0])))
for v in sorted(set(CYC[3].astype(int))):
    idx = np.nonzero(CYC[3] == v)[0]
    if len(idx) < 12:
        continue
    r = float(stats.pearsonr(pdist(EMB[idx], "cityblock"),
                             pdist(CYC[4][idx, None], "cityblock"))[0])
    rows.append(("fixed_C3", f"C3={v}", len(idx), "mantel_r_vs_dC4", r))
for a in sorted(set(CYC[3].astype(int))):
    for b in sorted(set(CYC[4].astype(int))):
        idx = np.nonzero((CYC[3] == a) & (CYC[4] == b))[0]
        if len(idx) < 15:
            continue
        dt = pdist(CYC[5][idx, None], "cityblock")
        if dt.std() == 0:
            continue
        r = float(stats.pearsonr(pdist(EMB[idx], "cityblock"), dt)[0])
        rows.append(("fixed_C3_C4", f"({a},{b})", len(idx), "mantel_r_vs_dC5", r))
for k in (3, 4, 5, 6):
    rows.append(("linear_probe", f"C{k}", 509, "heldout_ridge_r2",
                 ridge_probe_r2(EMB, CYC[k])))
rng = np.random.default_rng(1)
rows.append(("linear_probe", "C3_shuffled_control", 509, "heldout_ridge_r2",
             ridge_probe_r2(EMB, rng.permutation(CYC[3]))))
with open(RESULTS / "hierarchy_summary.csv", "w") as f:
    f.write("analysis,stratum,n_graphs,metric,value\n")
    for row in rows:
        f.write(",".join(map(str, row[:4])) + f",{row[4]:.6f}\n")
print(f"hierarchy_summary.csv: {len(rows)} rows  [{time.time()-t0:.0f}s]")

# witness_separation.csv
with open(RESULTS / "witness_separation.csv", "w") as f:
    f.write("n,group_indices,differing_invariants,min_quic_l1_separation\n")
    for n in ("14", "16"):
        for grp in witnesses[n]["groups"]:
            f.write(f"{n},\"{grp['census_indices']}\","
                    f"\"{grp['differing_invariants']}\","
                    f"{grp['min_quic_l1_separation']:.6e}\n")
print(f"witness_separation.csv written  [{time.time()-t0:.0f}s]")

# finite_shot_summary.csv — notebook-05 grid
adjacency = [nx.to_numpy_array(g) for g in stratum]
target = np.array(stratum_records["mixing_target_m24"], dtype=float)
structures = [sector_structure(a.sum(axis=1).astype(int)) for a in adjacency]
ladder = tuple(1 << k for k in (10, 12, 14, 16, 18))
with open(RESULTS / "finite_shot_summary.csv", "w") as f:
    f.write("prep,readout,exact_ceiling,"
            + ",".join(f"r2_mean_2e{int(np.log2(s))}" for s in ladder)
            + ",r2_sd_2e18\n")
    for prep in ("flat", "hadamard", "degree"):
        probs = [circuit_probabilities(a, prep=prep) for a in adjacency]
        res = recovery_curves(probs, target, structures=structures,
                              shot_ladder=ladder, n_replicates=6, seed=0)
        for kind in ("R0", "R1", "R2"):
            means = [float(np.mean(res["curves"][kind][s])) for s in ladder]
            sd18 = float(np.std(res["curves"][kind][1 << 18]))
            f.write(f"{prep},{kind},{res['ceilings'][kind]:.4f},"
                    + ",".join(f"{m:.4f}" for m in means) + f",{sd18:.4f}\n")
        print(f"finite_shot_summary: {prep} done  [{time.time()-t0:.0f}s]")

print(f"\nall data regenerated in {time.time()-t0:.0f}s")
