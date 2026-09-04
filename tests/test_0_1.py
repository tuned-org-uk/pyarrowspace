"""
EnergyMaps lambda-band (linear sorted) search test.

builder type: build_energy
search type: search_linear_sorted
dataset: tests/data/eigenmaps_controlled.parquet — 1000 items x 128 dims,
3 orthonormal clusters of increasing sparsity (see tests/data/CALIBRATION.md)

Semantics under arrowspace 0.28.0 (post #167 DenseMatrix layout fix):

search_linear_sorted is a lambda-proximity scan, not cosine search: it
returns up to k items whose normalised lambda falls inside the query's band
(std_dev / 2^p around the query's lambda). On the energy track this means:

  - results are deterministic for identical queries and invariant to
    positive query scaling (the lambda path is scale-tolerant)
  - hits come back lambda-ascending (BTreeMap range order)
  - hits are spectrally coherent: they all sit in one cluster, because a
    lambda bucket is populated by one subcentroid's items
  - the self item is NOT guaranteed to rank first (or appear at all): the
    query maps to its subcentroid's lambda, and the band is answered in
    lambda order — "who shares my spectral role", not "who is cosine-near"

The old (<= 0.27.4) expectations of exact [0, 1, 2] hit ordering were
artifacts of the scrambled DenseMatrix buffers fixed in #167 and were
retired with this release.
"""
import os

import numpy as np
import pyarrow.parquet as pq
from arrowspace import ArrowSpaceBuilder

import logging
logging.basicConfig(level=logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
table = pq.read_table(os.path.join(HERE, "data", "eigenmaps_controlled.parquet")).to_pandas()
labels = table["cluster"].to_numpy()
items = table.drop(columns=["cluster"]).to_numpy(dtype=np.float64)

graph_params = {
    "eps": 0.5,
    "k": 12,
    "topk": 3,
    "p": 2.0,
    "sigma": 0.5,
}
energy_params = {
    "optical_tokens": None,
    "trim_quantile": 0.005,
    "eta": 0.08,
    "steps": 8,
    "split_quantile": 0.8,
    "neighbor_k": 4,
    "split_tau": 0.15,
    "w_lambda": 1.0,
    "w_disp": 0.5,
    "w_dirichlet": 0.25,
    "candidate_m": 40,
}

aspace, gl = ArrowSpaceBuilder().build_energy(
    items, energy_params=energy_params, graph_params=graph_params
)

# Lambdas are min-max normalised to [0, 1]; the minimum subcentroid anchors
# an exact-zero floor, so at least one item sits at 0 by construction.
lam = aspace.lambdas()
print("lambda stats", lam.min(), lam.max(), lam.mean())
assert(lam.min() >= 0.0 and lam.max() <= 1.0)
assert(int(np.sum(np.abs(lam) < 1e-12)) >= 1)  # min-max normalisation floor

for seed in (0, 500, 900):        # one query per planted cluster
    for k in (3, 5):
        query = np.array(items[seed] * 1.05, dtype=np.float64)
        hits = aspace.search_linear_sorted(query, gl, k)
        hits_again = aspace.search_linear_sorted(query, gl, k)

        print(f"seed={seed} k={k} hits={hits}")
        assert(len(hits) == k)                          # band is dense enough
        assert(hits == hits_again)                      # deterministic
        scores = [s for _, s in hits]
        assert(scores == sorted(scores))                # lambda-ascending
        assert(max(scores) - min(scores) <= 0.2)        # spectrally tight band
        hit_clusters = set(labels[idx] for idx, _ in hits)
        assert(len(hit_clusters) == 1)                  # one subcentroid's bucket

# The low-lambda cluster retrieves its own items: its bucket anchors the
# band the query maps into.
query = np.array(items[500] * 1.05, dtype=np.float64)
hits = aspace.search_linear_sorted(query, gl, 5)
assert(all(labels[idx] == labels[500] for idx, _ in hits))

# Positive query scaling does not move the lambda band.
q1 = np.array(items[0] * 1.05, dtype=np.float64)
q2 = np.array(items[0] * 1.2, dtype=np.float64)
assert(
    [i for i, _ in aspace.search_linear_sorted(q1, gl, 3)]
    == [i for i, _ in aspace.search_linear_sorted(q2, gl, 3)]
)
