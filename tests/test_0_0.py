"""
EigenMaps smoke test on the shared clustered calibration dataset.

builder type: build_and_store
search type: search
dataset: tests/data/eigenmaps_controlled.parquet — 1000 items x 128 dims,
3 orthonormal clusters of increasing sparsity (see tests/data/CALIBRATION.md)

Semantics under arrowspace 0.27.x:
  - the query (from the dense cluster, scaled 1.2) is answered exclusively
    inside its own cluster at every tau
  - the self item always ranks first
  - mean lambda increases with cluster sparsity
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

# Returns an ArrowSpace with computed signal graph and lambdas
aspace, gl = ArrowSpaceBuilder().build_and_store(graph_params, items)

query = np.array(items[0] * 1.2, dtype=np.float64)
for tau in (1.0, 0.9, 0.6, 0.55):
    hits = aspace.search(query, gl, tau)  # list[(idx, score)]
    print(hits)
    assert(len(hits) == 3)
    assert(hits[0][0] == 0)                       # self match first
    assert(all(labels[idx] == 0 for idx, _ in hits))  # cluster containment

# Spectral statistic tracks structural density: sparser cluster -> higher lambda
lam = aspace.lambdas()
means = [lam[labels == c].mean() for c in range(3)]
print("mean lambda per cluster", means)
assert(means[0] < means[1] < means[2])
assert(int(np.sum(np.abs(lam) < 1e-12)) <= 1)  # min-max normalization floor only
