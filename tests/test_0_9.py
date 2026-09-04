"""
Clustered smoke test for eigenmaps with sampling and cluster caps enabled.

builder type: build (with_dims_reduction off, with_sampling 1.0,
             with_cluster_max_clusters 5)
search type: search
dataset: tests/data/eigenmaps_controlled.parquet — 1000 items x 128 dims,
3 orthonormal clusters of increasing sparsity (see tests/data/CALIBRATION.md)

Semantics under arrowspace 0.28.x: search returns `topk` hits (5 here); the
query drawn from the dense cluster is answered exclusively inside it, and the
self item always ranks first.
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
    "topk": 5,
    "p": 2.0,
    "sigma": 0.5,
}

builder = (ArrowSpaceBuilder()
    .with_dims_reduction(False, None)
    .with_sampling("simple", 1.0)
    .with_cluster_max_clusters(5)
)
# Returns an ArrowSpace with computed signal graph and lambdas
aspace, gl = builder.build_and_store(graph_params, items)

# Query from the dense cluster: all hits must stay inside it
query = np.array(items[0] * 1.2, dtype=np.float64)
for tau in (1.0, 0.9, 0.6, 0.55):
    hits = aspace.search(query, gl, tau)  # list[(idx, score)]
    print(hits)
    assert(len(hits) == graph_params["topk"])
    assert(hits[0][0] == 0)                       # self match first
    assert(all(labels[idx] == 0 for idx, _ in hits))  # cluster containment
