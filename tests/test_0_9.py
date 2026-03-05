"""
The simplest test for eigenmaps.

builder type: build
search type: search
"""
import numpy as np
from arrowspace import ArrowSpaceBuilder, GraphLaplacian

import logging
logging.basicConfig(level=logging.INFO)

def make_motifs(seed=7, n=2, items_per_motif=500):
    rng = np.random.default_rng(seed)
    d = 12

    # Generate n well-spread centroids on the unit sphere via random orthogonalisation
    raw = rng.standard_normal((n, d))
    Q, _ = np.linalg.qr(raw.T)          # Q is d×n, columns are orthonormal
    centroids = Q.T                       # n×d, each row is a unit vector

    def jitter(center, count, sigma=0.012):
        X = center[None, :] + sigma * rng.standard_normal((count, center.size))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    clusters = [jitter(centroids[i], items_per_motif) for i in range(n)]
    return np.vstack(clusters).astype(np.float64)


temp0 = make_motifs(n=2)
temp1 = make_motifs(34, n=5)
temp2 = make_motifs(58, n=8)

items = np.vstack([temp0, temp1, temp2]).astype(np.float64)

graph_params = {
    "eps": 0.25,
    "k": int(len(items[0]) / 4),
    "topk": 5,
    "p": 2.0,
    "sigma": 0.05,
}

builder = (ArrowSpaceBuilder()
    .with_dims_reduction(False, None)
    .with_sampling("simple", 0.6)
    .with_cluster_max_clusters(5)
)
# Returns an ArrowSpace with computed signal graph and lambdas
aspace, gl = builder.build_and_store(graph_params, items)
print("aspace sorted lambdas", aspace.lambdas_sorted())

# Search comparable items (defaults: k = nitems, alpha = 1.0, beta = 0.0)
query1 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query1, gl, 1.0)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)

# Search comparable items (defaults: k = nitems, alpha = 0.9, beta = 0.1)
query2 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query2, gl, 0.9)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)

# Search comparable items (defaults: k = nitems, alpha = 0.6, beta = 0.4)
query3 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query3, gl, 0.6)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)

query4 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query4, gl, 0.55)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)
