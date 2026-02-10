"""
Fine grained builder interface to set sampling and dimensionality reduction
"""
import arrowspace
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


items = np.array([
[0.82,0.11,0.43,0.28,0.64,0.32,0.55,0.48,0.19,0.73,0.07,0.36,0.58,0.23,0.44,0.31,0.52,0.16,0.61,0.40,0.27,0.49,0.35,0.29],
[0.79,0.12,0.45,0.29,0.61,0.33,0.54,0.47,0.21,0.70,0.08,0.37,0.56,0.22,0.46,0.30,0.51,0.18,0.60,0.39,0.26,0.48,0.36,0.30],
[0.78,0.13,0.46,0.27,0.62,0.34,0.53,0.46,0.22,0.69,0.09,0.35,0.55,0.24,0.45,0.29,0.50,0.17,0.59,0.38,0.28,0.47,0.34,0.31],
[0.81,0.10,0.44,0.26,0.63,0.31,0.56,0.45,0.20,0.71,0.06,0.34,0.57,0.25,0.47,0.33,0.53,0.15,0.62,0.41,0.25,0.50,0.37,0.27],
[0.80,0.12,0.42,0.25,0.60,0.35,0.52,0.49,0.23,0.68,0.10,0.38,0.54,0.21,0.43,0.28,0.49,0.19,0.58,0.37,0.29,0.46,0.33,0.32]],
dtype=np.float64)

graph_params = {
    "eps": 0.05,
    "k": len(items),
    "topk": 3,
    "p": 2.0,
    "sigma": 0.05,
}

# Eigenmaps pipeline
builder = (arrowspace.ArrowSpaceBuilder()
    .with_seed(42)
    .with_cluster_max_clusters(2)
    .with_cluster_radius(0.002)
    .with_dims_reduction(enabled=True, eps=0.14)
    .with_sampling("simple", 1.0))

aspace, gl = builder.build(graph_params, items)

assert aspace.nclusters == 2

query = np.array(items[2] * 1.05, dtype=np.float64)
results = aspace.search(query, gl, tau=0.7)
