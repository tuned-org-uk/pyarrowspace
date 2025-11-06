"""
The simplest test for eigenmaps.

builder type: build
search type: search
"""
import numpy as np
from arrowspace import ArrowSpaceBuilder, GraphLaplacian

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

# Returns an ArrowSpace with computed signal graph and lambdas
aspace, gl = ArrowSpaceBuilder.build(graph_params, items)
print("aspace sorted lambdas", aspace.lambdas_sorted())

# Search comparable items (defaults: k = nitems, alpha = 1.0, beta = 0.0)
query1 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query1, gl, 1.0)  # list[(idx, score)]


print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 2)
assert(hits[1][0] == 1)
assert(hits[2][0] == 4)

# Search comparable items (defaults: k = nitems, alpha = 0.9, beta = 0.1)
query2 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query2, gl, 0.9)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 4)
assert(hits[1][0] == 2)
assert(hits[2][0] == 1)

# Search comparable items (defaults: k = nitems, alpha = 0.6, beta = 0.4)
query3 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query3, gl, 0.6)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 4)
assert(hits[1][0] == 2)
assert(hits[2][0] == 1)

query4 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search(query4, gl, 0.55)  # list[(idx, score)]

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 4)
assert(hits[1][0] == 2)
assert(hits[2][0] == 1)