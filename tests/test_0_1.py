"""
The linear search test for eigenmaps.

builder type: build
search type: search_linear_sorted
"""
import time
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
    "eps": 0.2,
    "k": 4,
    "topk": 4,
    "p": 2.0,
    "sigma": 0.08,
}

def build_energy_index(emb, eta, steps, optical_tokens=None):
    """Build energy-only index with specified diffusion parameters."""
    energy_params = {
        "optical_tokens": None, # ← Rust computes 2√N automatically
        "trim_quantile": 0.005,
        "eta": eta,
        "steps": steps,
        "split_quantile": 0.8,
        "neighbor_k": 4,
        "split_tau": 0.15,
        "w_lambda": 1.0,
        "w_disp": 0.5,
        "w_dirichlet": 0.25,
        "candidate_m": 40,
    }
    
    print(f"Building energy index: η={eta}, steps={steps}, optical_tokens={optical_tokens}")
    start = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder.build_energy(
        emb,
        energy_params=energy_params,
        graph_params=graph_params
    )
    build_time = time.perf_counter() - start
    print(f"Energy build time: {build_time:.2f}s")
    
    return aspace, gl

# Returns an ArrowSpace with computed signal graph and lambdas
aspace, gl = build_energy_index(items, 0.08, 8)
print("aspace sorted lambdas", aspace.lambdas_sorted())

# Search comparable items
query1 = np.array(items[0] * 1.05, dtype=np.float64)
hits = aspace.search_linear_sorted(query1, gl, 3)


print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 0)
assert(hits[1][0] == 1)
assert(hits[2][0] == 2)

query2 = np.array(items[1] * 1.05, dtype=np.float64)
hits = aspace.search_linear_sorted(query2, gl, 3)

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 0)
assert(hits[1][0] == 1)
assert(hits[2][0] == 2)


query3 = np.array(items[2] * 1.05, dtype=np.float64)
hits = aspace.search_linear_sorted(query3, gl, 3)

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 0)
assert(hits[1][0] == 1)
assert(hits[2][0] == 2)

query4 = np.array(items[3] * 1.05, dtype=np.float64)
hits = aspace.search_linear_sorted(query4, gl, 3)

print(hits)
assert(len(hits) == 3)
assert(hits[0][0] == 0)
assert(hits[1][0] == 1)
assert(hits[2][0] == 2)