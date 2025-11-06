import numpy as np
from arrowspace import ArrowSpaceBuilder

def make_two_motifs_small(seed=7):
    rng = np.random.default_rng(seed)
    d = 12
    a = np.array([0.8,0.1,0.45,0.27,0.62,0.33,0.55,0.47,0.20,0.70,0.08,0.36], dtype=np.float64)
    b = np.array([0.15,0.72,0.22,0.66,0.18,0.60,0.24,0.58,0.26,0.64,0.20,0.62], dtype=np.float64)
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    def jitter(center, n, sigma=0.012):
        X = center[None,:] + sigma*rng.standard_normal((n, center.size))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    A = jitter(a, 5); B = jitter(b, 5)
    return np.vstack([A, B]).astype(np.float64)

items = make_two_motifs_small()

# Eigen graph parameters: small eps, modest k; normalize to favor angular structure
graph_params_eigen = dict(eps=0.5, k=5, topk=4, p=2.0, sigma=None)

# Energy graph parameters can reuse same lambda graph for consistency
graph_params_eng = dict(eps=0.4, k=8, topk=4, p=2.0, sigma=None)

# Build EigenMaps
aspace, gl = ArrowSpaceBuilder.build(graph_params_eigen, items)
# Motives config: relaxed for tiny graphs; avoid double pruning by keeping top_l >= topk
cfg_eigen = dict(top_l=6, min_triangles=2, min_clust=0.4, max_motif_size=8, max_sets=8, jaccard_dedup=0.8)
motifs_eig = aspace.spot_motives_eigen(gl, cfg_eigen)
print("Motives eigen:", motifs_eig)

# Build EnergyMaps with slightly higher neighbor_k so subcentroids tie within motifs
energy_params = {
    "optical_tokens": None,
    "trim_quantile": 0.0,
    "eta": 0.06,
    "steps": 6,
    "split_quantile": 0.9,
    "neighbor_k": 6,          # ↑ a bit to ensure triangles among subcentroids
    "split_tau": 0.12,
    "w_lambda": 1.0,
    "w_disp": 0.5,
    "w_dirichlet": 0.25,
    "candidate_m": 32,
}
aspace_e, gl_e = ArrowSpaceBuilder.build_energy(items, energy_params, graph_params_eng)

# For EnergyMaps on very small graphs
cfg_eng = dict(
    top_l=10,            # ≥ energy neighbor_k to keep triangle closures
    min_triangles=1,     # admit seeds with scarce triangles
    min_clust=0.30,      # slight relaxation
    max_motif_size=8,   # allow full motif growth
    max_sets=12,
    jaccard_dedup=0.8
)
motifs_eng = aspace_e.spot_motives_energy(gl_e, cfg_eng)
print("Motives energy:", motifs_eng)

# Simple assertion: expect at least 2 motifs for this toy
assert len(motifs_eig) >= 2, f"Eigen motifs too few: {motifs_eig}"
assert len(motifs_eng) >= 2, f"Energy motifs too few: {motifs_eng}"
