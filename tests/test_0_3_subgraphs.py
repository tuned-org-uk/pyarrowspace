import numpy as np
from arrowspace import ArrowSpaceBuilder
import logging

# Configure logging to see ArrowSpace internals
logging.basicConfig(level=logging.INFO)

def make_dense_clusters(seed=7):
    """Generate 60 items: three dense, well-separated clusters with strong internal structure."""
    rng = np.random.default_rng(seed)
    
    # Create three orthogonal cluster centers with stronger signals
    # Cluster A: dominant in dimensions 0-3
    a = np.array([0.85, 0.85, 0.85, 0.85, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # Cluster B: dominant in dimensions 4-7
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 0.85, 0.85, 0.1, 0.1, 0.0, 0.0], dtype=np.float64)
    # Cluster C: dominant in dimensions 8-11
    c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 0.85, 0.85], dtype=np.float64)
    
    # Normalize centers
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    def jitter(center, n, sigma=0.05):
        """Add controlled noise to create dense intra-cluster structure."""
        # Generate points near the center
        X = center[None, :] + sigma * rng.standard_normal((n, center.size))
        # Normalize to unit sphere
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # Add small systematic variations within cluster to create substructure
        for i in range(n):
            # Add subtle bias to create motif-friendly substructure
            if i % 2 == 0:
                X[i] += 0.02 * center
            else:
                X[i] -= 0.02 * center
            X[i] /= np.linalg.norm(X[i])
        return X

    # Create three clusters of 20 items each
    A = jitter(a, 20)
    B = jitter(b, 20)
    C = jitter(c, 20)
    
    return np.vstack([A, B, C]).astype(np.float64)


# Generate denser dataset
items = make_dense_clusters()
print("Dataset shape:", items.shape)  # (60, 12)

# Check sample density
sample_density = np.mean([np.dot(items[0], items[i]) for i in range(1, min(10, len(items)))])
print(f"Dataset density check (mean pairwise cosine sample): {sample_density:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# Build eigen ArrowSpace with permissive parameters
# ────────────────────────────────────────────────────────────────────────────
# Configure builder with more permissive graph parameters (eps=0.8)
builder = (
    ArrowSpaceBuilder()
    .with_seed(42)  # Ensure reproducibility
)

# Build using ONLY the data matrix
aspace, gl = builder.build(dict(eps=0.8, k=10, topk=8, p=2.0, sigma=None), items)

print(f"\nbuild complete: nitems={aspace.nitems}, nfeatures={aspace.nfeatures}, lambdas={len(aspace.lambdas())}")
print(f"ArrowSpace built: nitems={aspace.nitems}, centroids={gl.nnodes}")


# ────────────────────────────────────────────────────────────────────────────
# 1. Centroid-based subgraphs (hierarchy)
# ────────────────────────────────────────────────────────────────────────────
print(f"\nspot_subg_centroids -- gl.shape={gl.shape}, max_depth=2, min_centroids=2")
centroid_params = dict(
    eps=0.8,           # increased from 0.4 for denser connectivity
    k=10,
    topk=8,
    p=2.0,
    sigma=None,
    normalise=True,
    sparsitycheck=False,
    min_centroids=2,
    max_depth=2,
)
subg_centroids = aspace.spot_subg_centroids(gl, centroid_params)
print(f"\n✓ Centroid subgraphs: {len(subg_centroids)}")
for i, sg in enumerate(subg_centroids):
    print(f"  C{i}: level={sg['level']}, nnodes={sg['nnodes']}, nfeatures={sg['nfeatures']}")


# ────────────────────────────────────────────────────────────────────────────
# 2. Motif-based subgraphs (on eigen centroid graph)
# ────────────────────────────────────────────────────────────────────────────
print(f"\nspot_subg_motives -- gl.shape={gl.shape}, min_size=1, rayleigh_max=None")
cfg_motives = dict(
    top_l=8,              # use fewer eigenvectors (centroids < 10)
    min_triangles=1,      # allow sparse motifs
    min_clust=0.10,       # lowered from 0.15 for more permissive clustering
    max_motif_size=10,    # can cover most centroids
    max_sets=20,
    jaccard_dedup=0.25,   # lowered from 0.3 for less aggressive dedup
    min_size=1,           # allow singleton motifs to pass filter
    rayleigh_max=None,
)
subg_motives = aspace.spot_subg_motives(gl, cfg_motives)
print(f"\n✓ Motif subgraphs (eigen): {len(subg_motives)}")
for i, sg in enumerate(subg_motives):
    items_i = sg.get("item_indices", []) or []
    print(f"  M{i}: nnodes={sg['nnodes']}, items={len(items_i)}, rayleigh={sg['rayleigh']}")


# ────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────
print(f"\nDiagnostic:")
print(f"  Centroids created: {gl.nnodes}")
print(f"  Graph shape: {gl.shape}")
try:
    print(f"  Graph NNZ: {gl.matrix.nnz()}")
except:
    print("  Graph NNZ: (method not available)")
    
if hasattr(aspace, 'lambdas') and len(aspace.lambdas()) > 0:
    print(f"  Lambda stats: min={min(aspace.lambdas()):.6f}, max={max(aspace.lambdas()):.6f}, mean={np.mean(aspace.lambdas()):.6f}")


# ────────────────────────────────────────────────────────────────────────────
# Assertions
# ────────────────────────────────────────────────────────────────────────────
assert len(subg_centroids) >= 1, f"Centroid subgraphs too few: {len(subg_centroids)}"
print("motives: ", len(subg_motives)) 
