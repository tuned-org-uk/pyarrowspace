import numpy as np
from arrowspace import ArrowSpaceBuilder


def make_two_clusters(seed=7):
    """Generate 60 items: two well-separated clusters of 30 each."""
    rng = np.random.default_rng(seed)
    d = 12
    # Orthogonal cluster centers for clear separation
    a = np.array([0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    def jitter(center, n, sigma=0.015):
        X = center[None, :] + sigma * rng.standard_normal((n, center.size))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    A = jitter(a, 30)
    B = jitter(b, 30)
    return np.vstack([A, B]).astype(np.float64)


items = make_two_clusters()
print("Dataset shape:", items.shape)  # (60, 12)

# Build eigen ArrowSpace
graph_params = dict(eps=0.4, k=10, topk=8, p=2.0, sigma=None)
aspace, gl = ArrowSpaceBuilder.build_and_store(graph_params, items)

print(f"ArrowSpace built: nitems={aspace.nitems}, centroids={gl.nnodes}")

# ────────────────────────────────────────────────────────────────────────────
# 1. Centroid-based subgraphs (hierarchy)
# ────────────────────────────────────────────────────────────────────────────
centroid_params = dict(
    eps=0.4,
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
# Assertions
# ────────────────────────────────────────────────────────────────────────────
assert len(subg_centroids) >= 1, f"Centroid subgraphs too few: {len(subg_centroids)}"

