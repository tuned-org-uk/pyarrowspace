"""
Generates the shared parquet dataset used by tests/test_0_*.py.

Dataset is deterministic (fixed seed). Regenerate with:
    uv run python tests/data/make_datasets.py

Design (see CALIBRATION.md): 1000 items x 128 dims in 3 orthonormal clusters
of increasing sparsity. Dense clusters get low lambdas, sparse clusters high
lambdas — the spectral statistic tracks structural density.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

D = 128
CLUSTER_SIZES = (400, 350, 250)      # dense, medium, sparse
CLUSTER_SIGMA = (0.02, 0.05, 0.12)
SEED = 11


def make_clustered(seed=SEED, d=D, sizes=CLUSTER_SIZES, sigmas=CLUSTER_SIGMA):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(sizes), d))
    Q, _ = np.linalg.qr(raw.T)            # orthonormal 128-dim centroids
    centroids = Q.T

    rows, labels = [], []
    for c, (n, sig) in enumerate(zip(sizes, sigmas)):
        X = centroids[c][None, :] + sig * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        rows.append(X)
        labels.append(np.full(n, c))
    return np.vstack(rows).astype(np.float64), np.concatenate(labels)


def save_parquet(items: np.ndarray, labels: np.ndarray, name: str):
    df = pd.DataFrame(items, columns=[f"f{i}" for i in range(items.shape[1])])
    df["cluster"] = labels.astype(np.int64)
    path = os.path.join(HERE, name)
    df.to_parquet(path, index=False)
    print(f"wrote {path}  shape={items.shape}")


if __name__ == "__main__":
    items, labels = make_clustered()
    save_parquet(items, labels, "eigenmaps_controlled.parquet")
