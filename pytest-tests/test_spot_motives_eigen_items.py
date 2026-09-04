"""Tests for the item-space EigenMaps motif API (arrowspace-rs #165).

`spot_motives_eigen_items` mirrors `spot_motives_energy` on the EigenMaps
track: it rebuilds the X×X Laplacian over cluster centroids, detects motifs
there, and expands each centroid set to **item indices**. The historical
`spot_motives_eigen` keeps the feature-space contract (ids enumerate the
nodes of the F×F bootstrap Laplacian) under a name that says so.

Contract pinned here:

  * EigenMaps build with full item→cluster coverage → item-index motifs,
    deterministic across calls;
  * EigenMaps build with clustering outliers (items without a cluster
    assignment) → ValueError (`EigenModeRequired`), never a degraded or
    mislabelled result;
  * EnergyMaps build → ValueError (`EigenModeRequired`): the energy track
    already owns item-space motifs via `spot_motives_energy`.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from arrowspace import ArrowSpaceBuilder

GRAPH_PARAMS = {"eps": 0.5, "k": 12, "topk": 3, "p": 2.0, "sigma": 0.5}
ENERGY_PARAMS = {
    "optical_tokens": None,
    "trim_quantile": 0.005,
    "eta": 0.08,
    "steps": 8,
    "split_quantile": 0.8,
    "neighbor_k": 4,
    "split_tau": 0.15,
    "w_lambda": 1.0,
    "w_disp": 0.5,
    "w_dirichlet": 0.25,
    "candidate_m": 40,
}
MOTIVES_CFG = {
    "top_l": 16,
    "min_triangles": 2,
    "min_clust": 0.0,
    "max_motif_size": 8,
    "max_sets": 12,
    "jaccard_dedup": 0.8,
}

_CALIBRATION = Path(__file__).resolve().parents[1] / "tests" / "data" / "eigenmaps_controlled.parquet"


def _calibration_items():
    table = pq.read_table(str(_CALIBRATION)).to_pandas()
    return table.drop(columns=["cluster"]).to_numpy(dtype=np.float64)


def _outlier_corpus():
    """48 items × 120 features in four tight clusters; the incremental
    clusterer leaves outliers unassigned on this corpus at any radius, which
    is exactly the refusal case the upstream contract mandates."""
    rng = np.random.default_rng(3407)
    centers = np.zeros((4, 120))
    for c in range(4):
        centers[c, c * 20:(c + 1) * 20] = 0.85
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    items = np.vstack(
        [centers[c] + 0.05 * rng.standard_normal((12, 120)) for c in range(4)]
    )
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return items


@pytest.fixture(scope="module")
def eigen_build():
    aspace, gl = ArrowSpaceBuilder().with_seed(3407).build(
        dict(GRAPH_PARAMS), _calibration_items()
    )
    return aspace, gl


@pytest.fixture(scope="module")
def energy_build():
    aspace, gl = ArrowSpaceBuilder().with_seed(3407).build_energy(
        _calibration_items(), dict(ENERGY_PARAMS), dict(GRAPH_PARAMS)
    )
    return aspace, gl


@pytest.fixture(scope="module")
def outlier_eigen_build():
    aspace, gl = ArrowSpaceBuilder().with_seed(3407).build(
        {"eps": 1.29, "k": 8, "topk": 4, "p": 2.0, "sigma": None}, _outlier_corpus()
    )
    return aspace, gl


def test_item_space_motifs_are_item_indices_and_deterministic(eigen_build):
    aspace, gl = eigen_build
    motifs = aspace.spot_motives_eigen_items(gl, dict(MOTIVES_CFG))
    motifs_again = aspace.spot_motives_eigen_items(gl, dict(MOTIVES_CFG))

    assert len(motifs) >= 1
    assert motifs == motifs_again  # deterministic across calls
    for motif in motifs:
        assert len(motif) >= 1
        assert all(0 <= idx < aspace.nitems for idx in motif)  # item space
        assert len(set(motif)) == len(motif)  # deduplicated


def test_item_space_motifs_refused_when_cluster_coverage_is_partial(outlier_eigen_build):
    aspace, gl = outlier_eigen_build
    with pytest.raises(ValueError, match="eigen mode required"):
        aspace.spot_motives_eigen_items(gl, dict(MOTIVES_CFG))


def test_item_space_motifs_refused_on_energy_build(energy_build):
    aspace, gl = energy_build
    with pytest.raises(ValueError, match="eigen mode required"):
        aspace.spot_motives_eigen_items(gl, dict(MOTIVES_CFG))


def test_item_space_motives_guard_fires_without_cfg(energy_build):
    """The guard fires before any cfg parsing — even a None cfg is rejected."""
    aspace, gl = energy_build
    with pytest.raises(ValueError, match="eigen mode required"):
        aspace.spot_motives_eigen_items(gl, None)
