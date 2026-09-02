"""Tests for #35 finding 1: the subgraph/motives API must enforce energy-mode
requirements instead of silently returning feature-space indices labelled as
item indices.

On an EigenMaps build there is no centroid_map and no sub_centroids, so
`spot_subg_motives` detects motifs over the F×F bootstrap Laplacian (whose
nodes enumerate FEATURES) and returns them unprojected. With F > N the two
spaces cannot be confused: any "item index" >= N proves feature leakage.

The corpus is built 48 items × 120 features deliberately, so:

  * EigenMaps build → ValueError (guard), never silent garbage;
  * EnergyMaps build → item_indices must stay within 0..N-1 (< 48).

Also covers #35 findings 2–4: cfg unknown keys must raise (same contract as
graph_params), the rayleigh/rayleigh_max coupling must be documented, and the
spot_motives_eigen docstring must state the real (gl, cfg) signature.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpace, ArrowSpaceBuilder

N_ITEMS, N_FEATURES = 48, 120

# eps=1.29 is the value the pytest suite calibrates on unit-normalised rows;
# lower eps leaves the rebuilt subcentroid graph without triangles and the
# energy motif pipeline returns zero motifs.
GRAPH_PARAMS = {"eps": 1.29, "k": 8, "topk": 4, "p": 2.0, "sigma": None}

# Energy pipeline params modelled on tests/test_0_2_motives.py.
ENERGY_PARAMS = {
    "optical_tokens": None,
    "trim_quantile": 0.0,
    "eta": 0.06,
    "steps": 6,
    "split_quantile": 0.9,
    "neighbor_k": 6,
    "split_tau": 0.12,
    "w_lambda": 1.0,
    "w_disp": 0.5,
    "w_dirichlet": 0.25,
    "candidate_m": 32,
}

SUBG_CFG = {
    "top_l": 10,
    "min_triangles": 1,
    "min_clust": 0.0,
    "max_motif_size": 8,
    "max_sets": 12,
    "jaccard_dedup": 0.8,
    "min_size": 3,
    "rayleigh_max": None,
}


def _corpus():
    """48 items × 120 features in four tight clusters — item space (48) ≠
    feature space (120), and enough structure for the energy pipeline to
    yield at least one motif."""
    rng = np.random.default_rng(3407)
    centers = np.zeros((4, N_FEATURES))
    for c in range(4):
        centers[c, c * 20:(c + 1) * 20] = 0.85
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    items = np.vstack(
        [centers[c] + 0.05 * rng.standard_normal((12, N_FEATURES)) for c in range(4)]
    )
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return items


@pytest.fixture(scope="module")
def eigen_build():
    aspace, gl = (
        ArrowSpaceBuilder().with_seed(3407).build(dict(GRAPH_PARAMS), _corpus())
    )
    return aspace, gl


@pytest.fixture(scope="module")
def energy_build():
    aspace, gl = (
        ArrowSpaceBuilder()
        .with_seed(3407)
        .build_energy(_corpus(), ENERGY_PARAMS, GRAPH_PARAMS)
    )
    return aspace, gl


# --- finding 1: EigenMaps builds must be rejected, not served garbage ---

def test_spot_subg_motives_on_eigenmaps_build_raises(eigen_build):
    aspace, gl = eigen_build
    with pytest.raises(ValueError, match="EnergyMaps"):
        aspace.spot_subg_motives(gl, dict(SUBG_CFG))


def test_spot_subg_motives_on_eigenmaps_without_cfg_raises(eigen_build):
    """The guard fires before any cfg parsing — even a None cfg must be rejected."""
    aspace, gl = eigen_build
    with pytest.raises(ValueError, match="EnergyMaps"):
        aspace.spot_subg_motives(gl, None)


def test_spot_motives_energy_on_eigenmaps_build_raises(eigen_build):
    """Regression guard for the precedent this fix extends (0.27-era behaviour)."""
    aspace, gl = eigen_build
    with pytest.raises(ValueError, match="EnergyMaps"):
        aspace.spot_motives_energy(gl, {"min_triangles": 1})


# --- finding 1: energy builds return honest item-space indices ---

def test_energy_build_item_indices_stay_in_item_space(energy_build):
    """With F=120 > N=48, any index >= 48 is a leaked feature index."""
    aspace, gl = energy_build
    subgs = aspace.spot_subg_motives(gl, dict(SUBG_CFG))
    assert len(subgs) >= 1, "corpus must yield at least one subgraph to make this oracle meaningful"
    for sg in subgs:
        items_i = sg["item_indices"]
        assert items_i is not None, "energy subgraphs must carry item_indices"
        assert max(items_i) < N_ITEMS, (
            f"item_indices leaked outside item space: max={max(items_i)} >= N={N_ITEMS}"
        )


def test_energy_build_node_indices_are_centroid_ids(energy_build):
    """node_indices are centroid ids, bounded by the subgraph's own nnodes."""
    aspace, gl = energy_build
    subgs = aspace.spot_subg_motives(gl, dict(SUBG_CFG))
    for sg in subgs:
        assert max(sg["node_indices"]) < sg["nnodes"]


# --- finding 2: rayleigh is computed only when rayleigh_max is set ---

def test_rayleigh_is_none_without_rayleigh_max(energy_build):
    aspace, gl = energy_build
    subgs = aspace.spot_subg_motives(gl, dict(SUBG_CFG))
    assert all(sg["rayleigh"] is None for sg in subgs)


def test_rayleigh_is_computed_with_rayleigh_max(energy_build):
    aspace, gl = energy_build
    cfg = dict(SUBG_CFG, rayleigh_max=1e9)
    subgs = aspace.spot_subg_motives(gl, cfg)
    assert len(subgs) >= 1
    assert all(
        isinstance(sg["rayleigh"], float) for sg in subgs
    ), "rayleigh must be populated when rayleigh_max is requested"


def test_docstring_documents_rayleigh_max_coupling():
    """#35 finding 2: the docstring must say rayleigh needs rayleigh_max."""
    doc = ArrowSpace.spot_subg_motives.__doc__ or ""
    assert "rayleigh_max" in doc


# --- finding 3: unknown cfg keys must raise, like graph_params ---

def test_unknown_subg_cfg_key_raises_type_error(eigen_build):
    aspace, gl = eigen_build
    with pytest.raises(TypeError, match="bogus_key"):
        aspace.spot_subg_motives(gl, {"min_size": 2, "bogus_key": 1})


def test_unknown_centroid_cfg_key_raises_type_error(eigen_build):
    aspace, gl = eigen_build
    with pytest.raises(TypeError, match="bogus_param"):
        aspace.spot_subg_centroids(gl, {"bogus_param": 1})


def test_known_subg_cfg_keys_accepted(energy_build):
    """Every documented key passes validation (guard first, then no raise)."""
    aspace, gl = energy_build
    aspace.spot_subg_motives(gl, dict(SUBG_CFG))  # must not raise


# --- finding 4: stale docstring arity ---

def test_spot_motives_eigen_docstring_documents_gl_argument():
    doc = ArrowSpace.spot_motives_eigen.__doc__ or ""
    assert "gl" in doc, "docstring must state the (gl, cfg) signature, not (cfg)"
