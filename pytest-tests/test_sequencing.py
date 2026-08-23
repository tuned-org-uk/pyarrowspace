"""Tests for the sequencing API — pyarrowspace bindings over
`arrowspace::analysis::sequencing` (arrowspace-rs >= 0.26.12).

Sequencing produces total orderings over the NODES OF THE STORED LAPLACIAN
MATRIX (`gl.to_dense().shape[0]`). For standard eigen builds that matrix is
the FEATURE-space graph — shape (n_features, n_features) — so sequenced nodes
are features, NOT items (`gl.nnodes` counts items and differs from the
matrix node count whenever n_items != n_features). Per-item λ read-outs
(`aspace.lambdas()`) are a separate index space consumed by
`sequence_by_lambda`.

Two strategies are exposed:

  * `sequence_by_lambda(lambdas, descending=False)` — spectral curriculum:
    items ordered by their per-node λ (Rayleigh) score, ties on ascending index.
  * `sequence_by_graph(gl)` — MST-chain seriation: DFS preorder walk of the
    minimum spanning forest recovered from L, one contiguous block per
    connected component, larger components first.

Both return a `Sequence` with `order` (int64 node ids),
`positions` (float64 per-step coordinate), `components` (component count).

Contract pinned here mirrors the Rust unit tests in
arrowspace-rs/src/tests/test_sequencing.rs.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder, sequence_by_graph, sequence_by_lambda


# --- helpers -------------------------------------------------------------

def _build(items, gp=None):
    """Build (aspace, gl) from a small deterministic corpus."""
    params = gp or {"eps": 1.0, "k": 3, "topk": 3, "p": 2.0, "sigma": None}
    aspace, gl = (
        ArrowSpaceBuilder()
        .with_seed(3407)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build(params, items)
    assert gl is not None and gl.nnodes == len(items)
    return aspace, gl


@pytest.fixture(scope="module")
def small_graph():
    """A small deterministic build reused across sequence_by_graph tests."""
    rng = np.random.default_rng(3407)
    items = rng.standard_normal((40, 8))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return _build(items)


# --- Sequence type -------------------------------------------------------

def test_sequence_type_exposes_order_positions_components(small_graph):
    _, gl = small_graph
    seq = sequence_by_graph(gl)
    assert isinstance(seq.order, np.ndarray) and seq.order.dtype == np.int64
    assert isinstance(seq.positions, np.ndarray) and seq.positions.dtype == np.float64
    assert isinstance(seq.components, int)


# --- sequence_by_lambda ---------------------------------------------------

def test_lambda_ascending_matches_reference_ordering():
    # Reference: arrowspace-rs doc example — [0.9, 0.1, 0.4] asc -> [1, 2, 0]
    seq = sequence_by_lambda([0.9, 0.1, 0.4])
    assert seq.order.tolist() == [1, 2, 0]
    assert seq.positions.tolist() == [0.1, 0.4, 0.9]
    assert seq.components == 1


def test_lambda_descending_reverses_order():
    seq = sequence_by_lambda([0.5, 0.1, 0.9], descending=True)
    assert seq.order.tolist() == [2, 0, 1]
    assert seq.positions.tolist() == [0.9, 0.5, 0.1]


def test_lambda_ties_break_on_ascending_index():
    seq = sequence_by_lambda([0.7, 0.7, 0.1])
    assert seq.order.tolist() == [2, 0, 1]


def test_lambda_accepts_numpy_array_input():
    seq = sequence_by_lambda(np.array([0.5, 0.1, 0.9]))
    assert seq.order.tolist() == [1, 0, 2]


def test_lambda_is_deterministic_across_calls():
    lambdas = [0.7, 0.3, 0.9, 0.1]
    a, b = sequence_by_lambda(lambdas), sequence_by_lambda(lambdas)
    assert a.order.tolist() == b.order.tolist()
    assert a.positions.tolist() == b.positions.tolist()


def test_lambda_output_is_a_permutation_of_item_indices():
    seq = sequence_by_lambda([0.4, 0.8, 0.1, 0.9, 0.5])
    assert sorted(seq.order.tolist()) == list(range(5))


def test_lambda_positions_align_with_order():
    lambdas = [0.4, 0.8, 0.1]
    seq = sequence_by_lambda(lambdas)
    assert seq.positions.tolist() == [lambdas[i] for i in seq.order.tolist()]


def test_lambda_rejects_fewer_than_two_items():
    with pytest.raises(ValueError):
        sequence_by_lambda([1.0])


def test_lambda_rejects_empty_input():
    with pytest.raises(ValueError):
        sequence_by_lambda([])


# --- sequence_by_graph ----------------------------------------------------

def test_graph_returns_full_permutation_of_laplacian_nodes(small_graph):
    _, gl = small_graph
    seq = sequence_by_graph(gl)
    n = np.asarray(gl.to_dense()).shape[0]
    assert len(seq.order) == n
    assert sorted(seq.order.tolist()) == list(range(n))


def test_graph_node_count_is_laplacian_matrix_not_items(small_graph):
    """Pins the index-space distinction: sequenced nodes follow the stored
    matrix (feature-space graph for eigen builds), not `gl.nnodes` (items)."""
    aspace, gl = small_graph
    seq = sequence_by_graph(gl)
    n_mat = np.asarray(gl.to_dense()).shape[0]
    assert len(seq.order) == n_mat
    assert gl.nnodes != n_mat  # 40 items vs 8 feature nodes in this fixture
    assert len(aspace.lambdas()) == gl.nnodes


def test_graph_positions_are_discovery_depths_within_bounds(small_graph):
    _, gl = small_graph
    seq = sequence_by_graph(gl)
    n = np.asarray(gl.to_dense()).shape[0]
    assert all(d >= 0.0 for d in seq.positions.tolist())
    assert max(seq.positions) <= float(n - 1)


def test_graph_dense_corpus_is_single_component(small_graph):
    # The feature co-occurrence graph over this corpus is fully connected.
    _, gl = small_graph
    seq = sequence_by_graph(gl)
    assert seq.components == 1


def test_graph_is_deterministic_across_calls(small_graph):
    _, gl = small_graph
    a, b = sequence_by_graph(gl), sequence_by_graph(gl)
    assert a.order.tolist() == b.order.tolist()
    assert a.positions.tolist() == b.positions.tolist()
    assert a.components == b.components


def test_lambda_and_graph_cover_distinct_index_spaces(small_graph):
    """λ curriculum walks item indices; graph seriation walks laplacian-matrix
    nodes. Both are valid permutations of their own space."""
    aspace, gl = small_graph
    seq_g = sequence_by_graph(gl)
    seq_l = sequence_by_lambda(aspace.lambdas())
    assert sorted(seq_l.order.tolist()) == list(range(gl.nnodes))
    n_mat = np.asarray(gl.to_dense()).shape[0]
    assert sorted(seq_g.order.tolist()) == list(range(n_mat))


def test_degenerate_single_item_build_is_refused_before_sequencing():
    """Upstream refuses 1-arrow builds (`cannot create a arrowspace of one
    arrow only`); the panic surfaces as a BaseException, never as a Sequence.
    A <2-node Laplacian therefore cannot reach `sequence_by_graph`."""
    items = np.array([[1.0, 0.0]])
    with pytest.raises(BaseException):
        _build(items, {"eps": 1.0, "k": 1, "topk": 1, "p": 2.0, "sigma": None})
