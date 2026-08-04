"""Tests for #25 item 4: Rust panics must not cross the FFI boundary as
PanicException. `search` should raise a typed, catchable `ValueError`
(documented by the existing guard at lib.rs:246) instead of the upstream
`prepare_query_item` panic.

The upstream `arrowspace 0.26.5` added `try_prepare_query_item` returning
`Result<f64, ArrowSpaceError>` precisely so FFI bindings can surface catchable
exceptions; these tests pin that the binding uses it.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


@pytest.fixture(scope="module")
def degenerate_index():
    """3-item corpus where every lambda is ~0 (mis-tuned eps)."""
    deg = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]])
    deg /= np.linalg.norm(deg, axis=1, keepdims=True)
    a, gl = ArrowSpaceBuilder().with_seed(42).build(
        {"eps": 1.29, "k": 2, "topk": 3, "p": 2.0, "sigma": None}, deg
    )
    return a, gl, deg


def test_search_on_degenerate_query_raises_value_error_not_panic(degenerate_index):
    a, gl, deg = degenerate_index
    with pytest.raises(ValueError):
        a.search(deg[0], gl, 0.55)


def test_search_degenerate_error_message_mentions_eps(degenerate_index):
    a, gl, deg = degenerate_index
    with pytest.raises(ValueError, match="eps"):
        a.search(deg[0], gl, 0.55)


def test_search_zero_vector_query_raises_value_error():
    """A zero-vector query is degenerate against a normal index too."""
    items = np.random.default_rng(0).standard_normal((120, 48))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    a, gl = (
        ArrowSpaceBuilder().with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build({"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}, items)
    with pytest.raises(ValueError):
        a.search(np.zeros(48), gl, 0.55)


def test_search_non_finite_query_raises_value_error():
    """NaN in the query should raise ValueError, not panic."""
    items = np.random.default_rng(0).standard_normal((120, 48))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    a, gl = (
        ArrowSpaceBuilder().with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build({"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}, items)
    q = items[0].copy()
    q[0] = np.nan
    with pytest.raises(ValueError):
        a.search(q, gl, 0.55)
