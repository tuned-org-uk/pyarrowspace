"""Tests for #25 item 3: `search` / `search_batch` should accept an optional
`k` override so the number of retrieved neighbours can change without
rebuilding the index (which conflates the construction hyperparameter `topk`
with the retrieval parameter `k`).

New signature: `search(item, gl, tau, k=None)` / `search_batch(items, gl, tau,
k=None)`, defaulting to `graph_params.topk`. Backwards compatible.
"""
import inspect
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


@pytest.fixture(scope="module")
def normal_index():
    items = np.random.default_rng(0).standard_normal((120, 48))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    a, gl = (
        ArrowSpaceBuilder().with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build({"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}, items)
    return a, gl, items


def test_search_accepts_k_kwarg(normal_index):
    a, gl, items = normal_index
    sig = inspect.signature(a.search)
    assert "k" in sig.parameters, f"search has no k parameter: {sig}"


def test_search_batch_accepts_k_kwarg(normal_index):
    a, gl, items = normal_index
    sig = inspect.signature(a.search_batch)
    assert "k" in sig.parameters, f"search_batch has no k parameter: {sig}"


def test_search_k_override_changes_result_count(normal_index):
    a, gl, items = normal_index
    default = a.search(items[0], gl, 0.55)
    fewer = a.search(items[0], gl, 0.55, k=5)
    assert len(fewer) <= 5
    assert len(fewer) != len(default) or len(default) <= 5


def test_search_k_none_defaults_to_topk(normal_index):
    a, gl, items = normal_index
    explicit_none = a.search(items[0], gl, 0.55, k=None)
    default = a.search(items[0], gl, 0.55)
    assert explicit_none == default


def test_search_batch_k_override_applies_to_all_rows(normal_index):
    a, gl, items = normal_index
    results = a.search_batch(items[:4], gl, 0.55, k=3)
    assert all(r is not None for r in results)
    assert all(len(r) <= 3 for r in results)
