"""Tests for #25 item 1: `search_batch` must not abort the entire batch when a
single row is degenerate (lambda = 0).

Previously a degenerate row caused `return Err(...)`, discarding all already-
scored rows — unusable for whole-corpus work where ~1 row is typically
degenerate. The new contract returns `list[list[(idx, score)] | None]`: `None`
for degenerate rows, results for the rest. The single-query `search` still
raises (correct — it is the batch semantics that need partial-failure
tolerance).
"""
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


def test_batch_with_one_degenerate_row_returns_none_for_that_row_only(normal_index):
    a, gl, items = normal_index
    # row 1 is the zero-vector -> degenerate; rows 0 and 2 are valid
    batch = np.vstack([items[0], np.zeros(48), items[1]])
    results = a.search_batch(batch, gl, 0.55)
    assert len(results) == 3
    assert results[0] is not None, "valid row 0 was dropped"
    assert results[1] is None, "degenerate row 1 should be None, not raise"
    assert results[2] is not None, "valid row 2 was dropped"
    assert len(results[0]) > 0


def test_batch_all_valid_returns_no_none(normal_index):
    a, gl, items = normal_index
    results = a.search_batch(items[:10], gl, 0.55)
    assert len(results) == 10
    assert all(r is not None for r in results)


def test_batch_all_degenerate_returns_all_none():
    deg = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]])
    deg /= np.linalg.norm(deg, axis=1, keepdims=True)
    a, gl = ArrowSpaceBuilder().with_seed(42).build(
        {"eps": 1.29, "k": 2, "topk": 3, "p": 2.0, "sigma": None}, deg
    )
    results = a.search_batch(deg, gl, 0.55)
    assert len(results) == 3
    assert results == [None, None, None]


def test_batch_degenerate_row_does_not_discard_already_scored_rows(normal_index):
    """The issue's core complaint: 823 scored rows thrown away because row 824
    was degenerate. Verify a degenerate row in position 0 does not abort the
    rest, and one in the last position keeps the earlier results."""
    a, gl, items = normal_index
    # degenerate first, then 4 valid
    batch = np.vstack([np.zeros(48), items[0], items[1], items[2], items[3]])
    results = a.search_batch(batch, gl, 0.55)
    assert results[0] is None
    assert all(r is not None for r in results[1:])
    # degenerate last, after 4 valid
    batch2 = np.vstack([items[0], items[1], items[2], items[3], np.zeros(48)])
    results2 = a.search_batch(batch2, gl, 0.55)
    assert all(r is not None for r in results2[:4])
    assert results2[4] is None
