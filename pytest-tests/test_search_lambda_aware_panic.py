"""Tests for #123: `search` / `search_batch` must not let a `PanicException`
cross the FFI boundary when the query maps to a degenerate lambda.

0.26.5 added `try_search_lambda_aware -> Result<_, ArrowSpaceError>` upstream
*and* `try_prepare_query_item`, but the binding still called the **infallible**
`search_lambda_aware` (which `.expect()`s the `Err` into a panic). When the
subcentroid path of `try_prepare_query_item` returns `Ok(0.0)` — a stored
subcentroid lambda of 0 with no zero-guard — the panic fires:

    pyo3_runtime.PanicException: search_lambda_aware: DegenerateLambda { raw: 0.0 }

`PanicException` bypasses `except Exception`, so a single degenerate row takes
down any full-corpus sweep.

Two-part fix:
- `search` (single): call `try_search_lambda_aware` and map `DegenerateLambda`
  to a catchable `ValueError` (restoring 0.26.4 behaviour with the better
  error type).
- `search_batch`: the energy/subcentroid path is fundamentally panic-prone
  (per-row degeneracy is undetectable at prepare time), so batch on energy
  indexes is refused with `NotImplementedError` instead of shipping a
  panic-prone path. Use `search` per-row, or `search_hybrid` /
  `search_linear_sorted`, which tolerate the degenerate case.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


@pytest.fixture(scope="module")
def energy_degenerate_index():
    """3-item corpus built with `build_energy`. Several query vectors map to a
    subcentroid whose stored lambda is 0, so `try_prepare_query_item` returns
    `Ok(0.0)` and 0.26.5 panicked in `search_lambda_aware`."""
    deg = np.array([[1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0]])
    deg /= np.linalg.norm(deg, axis=1, keepdims=True)
    gp = {"eps": 1.29, "k": 2, "topk": 3, "p": 2.0, "sigma": None}
    a, gl = ArrowSpaceBuilder().with_seed(42).build_energy(deg, None, gp)
    return a, gl, deg


# A query that maps to a zero-lambda subcentroid for this index. Confirmed
# to panic on 0.26.5 (PanicException, not ValueError).
DEGENERATE_QUERY = np.array([-1.0, 0.0, 0.0, 0.0])


def test_search_on_subcentroid_degenerate_query_raises_value_error_not_panic(
    energy_degenerate_index,
):
    """The core #123 regression for single-search: a query whose lambda is 0
    must raise a catchable ValueError, not a PanicException that bypasses
    `except Exception`."""
    a, gl, _ = energy_degenerate_index
    with pytest.raises(ValueError):
        a.search(DEGENERATE_QUERY, gl, 0.55)


def test_search_subcentroid_degenerate_error_message_mentions_eps(
    energy_degenerate_index,
):
    """The mapped error should carry the upstream `DegenerateLambda` message
    (which mentions `eps`), so users get the same guidance as 0.26.4."""
    a, gl, _ = energy_degenerate_index
    with pytest.raises(ValueError, match="eps"):
        a.search(DEGENERATE_QUERY, gl, 0.55)


def test_search_batch_on_energy_index_raises_not_implemented_not_panic(
    energy_degenerate_index,
):
    """`search_batch` on energy/subcentroid indexes is refused with
    `NotImplementedError` rather than shipping a panic-prone batch path. The
    subcentroid branch of `try_prepare_query_item` can return `Ok(0.0)` with no
    upstream guard, so per-row degeneracy is undetectable at prepare time and
    would panic inside `search_lambda_aware`. See #123."""
    a, gl, deg = energy_degenerate_index
    with pytest.raises(NotImplementedError, match="energy"):
        a.search_batch(deg, gl, 0.55)


def test_search_batch_on_eigen_index_still_works(normal_index):
    """The energy-mode refusal must not regress the eigen-mode batch path,
    which remains the supported partial-failure-tolerant batch API."""
    a, gl, items = normal_index
    results = a.search_batch(items[:5], gl, 0.55)
    assert len(results) == 5
    assert all(r is not None for r in results)
