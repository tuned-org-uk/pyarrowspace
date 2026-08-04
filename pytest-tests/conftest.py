"""Shared fixtures for the pyarrowspace pytest suite.

The legacy `tests/` directory holds experiment scripts; this `pytest-tests/`
directory holds collected, assert-driven tests. Build the extension with
`maturin develop --release` before running `pytest pytest-tests/`.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


@pytest.fixture(scope="session")
def small_corpus():
    """120 unit-normalised 48-d vectors — deterministic across the session."""
    rng = np.random.default_rng(0)
    items = rng.standard_normal((120, 48))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return items


def _builder():
    return (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    )


@pytest.fixture
def build_graph(small_corpus):
    """Return a function that builds (aspace, gl) for a given graph-params dict.

    Asserts the build succeeded so a silent failure surfaces as a test error
    rather than a confusing downstream assertion miss.
    """
    def _build(gp):
        aspace, gl = _builder().build(gp, small_corpus)
        assert gl is not None
        return aspace, gl
    return _build
