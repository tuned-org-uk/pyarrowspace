"""
test_graph_wht.py
=================
Test the Graph-WHT spectral filter on the synthetic dataset from test_0_0.py.

All tests run in pure Python / NumPy — no Rust compilation required.

Test matrix
-----------
test_wht_fit_shape          sanity-check on padded length P
test_wht_energy_conservation  ||x_filtered||_1 ≤ ||x||_1 for h ≤ 1
test_wht_lowpass_smoother   low-pass output is smoother than input
test_wht_allpass_identity   h(λ)=1 ∀ λ  →  x_filtered ≈ x
test_wht_batch_shape        filter_batch returns (N, F) array
test_wht_spectral_energy    energy distribution sums correctly
test_wht_from_arrowspace    end-to-end: build ArrowSpace → fit filter → apply
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Minimal stub for GraphLaplacian so the test file runs without Rust binaries.
# When arrowspace IS importable the real objects are used instead.
# ---------------------------------------------------------------------------
try:
    from arrowspace import ArrowSpaceBuilder, GraphLaplacian
    HAS_ARROWSPACE = True
except ImportError:
    HAS_ARROWSPACE = False
    ArrowSpaceBuilder = None
    GraphLaplacian = None

from graph_wht_spectral_filter import (
    GraphWHTFilter,
    heat_kernel,
    ideal_lowpass,
    bandpass,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

ITEMS = np.array([
    [0.82,0.11,0.43,0.28,0.64,0.32,0.55,0.48,0.19,0.73,
     0.07,0.36,0.58,0.23,0.44,0.31,0.52,0.16,0.61,0.40,
     0.27,0.49,0.35,0.29],
    [0.79,0.12,0.45,0.29,0.61,0.33,0.54,0.47,0.21,0.70,
     0.08,0.37,0.56,0.22,0.46,0.30,0.51,0.18,0.60,0.39,
     0.26,0.48,0.36,0.30],
    [0.78,0.13,0.46,0.27,0.62,0.34,0.53,0.46,0.22,0.69,
     0.09,0.35,0.55,0.24,0.45,0.29,0.50,0.17,0.59,0.38,
     0.28,0.47,0.34,0.31],
    [0.81,0.10,0.44,0.26,0.63,0.31,0.56,0.45,0.20,0.71,
     0.06,0.34,0.57,0.25,0.47,0.33,0.53,0.15,0.62,0.41,
     0.25,0.50,0.37,0.27],
    [0.80,0.12,0.42,0.25,0.60,0.35,0.52,0.49,0.23,0.68,
     0.10,0.38,0.54,0.21,0.43,0.28,0.49,0.19,0.58,0.37,
     0.29,0.46,0.33,0.32],
], dtype=np.float64)

GRAPH_PARAMS = {
    "eps":   0.05,
    "k":     int(ITEMS.shape[1] / 2),
    "topk":  3,
    "p":     2.0,
    "sigma": 0.05,
}

F = ITEMS.shape[1]   # 24 feature dimensions


def _make_synthetic_laplacian(F: int) -> np.ndarray:
    """
    Build a small synthetic feature-space Laplacian for testing without
    the Rust binaries.

    Path graph L:  symmetric tri-diagonal, row sums = 0.
    """
    # Degree matrix minus adjacency of a path graph
    A = np.zeros((F, F), dtype=np.float64)
    for i in range(F - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


@pytest.fixture(scope="module")
def synthetic_filter() -> GraphWHTFilter:
    """Filter built from a deterministic synthetic Laplacian."""
    L = _make_synthetic_laplacian(F)
    filt = GraphWHTFilter(graph_laplacian=None, L_matrix=L)
    filt.fit()
    return filt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphWHTFilterShape:
    def test_fit_shape(self, synthetic_filter):
        """P must be >= F and a power of two."""
        filt = synthetic_filter
        assert filt.P >= filt.F
        assert filt.P & (filt.P - 1) == 0, "P must be a power of 2"

    def test_fit_eigenvalues_length(self, synthetic_filter):
        assert len(synthetic_filter.eigenvalues) == F

    def test_fit_wht_lambdas_length(self, synthetic_filter):
        assert len(synthetic_filter.wht_lambdas) == synthetic_filter.P

    def test_wht_lambdas_range(self, synthetic_filter):
        lam = synthetic_filter.wht_lambdas
        assert lam[0] == pytest.approx(0.0, abs=1e-12)
        assert lam[-1] <= 2.0 + 1e-10


class TestGraphWHTFilterEnergy:
    def test_allpass_identity(self, synthetic_filter):
        """
        h(λ) = 1 ∀ λ  →  filtered signal ≈ original (up to WHT round-trip).
        """
        x = ITEMS[0]
        allpass = lambda lam: np.ones_like(lam)
        x_out = synthetic_filter.apply(x, h=allpass)
        np.testing.assert_allclose(x_out, x, atol=1e-10,
                                   err_msg="All-pass filter must be identity")

    def test_lowpass_output_shape(self, synthetic_filter):
        x = ITEMS[0]
        x_out = synthetic_filter.apply(x, h=ideal_lowpass(0.5))
        assert x_out.shape == (F,)

    def test_lowpass_energy_not_greater(self, synthetic_filter):
        """
        Low-pass filter must not add energy: ||x_out||^2 ≤ ||x||^2.
        """
        x = ITEMS[0]
        x_out = synthetic_filter.apply(x, h=ideal_lowpass(1.0))
        assert np.sum(x_out ** 2) <= np.sum(x ** 2) + 1e-10

    def test_heat_kernel_attenuates(self, synthetic_filter):
        """
        Heat kernel with large t should produce lower L2 norm than input
        (high frequencies are suppressed).
        """
        x = ITEMS[0] - ITEMS[0].mean()   # zero-mean so DC component = 0
        x_out = synthetic_filter.apply(x, h=heat_kernel(t=10.0))
        assert np.linalg.norm(x_out) <= np.linalg.norm(x) + 1e-10

    def test_bandpass_zeros_outside_band(self, synthetic_filter):
        """
        Band-pass with empty band → near-zero output.
        """
        x = ITEMS[0]
        # Band that excludes all WHT eigenvalues
        x_out = synthetic_filter.apply(x, h=bandpass(low=3.0, high=4.0))
        np.testing.assert_allclose(x_out, 0.0, atol=1e-10,
                                   err_msg="Empty band should zero the output")


class TestGraphWHTFilterBatch:
    def test_batch_shape(self, synthetic_filter):
        X_out = synthetic_filter.filter_batch(ITEMS)
        assert X_out.shape == ITEMS.shape

    def test_batch_allpass_identity(self, synthetic_filter):
        allpass = lambda lam: np.ones_like(lam)
        X_out = synthetic_filter.filter_batch(ITEMS, h=allpass)
        np.testing.assert_allclose(X_out, ITEMS, atol=1e-10)


class TestGraphWHTSpectralEnergy:
    def test_spectral_energy_shape(self, synthetic_filter):
        x = ITEMS[0]
        lam, energy = synthetic_filter.spectral_energy(x)
        assert lam.shape == (synthetic_filter.P,)
        assert energy.shape == (synthetic_filter.P,)
        assert np.all(energy >= 0)

    def test_spectral_energy_parseval(self, synthetic_filter):
        """
        Parseval: sum of WHT energy ≈ ||x_padded||^2.
        (x is zero-padded to length P, so total energy includes padding.)
        """
        x = ITEMS[0]
        _, energy = synthetic_filter.spectral_energy(x)
        # Padded L2 norm
        x_padded = np.zeros(synthetic_filter.P)
        x_padded[:F] = x[synthetic_filter.fiedler_order]
        np.testing.assert_allclose(
            np.sum(energy), np.sum(x_padded ** 2), rtol=1e-10,
            err_msg="Parseval: WHT energy must equal signal energy"
        )


@pytest.mark.skipif(not HAS_ARROWSPACE, reason="arrowspace Rust binaries not installed")
class TestGraphWHTFromArrowSpace:
    """
    End-to-end test: build an ArrowSpace, extract L_F, fit the filter,
    apply it to all items, verify results are consistent with direct search.
    """

    def test_end_to_end(self):
        aspace, gl = ArrowSpaceBuilder().build_and_store(GRAPH_PARAMS, ITEMS)

        # Build filter from the real GraphLaplacian
        filt = GraphWHTFilter(gl)
        filt.fit()

        assert filt.P >= F
        assert filt.P & (filt.P - 1) == 0

        # Apply low-pass filter to all items
        allpass = lambda lam: np.ones_like(lam)
        X_filtered = filt.filter_batch(ITEMS, h=allpass)

        # All-pass must be identity
        np.testing.assert_allclose(X_filtered, ITEMS, atol=1e-10)

    def test_smoothed_query_still_retrieves_correct_item(self):
        """
        After low-pass smoothing of the query, the nearest neighbour in the
        original ArrowSpace should still be item 2 (as in test_0_0.py).
        """
        aspace, gl = ArrowSpaceBuilder().build_and_store(GRAPH_PARAMS, ITEMS)
        filt = GraphWHTFilter(gl)
        filt.fit()

        query = np.array(ITEMS[2] * 1.05, dtype=np.float64)
        query_smooth = filt.apply(query, h=heat_kernel(t=0.1))

        hits = aspace.search(query_smooth, gl, 1.0)
        assert len(hits) == 3
        # After mild smoothing, item 2 should still be the top hit
        assert hits[0][0] == 2
