"""Eigen lambda-tau bit-reproducibility (arrowspace-rs #170, fixed in 0.28.1).

Building the same EigenMaps index twice from the same items with a fixed
seed must produce bit-identical lambda-tau scores. Before 0.28.1 the Eigen
lambda synthesis used a schedule-dependent par_bridge reduction: the order
in which rayon threads joined their partial reductions varied run to run,
so the last mantissa bits of the normalised lambdas could differ between
builds of an identical corpus.

A tolerance-based comparison would silently pass on the pre-fix build —
the divergence this test reproduces lives in the last bits, so all
comparisons are bytewise (ndarray.tobytes()).

Failure mode note: pre-0.28.1 the grouping of the f64 addends depended on
rayon work-stealing, so divergence appeared when the global rayon pool was
busy (upstream reproduced it by saturating the pool from Rust while
measuring). Through the GIL-holding Python binding that condition cannot
be provoked at will, so on an idle machine this suite can pass against
pre-fix builds; it pins the post-0.28.1 guarantee, which holds by
construction on any machine, any pool size, any load.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder

GRAPH_PARAMS = {"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}
N_BUILDS = 8


def _build_once(items):
    return (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build(GRAPH_PARAMS, items)


@pytest.fixture(scope="module")
def rebuilds():
    """600x64 unit-normalised corpus large enough that the parallel lambda
    pass fans out over many rayon batches; (aspace, gl) tuples from N_BUILDS
    identical builds of it."""
    rng = np.random.default_rng(7)
    items = rng.standard_normal((600, 64))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return [_build_once(items) for _ in range(N_BUILDS)]


def test_eigen_lambdas_bit_identical_across_rebuilds(rebuilds):
    reference = rebuilds[0][0].lambdas().tobytes()
    for i, (aspace, _) in enumerate(rebuilds[1:], start=1):
        assert aspace.lambdas().tobytes() == reference, (
            f"build {i} produced different lambda bytes than build 0 "
            "(schedule-dependent par_bridge reduction, arrowspace-rs #170)"
        )


def test_eigen_lambda_ordering_identical_across_rebuilds(rebuilds):
    reference = rebuilds[0][0].lambdas_sorted()
    for i, (aspace, _) in enumerate(rebuilds[1:], start=1):
        assert aspace.lambdas_sorted() == reference, (
            f"build {i} produced a different lambda ordering than build 0"
        )


def test_eigen_laplacian_bit_identical_across_rebuilds(rebuilds):
    data, indices, indptr, shape = rebuilds[0][1].to_csr()
    reference = (data.tobytes(), indices.tobytes(), indptr.tobytes(), shape)
    for i, (_, gl) in enumerate(rebuilds[1:], start=1):
        d, ix, ip, s = gl.to_csr()
        assert (d.tobytes(), ix.tobytes(), ip.tobytes(), s) == reference, (
            f"build {i} produced a different Laplacian than build 0"
        )
