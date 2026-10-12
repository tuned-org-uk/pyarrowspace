"""Contracts for arrowspace-rs v0.29 Rayleigh filter-bank outputs."""

import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


GRAPH_PARAMS = {"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}


def _builder():
    return (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    )


def test_filter_bank_default_off(build_graph):
    aspace, _ = build_graph(GRAPH_PARAMS)

    assert aspace.filter_bank_bands() == 0
    assert aspace.band_energies().shape == (aspace.nitems, 0)
    assert aspace.band_energies_for_item(0).shape == (0,)


def test_uniform_filter_bank_outputs_are_finite_and_query_ready(small_corpus):
    aspace, gl = _builder().with_filter_bank(4).build(GRAPH_PARAMS, small_corpus)

    bands = aspace.band_energies()
    assert aspace.filter_bank_bands() == 4
    assert bands.shape == (aspace.nitems, 4)
    assert np.all(np.isfinite(bands))
    assert np.all(bands >= 0.0)
    assert np.count_nonzero(np.sum(bands, axis=1) > 0.0) > 0

    lambda_q, query_bands = aspace.prepare_query_with_bands(small_corpus[3] * 1.2, gl)
    assert np.isfinite(lambda_q)
    assert query_bands.shape == (4,)
    assert np.all(np.isfinite(query_bands))
    assert np.all(query_bands >= 0.0)


def test_custom_filter_bank_edges_and_validation(small_corpus):
    aspace, _ = (
        _builder()
        .with_filter_bank(3, [0.0, 0.2, 0.8, 1.0])
        .build(GRAPH_PARAMS, small_corpus)
    )
    assert aspace.band_energies().shape == (aspace.nitems, 3)

    with pytest.raises(ValueError, match="strictly increasing"):
        _builder().with_filter_bank(3, [0.0, 0.5, 0.4, 1.0])
