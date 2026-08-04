"""Tests for `parse_graph_params` float coercion — GitHub issue #23.

`k` and `topk` were extracted with `extract::<usize>()`, which rejects Python
floats. The wrapper then silently substituted its own defaults, so a dict like
`{"k": 29.0, "topk": 14.0}` behaved as `{"k": 8, "topk": 3}`. These tests pin
the expected contract:

  * integral floats are coerced to the requested integer (no silent drop),
  * non-integral floats raise `TypeError`,
  * missing keys keep the documented defaults.

The exposed `gl.graph_params` reflects the post-build state, which includes the
upstream `define_result_k` heuristic (bumps `topk` when `k < 10`). Tests that
isolate the parser default for `topk` therefore use `k >= 10` so the heuristic
does not fire.
"""
import pytest

from arrowspace import ArrowSpaceBuilder


# --- integral floats are honoured, not silently replaced ---

def test_float_k_and_topk_are_honoured(build_graph):
    gp = {"eps": 1.29, "k": 29.0, "topk": 14.0, "p": 2.0, "sigma": None}
    _, gl = build_graph(gp)
    assert gl.graph_params["k"] == 29
    assert gl.graph_params["topk"] == 14


def test_float_and_int_params_produce_identical_graph_params(build_graph):
    base = {"eps": 1.29, "p": 2.0, "sigma": None}
    _, gl_f = build_graph({**base, "k": 29.0, "topk": 14.0})
    _, gl_i = build_graph({**base, "k": 29, "topk": 14})
    assert gl_f.graph_params == gl_i.graph_params


def test_float_topk_only_is_honoured(build_graph):
    gp = {"eps": 1.29, "k": 29, "topk": 9.0, "p": 2.0, "sigma": None}
    _, gl = build_graph(gp)
    assert gl.graph_params["topk"] == 9


def test_float_k_only_is_honoured(build_graph):
    gp = {"eps": 1.29, "k": 12.0, "topk": 8, "p": 2.0, "sigma": None}
    _, gl = build_graph(gp)
    assert gl.graph_params["k"] == 12


# --- non-integral floats raise, not silently round ---

def test_non_integral_float_k_raises(build_graph):
    gp = {"eps": 1.29, "k": 29.5, "topk": 14, "p": 2.0, "sigma": None}
    with pytest.raises(TypeError):
        build_graph(gp)


def test_non_integral_float_topk_raises(build_graph):
    gp = {"eps": 1.29, "k": 29, "topk": 14.5, "p": 2.0, "sigma": None}
    with pytest.raises(TypeError):
        build_graph(gp)


# --- missing keys keep defaults ---

def test_missing_k_uses_default(build_graph):
    gp = {"eps": 1.29, "topk": 3, "p": 2.0, "sigma": None}
    _, gl = build_graph(gp)
    # default k == 8; the upstream heuristic bumps topk because k < 10,
    # so only assert on k here.
    assert gl.graph_params["k"] == 8


def test_missing_topk_with_large_k_uses_default(build_graph):
    # k >= 10 so define_result_k does not adjust topk; observe the parser's
    # own default of 3 directly.
    gp = {"eps": 1.29, "k": 29, "p": 2.0, "sigma": None}
    _, gl = build_graph(gp)
    assert gl.graph_params["topk"] == 3


# --- regression: int inputs still work (no behaviour change) ---

def test_int_k_topk_unchanged(build_graph):
    gp = {"eps": 0.5, "k": 10, "topk": 5, "p": 2.0, "sigma": 0.05}
    _, gl = build_graph(gp)
    assert gl.graph_params["k"] == 10
    assert gl.graph_params["topk"] == 5
    assert gl.graph_params["eps"] == 0.5
    assert gl.graph_params["sigma"] == 0.05
