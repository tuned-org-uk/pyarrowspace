"""Tests for #25 item 2: unknown / misspelled `graph_params` keys must raise
`TypeError` rather than be silently ignored (the same failure family as #23's
silent float-drop).

A typo like `top_k` instead of `topk` currently falls back to the default
`topk=3`, silently producing a different index than configured. The parser
should reject unrecognised keys listing the offenders.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


def _items():
    items = np.random.default_rng(0).standard_normal((60, 24))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    return items


def test_misspelled_top_k_raises():
    items = _items()
    gp = {"eps": 0.5, "k": 10, "top_k": 9, "p": 2.0, "sigma": None}  # top_k typo
    with pytest.raises(TypeError):
        ArrowSpaceBuilder().with_seed(42).build(gp, items)


def test_bogus_key_raises():
    items = _items()
    gp = {"eps": 0.5, "k": 10, "topk": 9, "p": 2.0, "sigma": None, "bogus_param": 123}
    with pytest.raises(TypeError):
        ArrowSpaceBuilder().with_seed(42).build(gp, items)


def test_error_message_lists_unknown_key():
    items = _items()
    gp = {"eps": 0.5, "k": 10, "bogus": 1}
    with pytest.raises(TypeError, match="bogus"):
        ArrowSpaceBuilder().with_seed(42).build(gp, items)


def test_known_keys_still_accepted():
    items = _items()
    gp = {"eps": 0.5, "k": 10, "topk": 5, "p": 2.0, "sigma": 0.05}
    _, gl = ArrowSpaceBuilder().with_seed(42).build(gp, items)  # must not raise
    assert gl.graph_params["topk"] == 5
