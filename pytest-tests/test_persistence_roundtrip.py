"""Tests for #25 item 5 / #17: persistence must be addressable so a built index
can be reloaded after a process/container restart.

#17 documents three stacked bugs: `with_persistence` is not exposed to Python,
`build_and_store` writes to a hardcoded CWD/`storage` path with a random UUID
name the caller never sees, and the reload path looks in a different location.
#25 item 5 is the narrower addressing gap in the method that does persist today.

These tests pin the round-trip contract:
  * `with_persistence(path, dataset_name)` is exposed on the builder,
  * `build_and_store(graph_params, items, path=None, dataset_name=None)` accepts
    explicit addressing and surfaces the resolved `(storage_path, dataset_name)`
    on the returned objects so the caller can reload its own artifacts,
  * a built index round-trips through `load_arrowspace`.
"""
import inspect
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder, load_arrowspace


@pytest.fixture
def items():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


GP = {"eps": 0.5, "k": 8, "topk": 4, "p": 2.0, "sigma": None}


def test_builder_exposes_with_persistence():
    assert hasattr(ArrowSpaceBuilder(), "with_persistence")


def test_build_and_store_accepts_path_and_dataset_name():
    sig = inspect.signature(ArrowSpaceBuilder().build_and_store)
    assert "storage_path" in sig.parameters, f"build_and_store has no storage_path: {sig}"
    assert "dataset_name" in sig.parameters, f"build_and_store has no dataset_name: {sig}"


def test_build_and_store_surfaces_resolved_addressing(tmp_path, items):
    a, gl = (
        ArrowSpaceBuilder().with_seed(42).build_and_store(
            GP, items, storage_path=str(tmp_path), dataset_name="roundtrip_test"
        )
    )
    # the resolved name must be visible to the caller so reload is possible
    assert getattr(a, "dataset_name", None) == "roundtrip_test"
    assert str(tmp_path) in getattr(a, "storage_path", "")


def test_build_and_store_then_load_roundtrips(tmp_path, items):
    a, gl = ArrowSpaceBuilder().with_seed(42).build_and_store(
        GP, items, storage_path=str(tmp_path), dataset_name="rt"
    )
    a2, gl2 = load_arrowspace(str(tmp_path), "rt", GP, False)
    assert a2.nitems == a.nitems
    assert a2.nfeatures == a.nfeatures


def test_with_persistence_then_build_roundtrips(tmp_path, items):
    """The #17 Option A path: expose with_persistence, build() writes, reload reads."""
    a, gl = (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_persistence(str(tmp_path), "persist_test")
        .build(GP, items)
    )
    a2, gl2 = load_arrowspace(str(tmp_path), "persist_test", GP, False)
    assert a2.nitems == a.nitems
    assert a2.nfeatures == a.nfeatures
