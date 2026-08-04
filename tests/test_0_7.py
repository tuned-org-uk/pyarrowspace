"""Persistence round-trip: build an index to a known address, then reload it.

Previously build_and_store picked a random `dataset_<uuid>` name the caller
never saw, so reload had to guess which file was theirs (issue #17). With
the storage_path/dataset_name kwargs, the write and reload paths agree.
"""
from pathlib import Path
import numpy as np
from arrowspace import ArrowSpaceBuilder, load_arrowspace

graph_params = {"eps": 0.97, "k": 21, "topk": 10, "p": 2.0, "sigma": 0.1}

storage = Path(__file__).parent.parent / "storage"
storage.mkdir(exist_ok=True)

items = np.random.default_rng(42).standard_normal((40, 16))
items /= np.linalg.norm(items, axis=1, keepdims=True)

print("Building and storing ArrowSpace with explicit dataset_name...")
aspace, gl = ArrowSpaceBuilder().with_seed(42).build_and_store(
    graph_params, items.astype(np.float64),
    storage_path=str(storage), dataset_name="test_0_7_index",
)
print(f"  stored: {aspace.dataset_name} at {aspace.storage_path}")
assert aspace.dataset_name == "test_0_7_index"
assert aspace.nitems == 40

print("Reloading ArrowSpace from storage...")
aspace2, gl2 = load_arrowspace(
    storage_path=str(storage),
    dataset_name="test_0_7_index",
    graph_params=graph_params,
    energy=False,
)
print(f"  reloaded: {aspace2.nitems} items × {aspace2.nfeatures} features")
assert aspace2.nitems == aspace.nitems
assert aspace2.nfeatures == aspace.nfeatures

print("Round-trip OK")
