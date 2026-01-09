# pyarrowspace

Python bindings for [`arrowspace-rs`](https://github.com/Mec-iS/arrowspace-rs).

`arrowspace` is a database for vectors supported by a graph representation and a key-value store. The main use-cases targeted are: AI search capabilities as advanced vector similarity, graph characterisation analysis and search, indexing of high-dimensional vectors. Design principles described in [this article](https://www.tuned.org.uk/posts/010_game_changer_unifying_vectors_and_features_graphs).

For labs and tests please see [tests/](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests)

## Installation
From PyPi:
```
pip install arrowspace
```
or any other way of installing a Python library.

If you have cargo installed, to compile the libraries involved (from crates.io): 
```
pip install maturin[patchelf]
maturin develop
```

## Tests
Simple test:
```
python tests/test_0.py
```
Test with public QA dataset:
```
python tests/test_1_quora_questions.py
```
There are other tests but they require downloadin a dataset separately or fine-tuning the embeddings on a given dataset. Give it a try and let me know!

## Simplest Example

```python
from arrowspace import ArrowSpaceBuilder
import numpy as np

items: np.array = np.array(
    [[0.1, 0.2, 0.3], [0.0, 0.5, 0.1], [0.9, 0.1, 0.0]],
    dtype = np.float64
)

graph_params: dict = {
    "eps": 1.0,
    "k": 6,
    "topk": 3,
    "p": 2.0,
    "sigma": 1.0,
}

# Create an ArrowSpace instance, returning the computed
# signal graph and lambdas
aspace, gl = ArrowSpaceBuilder.build(graph_params, items)

# Search comparable items
# defaults: k = nitems, alpha = 0.9, beta = 0.1
query: np.array = np.array(
    [0.05, 0.2, 0.25],
    dtype = np.float64
)

tau: float = 1.0
hits: list = aspace.search(query, gl, tau)

# Search returns a list of `(index, score`) tuples, where
# expected value from the code above show the first index
# having the top score, i.e., being nearest.

print(hits)
# [ (0, 0.989743318610787), (1, 0.7565344158360029), (2, 0.22151940739207396) ]
```
