# pyarrowspace

Python bindings for [`arrowspace-rs`](https://github.com/Mec-iS/arrowspace-rs).

`arrowspace` is a graph-based analytics library for vector spaces, supported by a graph representation and a key-value store. The main use-cases targeted are: AI search capabilities as advanced vector similarity, graph characterisation analysis and search, indexing of high-dimensional vectors. Design principles described in [this article](https://www.tuned.org.uk/posts/010_game_changer_unifying_vectors_and_features_graphs).

For labs and tests please see [tests/](https://github.com/tuned-org-uk/pyarrowspace/tree/main/tests)

## Installation

### Core library

The core library ships as a compiled Rust extension. It only requires `numpy`, `pyarrow`, `pandas`, and `scikit-learn`:

```
pip install arrowspace
```

### Optional extras

Additional capabilities are available as optional extras:

| Extra | Packages included | Use case |
|---|---|---|
| `embeddings` | `datasets`, `sentence-transformers`, `transformers[torch]<5.0`, `tsdae` | Generating embeddings from text via HuggingFace models; required for `test_1_quora_questions.py` and all tests above |
| `benchmarks` | `beir`, `nltk` | Running BEIR/MS-MARCO benchmark suites and NLP preprocessing |
| `viz` | `matplotlib`, `seaborn`, `tqdm` | Plotting results and progress bars in test scripts |
| `full` | all of the above | Full development and research environment |

Install with one or more extras:

```bash
# Required to run test_1_quora_questions.py and above
pip install arrowspace[embeddings]

# Benchmarking + visualisation
pip install arrowspace[benchmarks,viz]

# Everything (for running the full test suite)
pip install arrowspace[full]
```

> **Note:** `tests/test_0_*.py` only require the core install. All tests numbered `test_1` and above require at minimum `arrowspace[embeddings]`.

### Build from source

If you have Cargo installed, you can compile and install locally using [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin[patchelf]
# quick development build
maturin develop
# optimised release build (recommended for large datasets)
maturin develop --release
```

## Tests

`test_0_*.py` scripts only require the core install:
```
python tests/test_0_0.py
```

`test_1` and above require the `embeddings` extra (`pip install arrowspace[embeddings]`):
```
python tests/test_1_quora_questions.py
```

Higher-numbered tests (`test_3` and above) additionally require `benchmarks` and `viz`:
```
pip install arrowspace[full]
python tests/test_3_beir.py
```

Some tests require downloading a dataset separately or fine-tuning embeddings on a given dataset.

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
aspace, gl = ArrowSpaceBuilder().build(graph_params, items)

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
