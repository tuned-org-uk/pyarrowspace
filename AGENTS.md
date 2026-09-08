# AGENTS.md — operational guide

PyO3 Python bindings (`pyarrowspace`, published to PyPI as `arrowspace`) over the
Rust crate `arrowspace` (arrowspace-rs). All index/search logic lives in Rust
(`src/`, thin `#[pyclass]`/`#[pyfunction]` wrappers); Python files are tests only.
Python >= 3.12 (local venv is 3.14).

## Build

uv-managed venv (`.venv`); maturin builds the extension. After **any** Rust edit
the installed extension is stale until rebuilt:

```bash
uv run maturin develop --release
```

`cargo check` alone does not refresh what `pytest` imports. `.venv28/` is kept
for old-version comparisons — do not touch it.

## Test protocol

Three layers, run in this order. All commands via `uv run`.

### 1. Contract suite — `pytest-tests/` (must pass)

Assert-driven pytest suite; the only suite CI runs (`pytest pytest-tests/ -v`,
ubuntu, Python 3.12 — see `.github/workflows/CI.yml`). New behaviour gets its
tests here.

```bash
uv run pytest pytest-tests/ -q
```

Conventions (match existing modules):

- Fixtures from `pytest-tests/conftest.py` (`build_graph`, `small_corpus`,
  `normal_index`); builder pattern is `with_seed(42)` +
  `with_dims_reduction(False, None)` + `with_sampling("simple", 1.0)`.
- Determinism/reproducibility contracts compare **bytewise** (`.tobytes()`),
  never with float tolerance.
- Reproducibility-sensitive failure modes are load-dependent (global rayon
  pool): if a failing condition cannot be induced through the GIL-holding
  binding, document that in the test module docstring instead of faking it.

### 2. Calibration scripts — `tests/test_0_*.py` (must pass)

Script-style (module-level asserts, not pytest). Run each:

```bash
uv run python tests/test_0_0.py   # … test_0_9.py
```

These pin value-level semantics of the indexed/searched output against the
shared calibration dataset `tests/data/eigenmaps_controlled.parquet`
(1000x128, 3 clusters; regenerate only with
`uv run python tests/data/make_datasets.py`, seed 11; rationale in
`tests/data/CALIBRATION.md`). Expectations are calibrated per arrowspace-rs
version — they may legitimately need re-tuning on a bump that changes stored
values. Prefer cluster-level invariants over exact result orderings; never
assert observed-but-brittle behaviour.

Quality gate for any index built in a test: `int(np.sum(np.abs(lam) < 1e-12))`
should be <= 1 (min-max normalisation floor). More zeros means mistuned `eps`.

### 3. Experiment scripts — `tests/test_1_*` and above

Need optional extras (`embeddings`, `benchmarks`, `viz`) and external datasets.
Not part of the verification gate.

### Rust

```bash
cargo check && cargo test
```

The crate is cdylib-only; there are no Rust unit tests here — a clean check is
the gate. Pre-existing clippy warnings exist; do not fix drive-by.

### Verification gate (any change to `src/` or Cargo.toml)

1. `uv run maturin develop --release`
2. `uv run pytest pytest-tests/ -q` — 85 passing as of arrowspace-rs 0.28.1
3. `tests/test_0_*` — 10/10
4. `cargo check && cargo test`

Quote the counts in the commit message (`Verified: test_0_* 10/10,
pytest-tests 85/85, cargo check/test clean.`), as prior bumps do.

## Bumping arrowspace-rs

1. Read the upstream release notes (`github.com/tuned-org-uk/arrowspace-rs/releases`)
   and classify the change: API surface vs value-level (last-bit / summation
   order / layout). Value-level changes are the ones that break calibration.
2. Bump `[package] version` **and** the `arrowspace` dependency in `Cargo.toml`
   together; then `cargo update -p arrowspace`.
3. If the release adds behaviour, write the contract test in `pytest-tests/`
   first (TDD) and note honestly whether the pre-fix failure is inducible.
4. Rebuild, run the full gate. If a value-level change broke `test_0_*`,
   re-verify expectations and re-tune minimally; document the rationale in the
   commit body (see `d22be93` for the pattern).
5. Commit title: `Bump arrowspace-rs to X.Y.Z`, body: what changed upstream,
   test adjustments, verification counts. One PR per bump.

## Binding gotchas

- All bound calls hold the GIL (no `allow_threads` in `src/`): Python threads
  cannot overlap binding calls, and parallelism of rayon inside a call is
  invisible from Python.
- `GraphLaplacian` surface: `nnodes`, `shape()`, `graph_params`, `to_csr()`,
  `to_dense()`. There is no `nnz()` — use the `to_csr()` buffers.
- Queries must never be exact item rows (`items[j]`): degenerate-lambda
  `ValueError`. Scale them (e.g. `items[j] * 1.2`).
- `search`/`search_batch` raise `ValueError` (not `PanicException`) for
  degenerate/non-finite/mismatched queries — assert on that, not on panics.
