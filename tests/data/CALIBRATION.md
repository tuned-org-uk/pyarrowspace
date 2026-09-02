# CALIBRATION — synthetic dataset for the `tests/test_0_*.py` scripts

Empirical findings from calibrating a shared controlled dataset against
`arrowspace==0.27.2` (pyarrowspace build, CPython 3.13, macOS arm64).

Final dataset: **`eigenmaps_controlled.parquet` — 1000 items x 128 dims**,
3 orthonormal clusters of increasing sparsity, with a `cluster` label column.
Regenerate with `uv run python tests/data/make_datasets.py` (seed 11).

## 1. Why the old tests failed

| Test | Failure | Root cause |
|---|---|---|
| `test_0_0`, `test_0_5` | `assert(hits[0][0] == 4)` at `tau<=0.9` | 5×24 dataset of near-identical rows → lambda spread is numerical noise, so the spectral term never reorders results. Expectations encoded 0.26-era lambdas; file last touched at v0.26.0. |
| `test_0_9` | `assert(len(hits) == 3)` | Contradicted build params `topk: 5`; 0.27.x correctly returns 5 hits. |

All three failed identically on 0.27.0 and 0.27.2 — pre-existing staleness,
not a regression from the version bump. The CI suite (`pytest-tests/`) passes
66/66 on 0.27.2.

## 2. 0.27.x scoring semantics (measured)

- `search(q, gl, tau)` score = `tau·cos(q, item) + (1−tau)·spec(item)`
  (**blend model confirmed exactly**: predicted vs actual orders matched on
  every probed config).
- `spec` is **not** `λi` directly: the query is placed into the graph and gets
  its **own lambda** `λq` (`prepare_query_item`); `spec` measures spectral
  closeness between `λq` and the item's `λi`. The shape is nonlinear — do not
  model it, measure it: `search(q, gl, 0.0, k=n)` (tau=0 ⇒ score IS spec).
- `q = items[j] * scale` is cosine-invariant (`cos(q, items[j]) = 1.0` for any
  `scale > 0`), but `λq` moves with `scale` — the only query-side tuning knob.
- `q = items[j]` exactly (scale 1.0) raises `ValueError: degenerate lambda
  (raw=0.000000)`. Always scale the query (e.g. 1.05–1.2).
- `aspace.lambdas()` are min-max normalized: exactly one item shows `0.0`
  (the floor). Harmless — the degenerate gate only fires for a query's own
  lambda. Quality gate: `count(λ < 1e-12) <= 1` for these datasets.
- `eps=0.05` (the old tests' value) leaves high-dim unit-norm data
  near-disconnected. These datasets use `eps=0.5, sigma=0.5`.
- Dense noise σ drives λ per item, roughly quadratically; sparse 2-dim spikes
  give far less λ per L² (L²=0.078 → λ=0.032 vs dense σ=0.02 curve).
- Lifting an item past a cos-1.0 self-anchor at tau=0.9 needs
  `0.1·Δspec > 0.9·(1−cos)` — a 9× barrier; naive twin-item designs sit right
  on it (see sweep log). Cluster-level invariants are far more robust.

## 3. Final dataset design (1000 x 128, seed 11)

3 clusters with orthonormal centroids (QR of random Gaussian), unit-norm rows:

| cluster | items | noise σ | mean λ (build_and_store) |
|---|---|---|---|
| 0 (dense)   | 400 | 0.02 | 0.0451 |
| 1 (medium)  | 350 | 0.05 | 0.1109 |
| 2 (sparse)  | 250 | 0.12 | 0.1372 |

Graph params for all variants: `eps=0.5, k=12, p=2.0, sigma=0.5`;
`topk=3` (tests 0_0, 0_5) / `topk=5` (test 0_9).

Query: `items[0] * 1.2` (dense-cluster member, scaled — never exact).

### Verified invariants (all three builder variants)

- `build_and_store` and `build_full` produce identical results.
- Sampled builder must use `with_sampling("simple", 1.0)` for a small
  dataset (0.6 changes the graph; 1.0 keeps all items).
- Cluster containment: every hit at every `tau ∈ {1.0, 0.9, 0.6, 0.55}` is
  from the query's own cluster.
- Self item ranks first at every tau.
- Mean λ increases with cluster sparsity: `λ̄₀ < λ̄₁ < λ̄₂` (the spectral
  statistic tracks structural density).
- Exactly one min-max zero λ.
- Bonus: within the dense cluster, tau ≤ 0.6 reorders the runner-ups
  (spectral blending active) — observed, not asserted (brittle).

## 4. Sweep log (calibration journey, kept for reference)

| Stage | Dataset | Outcome |
|-------|---------|---------|
| probe (d=24) | 5 handcrafted rows | spec ≈ λ-closeness to query's λ discovered; exact-self query is degenerate |
| v2–v7 (d=60) | 5 twin/filler rows | 9× cosine-economics barrier; min-max λ floor; fillers hijack at scale ≥ 1.6; linear spec model insufficient at large offsets |
| measurement | d=60 twins | blend model `tau·cos + (1−tau)·spec` CONFIRMED via tau=0 spec extraction; winner found for [2,1,4]→[1,2,4] |
| final | 1000×128 clusters | cluster-level invariants: robust on every builder variant; adopted |

## 5. File inventory

- `eigenmaps_controlled.parquet` — the shared dataset (features `f0..f127` + `cluster`).
- `make_datasets.py` — deterministic generator.
- `CALIBRATION.md` — this document.
