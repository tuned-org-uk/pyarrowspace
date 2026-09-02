"""Tests for `parse_motives_config` and `parse_subgraph_config` integer
coercion — same class of bug as issue #23, in the motives/subgraph config
parsers.

The motives parser extracted `usize` fields with `.ok().unwrap_or(default)`,
so a float `max_motif_size=8.0` was silently dropped to the default `32` —
no error, but the caller's bound was ignored. The subgraph parser used
`v.extract()?`, which raised on any float (loud, but inconsistent with the
graph-params contract: `4.0` should be accepted like `4`).

`spot_motives_eigen` is non-deterministic across repeated calls (unseeded
tie-breaking in motif seeding), so motif *membership* cannot be compared
between calls. Instead these tests use deterministic observables:

  * `max_motif_size` bounds every motif's length — verifiable regardless of
    which nodes seed a motif;
  * the raise/accept contract (non-integral → TypeError, integral → no raise)
    is deterministic.

The `max_motif_size` bound is the corruption oracle: with the bug, a float
`8.0` was dropped to default `32` and motifs exceeded 8; with the fix they
are bounded at 8.
"""
import numpy as np
import pytest

from arrowspace import ArrowSpaceBuilder


@pytest.fixture(scope="module")
def corpus_and_gl():
    items = np.random.default_rng(0).standard_normal((120, 48))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    a, gl = (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    ).build({"eps": 1.29, "k": 29, "topk": 14, "p": 2.0, "sigma": None}, items)
    return a, gl


def _max_motif_len(motifs):
    return max((len(m) for m in motifs), default=0)


# --- integral floats are coerced (deterministic observable: motif size bound) ---

def test_float_max_motif_size_bounds_motifs_like_int(corpus_and_gl):
    """max_motif_size=8.0 (float) must bound motifs at 8, not the default 32."""
    a, gl = corpus_and_gl
    r = a.spot_motives_eigen(gl, {"max_motif_size": 8.0})
    assert _max_motif_len(r) <= 8, "float max_motif_size=8.0 was silently dropped to default 32"


def test_float_max_motif_size_equal_int_behaviour(corpus_and_gl):
    """float and int of the same value bound motifs identically."""
    a, gl = corpus_and_gl
    r_int = a.spot_motives_eigen(gl, {"max_motif_size": 8})
    r_float = a.spot_motives_eigen(gl, {"max_motif_size": 8.0})
    assert _max_motif_len(r_int) == _max_motif_len(r_float) == 8


def test_default_max_motif_size_is_32(corpus_and_gl):
    """Oracle guard: missing key uses default 32, so motifs reach 32 —
    confirming the bound test above actually distinguishes coercion from drop."""
    a, gl = corpus_and_gl
    r = a.spot_motives_eigen(gl, {})
    # default corpus yields motifs of size 32 with the default bound
    assert _max_motif_len(r) == 32


# --- integral floats accepted for every usize field (no raise) ---

@pytest.mark.parametrize("cfg", [
    {"top_l": 4.0},
    {"min_triangles": 3.0},
    {"max_sets": 32.0},
])
def test_integral_float_usize_fields_accepted(corpus_and_gl, cfg):
    a, gl = corpus_and_gl
    a.spot_motives_eigen(gl, cfg)  # must not raise


# --- non-integral floats raise TypeError ---

def test_non_integral_top_l_raises(corpus_and_gl):
    a, gl = corpus_and_gl
    with pytest.raises(TypeError):
        a.spot_motives_eigen(gl, {"top_l": 4.5})


def test_non_integral_min_triangles_raises(corpus_and_gl):
    a, gl = corpus_and_gl
    with pytest.raises(TypeError):
        a.spot_motives_eigen(gl, {"min_triangles": 2.5})


def test_non_integral_max_motif_size_raises(corpus_and_gl):
    a, gl = corpus_and_gl
    with pytest.raises(TypeError):
        a.spot_motives_eigen(gl, {"max_motif_size": 8.5})


def test_non_integral_max_sets_raises(corpus_and_gl):
    a, gl = corpus_and_gl
    with pytest.raises(TypeError):
        a.spot_motives_eigen(gl, {"max_sets": 32.5})


# --- missing keys keep defaults (no raise, deterministic: default bound) ---

def test_missing_max_motif_size_uses_default_bound(corpus_and_gl):
    a, gl = corpus_and_gl
    r = a.spot_motives_eigen(gl, {"max_motif_size": 32})  # explicit default
    r_default = a.spot_motives_eigen(gl, {})              # missing -> default
    assert _max_motif_len(r) == _max_motif_len(r_default) == 32


# --- subgraph config: min_size is usize, should accept integral floats ---
#
# Since #35 (and arrowspace-rs 0.27.3), `spot_subg_motives` requires an
# EnergyMaps build — on EigenMaps it raises ValueError before cfg semantics
# can be observed. These tests therefore run against an energy build.


@pytest.fixture(scope="module")
def energy_corpus_and_gl():
    """Structured 48×120 corpus: four tight clusters so the energy pipeline
    yields at least one subgraph, keeping the coercion oracle non-vacuous."""
    rng = np.random.default_rng(3407)
    centers = np.zeros((4, 120))
    for c in range(4):
        centers[c, c * 20:(c + 1) * 20] = 0.85
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    items = np.vstack(
        [centers[c] + 0.05 * rng.standard_normal((12, 120)) for c in range(4)]
    )
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    a, gl = (
        ArrowSpaceBuilder()
        .with_seed(3407)
        .build_energy(
            items,
            {
                "optical_tokens": None,
                "trim_quantile": 0.0,
                "eta": 0.06,
                "steps": 6,
                "split_quantile": 0.9,
                "neighbor_k": 6,
                "split_tau": 0.12,
                "w_lambda": 1.0,
                "w_disp": 0.5,
                "w_dirichlet": 0.25,
                "candidate_m": 32,
            },
            {"eps": 1.29, "k": 8, "topk": 4, "p": 2.0, "sigma": None},
        )
    )
    return a, gl


def test_subgraph_float_min_size_accepted(energy_corpus_and_gl):
    a, gl = energy_corpus_and_gl
    a.spot_subg_motives(gl, {"min_size": 4.0})  # must not raise


def test_subgraph_non_integral_min_size_raises(energy_corpus_and_gl):
    a, gl = energy_corpus_and_gl
    with pytest.raises(TypeError):
        a.spot_subg_motives(gl, {"min_size": 4.5})


def test_subgraph_float_min_size_equal_int_behaviour(energy_corpus_and_gl):
    """float and int min_size must produce the same number of subgraphs.
    spot_subg_motives is deterministic (no unseeded tie-breaking here)."""
    a, gl = energy_corpus_and_gl
    r_int = a.spot_subg_motives(gl, {"min_size": 4})
    r_float = a.spot_subg_motives(gl, {"min_size": 4.0})
    assert len(r_int) >= 1, "corpus must yield subgraphs or the oracle is vacuous"
    assert len(r_int) == len(r_float)
    assert [sg["node_indices"] for sg in r_int] == [sg["node_indices"] for sg in r_float]
