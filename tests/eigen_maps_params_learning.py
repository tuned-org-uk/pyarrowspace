"""
EigenMaps Hyperparameter Learner (Seed-agnostic, Uniqueness-Aware, Cached Extrapolation)

Core features:
- Classical eigenmaps via ArrowSpaceBuilder.build
- Spectrum cost encourages connectivity, dispersion, uniqueness (no duplicates),
  spacing variability, and gentle gap regularization
- Robust search: scrambled low-discrepancy restarts, two-phase annealing
- Compact cache: only most meaningful runs retained (top-K by total cost with
  uniqueness and dispersion guards)
- Extrapolation from cache only (no recompute): local quadratic fit on cached
  neighbors around the best; proposed optimum reported only if present in cache

If LearnConfig.seed is None, randomness comes from OS entropy; set a seed to reproduce.
"""

import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from arrowspace import ArrowSpaceBuilder

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

LOGGER_NAME = "eigenmaps_learner"

def setup_logger(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] [%(levelname)s] %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

log = setup_logger()

# --------------------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------------------

@dataclass
class GraphParams:
    eps: float = 0.2
    sigma: float = 0.08
    k: int = 8
    topk: int = 8
    p: float = 2.0

@dataclass
class LearnConfig:
    # Search budget
    n_iter: int = 30
    restarts: int = 16
    seed: Optional[int] = None  # None => OS entropy
    # Objective shaping
    bins_init: int = 4
    bins_final: int = 10
    eps0: float = 1e-9  # near-zero threshold for Laplacian zero eigenvalues
    # weights: (connectivity, collapse, spacing, gap_reg, duplicates, repulsion)
    cost_weights: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 0.5, 0.10, 0.10, 0.05)
    # Gap regularization targets on nonzero spectrum
    target_median: float = 0.15
    target_range_max: float = 1.5
    # Bounds
    min_k: int = 3
    min_sigma: float = 1e-4
    max_sigma: float = 4.0
    min_eps: float = 1e-4
    max_eps: float = 4.0
    # Two-phase annealing
    explore_frac: float = 0.4
    T_explore: float = 3.0
    cool_explore: float = 0.92
    T_exploit: float = 0.7
    cool_exploit: float = 0.97
    # Jitter magnitudes (log-space for kernel widths)
    jitter_explore: Tuple[float, float] = (0.6, 0.6)  # eps, sigma
    jitter_exploit: Tuple[float, float] = (0.3, 0.3)
    # Ensemble-lite knobs (kept for parity; extrapolation is cache-only)
    elite_count: int = 3
    ensemble_M: int = 5
    ensemble_width_jitter: float = 0.05
    # Uniqueness controls
    dup_tol_rel: float = 1e-6
    repulse_eps: float = 1e-9

@dataclass
class CachePolicy:
    top_k: int = 64
    max_per_restart: int = 8
    max_dup_pen: float = 0.2
    max_collapse_pen: float = 0.8
    neighbor_log_radius: float = 0.7

@dataclass
class CacheEntry:
    key: Tuple
    graph: GraphParams
    lambdas: List[float]
    cost_final_bins: Dict[str, float]
    restart: int
    iter_index: int

@dataclass
class LearnedResult:
    best_cost: float
    best_graph: GraphParams
    best_lambdas: List[float]
    history: List[Dict[str, Any]]
    elites: List[Dict[str, Any]]
    cache: List[CacheEntry]

# --------------------------------------------------------------------------------------
# Utilities (I/O, math)
# --------------------------------------------------------------------------------------

def load_vectors_file(path: str) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []; rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l: continue
            parts = l.split(";", 1)
            ident = parts[0].strip()
            vals_str = parts[1].strip() if len(parts) > 1 else ""
            if not vals_str: continue
            vals = [float(s.strip()) for s in vals_str.split(",") if s.strip()]
            ids.append(ident); rows.append(vals)
    if not rows: return ids, np.empty((0, 0), dtype=np.float64)
    fdim = len(rows[0])
    if any(len(r) != fdim for r in rows):
        raise ValueError("Inconsistent row lengths in vector file")
    X = np.asarray(rows, dtype=np.float64)
    return ids, X

def _bounded(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _as_float_list(lambdas_sorted: List[Tuple[float, int]]) -> List[float]:
    return [float(v) for (v, _) in lambdas_sorted]

# --------------------------------------------------------------------------------------
# Spectrum cost (uniqueness-aware)
# --------------------------------------------------------------------------------------

def _safe_probs(vals: np.ndarray, bins: int, eps: float = 1e-12) -> np.ndarray:
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin + 1e-12:
        p = np.zeros(bins, dtype=float); p[0] = 1.0; return p
    hist, _ = np.histogram(vals, bins=bins, range=(vmin, vmax + 1e-12))
    p = hist.astype(float) + eps; p /= p.sum(); return p

def _entropy(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log(p + 1e-12)))

def _gap_regularizer(vals: np.ndarray, target_median: float, target_range_max: float) -> float:
    if vals.size == 0: return 1.0
    med = float(np.median(vals))
    rng = float(vals.max() - vals.min()) if vals.size > 1 else 0.0
    return abs(med - target_median) + max(0.0, rng - target_range_max)

def _duplicate_penalty(vals: np.ndarray, tol: float) -> float:
    if vals.size <= 1: return 0.0
    uniq = []
    for v in np.sort(vals):
        if len(uniq) == 0 or abs(v - uniq[-1]) > tol:
            uniq.append(v)
    U = len(uniq); N = int(vals.size)
    return float((N - U) / max(1, N - 1))

def _repulsion_penalty(vals: np.ndarray, eps: float) -> float:
    m = int(vals.size)
    if m <= 1: return 0.0
    v = np.sort(vals); acc = 0.0; cnt = 0
    for i in range(m):
        for j in range(i+1, m):
            acc += -math.log(abs(v[j] - v[i]) + eps); cnt += 1
    return float(max(0.0, acc / max(1, cnt)))

def spectrum_cost(lambdas: List[float], cfg: LearnConfig, bins: int) -> Dict[str, float]:
    lam = np.asarray(lambdas, dtype=float)
    if lam.size == 0:
        return {"total": float("inf"), "comp_pen": 1.0, "collapse_pen": 1.0, "spacing_pen": 1.0,
                "gap_reg": 1.0, "dup_pen": 1.0, "repulse_pen": 1.0, "entropy": 0.0, "max_entropy": 1.0,
                "median_nz": 0.0, "range_nz": 0.0, "unique_count": 0, "n": 0}
    lam_sorted = np.sort(lam)
    nonzero = lam_sorted[lam_sorted >= cfg.eps0] if np.any(lam_sorted >= cfg.eps0) else np.array([cfg.eps0])

    comp_zeros = int(np.sum(lam_sorted < cfg.eps0))
    comp_pen = float(max(0, comp_zeros - 1))  # [web:7][web:53]

    p = _safe_probs(nonzero, bins=bins)
    ent = _entropy(p); max_ent = math.log(len(p)) if len(p) > 0 else 1.0
    collapse_pen = float(1.0 - (ent / max_ent if max_ent > 0 else 0.0))  # [web:53][web:62]

    if len(nonzero) > 1:
        gaps = np.diff(nonzero); var_nonzero = float(np.var(nonzero))
        spacing_pen = float(1.0 - (float(np.var(gaps)) / (var_nonzero + 1e-9))) if var_nonzero > 0 else 1.0
    else:
        spacing_pen = 1.0

    gap_reg = _gap_regularizer(nonzero, cfg.target_median, cfg.target_range_max)
    scale = float(max(1.0, np.median(nonzero)))
    dup_tol = float(cfg.dup_tol_rel * scale)
    dup_pen = _duplicate_penalty(nonzero, dup_tol)                      # [web:62][web:53]
    repulse_pen = _repulsion_penalty(nonzero, cfg.repulse_eps)          # [web:86][web:92]

    a,b,c,d,e,f = cfg.cost_weights
    total = a*comp_pen + b*collapse_pen + c*spacing_pen + d*gap_reg + e*dup_pen + f*repulse_pen

    return {
        "total": float(total),
        "comp_pen": comp_pen,
        "collapse_pen": collapse_pen,
        "spacing_pen": spacing_pen,
        "gap_reg": float(gap_reg),
        "dup_pen": float(dup_pen),
        "repulse_pen": float(repulse_pen),
        "entropy": float(ent),
        "max_entropy": float(max_ent),
        "median_nz": float(np.median(nonzero)),
        "range_nz": float((nonzero.max() - nonzero.min()) if len(nonzero) > 0 else 0.0),
        "unique_count": int(len(np.unique(np.round(nonzero/dup_tol)))) if dup_tol > 0 else int(nonzero.size),
        "n": int(lam_sorted.size),
    }

# --------------------------------------------------------------------------------------
# Build wrapper (EigenMaps)
# --------------------------------------------------------------------------------------

def _build_once(
    X: np.ndarray,
    g: GraphParams
) -> Tuple[List[float], Any, Any, Dict[str, Any]]:
    t0 = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder.build(
        items=X,
        graph_params=asdict(g)
    )
    t1 = time.perf_counter()
    lambdas = _as_float_list(aspace.lambdas_sorted())
    meta = {"build_time_s": float(t1 - t0)}
    return lambdas, aspace, gl, meta  # [web:51][web:7]

# --------------------------------------------------------------------------------------
# Cache keys and policy
# --------------------------------------------------------------------------------------

def _key_from_params(g: GraphParams) -> Tuple:
    return (
        round(math.log(max(g.eps, 1e-12)), 8),
        round(math.log(max(g.sigma, 1e-12)), 8),
        int(g.k), int(g.topk), round(g.p, 6),
    )

def _maybe_add_to_cache(cache: List[CacheEntry], entry: CacheEntry, policy: CachePolicy) -> None:
    ci = entry.cost_final_bins
    if ci["dup_pen"] > policy.max_dup_pen: return  # [web:62]
    if ci["collapse_pen"] > policy.max_collapse_pen: return  # [web:62]
    cache.append(entry)

def _prune_cache(cache: List[CacheEntry], policy: CachePolicy) -> List[CacheEntry]:
    per_restart: Dict[int, List[CacheEntry]] = {}
    for c in cache:
        per_restart.setdefault(c.restart, []).append(c)
    trimmed: List[CacheEntry] = []
    for r, lst in per_restart.items():
        lst_sorted = sorted(lst, key=lambda x: x.cost_final_bins["total"])
        trimmed.extend(lst_sorted[:policy.max_per_restart])
    trimmed_sorted = sorted(trimmed, key=lambda x: x.cost_final_bins["total"])
    return trimmed_sorted[:policy.top_k]

# --------------------------------------------------------------------------------------
# Seeding and proposals
# --------------------------------------------------------------------------------------

def _van_der_corput(n: int, base: int = 2) -> np.ndarray:
    seq = np.zeros(n)
    for i in range(n):
        x, denom = 0.0, 1.0; k = i + 1
        while k > 0:
            k, rem = divmod(k, base)
            denom *= base; x += rem / denom
        seq[i] = x
    return seq  # [web:56][web:67]

def _low_discrepancy_2d(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.stack([_van_der_corput(n, 2), _van_der_corput(n, 3)], axis=1)
    shift = rng.random(2)
    return (base + shift) % 1.0  # [web:56][web:67]

def _map_unit_to_log(lo: float, hi: float, u: float) -> float:
    llo, lhi = math.log(lo), math.log(hi)
    return math.exp(llo + (lhi - llo) * u)

def _initial_params_from_sobol(
    n_restarts: int,
    bounds: Dict[str, Tuple[float, float]],
    k_bounds: Tuple[int, int],
    rng: np.random.Generator
) -> List[GraphParams]:
    pts2 = _low_discrepancy_2d(n_restarts, rng)
    seeds: List[GraphParams] = []
    for i in range(n_restarts):
        u1, u2 = float(pts2[i,0]), float(pts2[i,1])
        eps0 = _map_unit_to_log(bounds["eps"][0], bounds["eps"][1], u1)
        sigma0 = _map_unit_to_log(bounds["sigma"][0], bounds["sigma"][1], u2)
        k0 = int(max(k_bounds[0], min(k_bounds[1], round(k_bounds[0] + u1*(k_bounds[1]-k_bounds[0])))))
        topk0 = max(k0, k0)
        seeds.append(GraphParams(eps=eps0, sigma=sigma0, k=k0, topk=topk0, p=2.0))
    return seeds  # [web:56][web:67]

def _propose_neighbor(
    g: GraphParams,
    n: int,
    jitter: Tuple[float, float],
    cfg: LearnConfig,
    rng: np.random.Generator
) -> GraphParams:
    je, js = jitter
    g2 = GraphParams(**asdict(g))
    g2.eps   = _bounded(g2.eps   * math.exp(rng.uniform(-je, je)), cfg.min_eps, cfg.max_eps)
    g2.sigma = _bounded(g2.sigma * math.exp(rng.uniform(-js, js)), cfg.min_sigma, cfg.max_sigma)
    k_new = int(max(cfg.min_k, min(g2.k + int(rng.integers(-1, 2)), max(2, n - 1))))
    g2.k = k_new; g2.topk = max(g2.topk, k_new)
    return g2  # [web:53][web:62]

def _phase_schedule(iter_idx: int, total_iter: int, cfg: LearnConfig):
    split = int(round(cfg.explore_frac * total_iter))
    if iter_idx < split:
        return cfg.T_explore, cfg.cool_explore, cfg.jitter_explore, cfg.bins_init
    else:
        return cfg.T_exploit, cfg.cool_exploit, cfg.jitter_exploit, cfg.bins_final  # [web:56][web:58]

# --------------------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------------------

def learn_eigenmaps_params(
    X: np.ndarray,
    init_graph: GraphParams = None,
    learn_cfg: LearnConfig = None,
    cache_policy: CachePolicy = CachePolicy(),
) -> LearnedResult:
    if learn_cfg is None:
        learn_cfg = LearnConfig()
    rng = np.random.default_rng(learn_cfg.seed)

    n = int(X.shape[0])
    default_graph = GraphParams(
        k=min(8, max(learn_cfg.min_k, n - 1)),
        topk=min(8, max(learn_cfg.min_k, n - 1)),
    )
    g_base = init_graph or default_graph

    sobol_seeds = _initial_params_from_sobol(
        learn_cfg.restarts,
        bounds={"eps": (learn_cfg.min_eps, learn_cfg.max_eps),
                "sigma": (learn_cfg.min_sigma, learn_cfg.max_sigma)},
        k_bounds=(learn_cfg.min_k, max(learn_cfg.min_k, n - 1)),
        rng=rng
    )

    history: List[Dict[str, Any]] = []
    elites: List[Dict[str, Any]] = []
    cache: List[CacheEntry] = []

    for restart in range(learn_cfg.restarts):
        use_sobol = (restart % 2 == 0)
        if use_sobol:
            g = GraphParams(**{**asdict(g_base), **asdict(sobol_seeds[restart])})
        else:
            g = GraphParams(**asdict(g_base))
            g.eps = _map_unit_to_log(learn_cfg.min_eps, learn_cfg.max_eps, float(rng.random()))
            g.sigma = _map_unit_to_log(learn_cfg.min_sigma, learn_cfg.max_sigma, float(rng.random()))
            g.k = int(max(learn_cfg.min_k, min(max(2, n - 1), int(rng.integers(learn_cfg.min_k, max(learn_cfg.min_k, n-1)+1)))))
            g.topk = max(g.topk, g.k)

        lambdas, _, _, _ = _build_once(X, g)
        cost_info_init = spectrum_cost(lambdas, learn_cfg, bins=learn_cfg.bins_init)
        cost_info_final = spectrum_cost(lambdas, learn_cfg, bins=learn_cfg.bins_final)
        cost = cost_info_init["total"]

        _maybe_add_to_cache(
            cache,
            CacheEntry(
                key=_key_from_params(g),
                graph=g, lambdas=list(map(float, lambdas)),
                cost_final_bins=cost_info_final, restart=restart, iter_index=-1
            ),
            cache_policy
        )

        log.info(f"Restart {restart+1}/{learn_cfg.restarts} | init_cost={cost:.6f} | graph={asdict(g)}")

        T = None
        for t in range(learn_cfg.n_iter):
            T_curr, cool, jitter, bins_use = _phase_schedule(t, learn_cfg.n_iter, learn_cfg)
            if T is None:
                T = T_curr
            else:
                split = int(round(learn_cfg.explore_frac * learn_cfg.n_iter))
                if (t-1) < split <= t:
                    T = T_curr

            g_c = _propose_neighbor(g, n, jitter, learn_cfg, rng)
            lambdas_c, _, _, _ = _build_once(X, g_c)
            cost_info_c_init = spectrum_cost(lambdas_c, learn_cfg, bins=bins_use)
            cost_info_c_final = spectrum_cost(lambdas_c, learn_cfg, bins=learn_cfg.bins_final)
            cost_c = cost_info_c_init["total"]

            _maybe_add_to_cache(
                cache,
                CacheEntry(
                    key=_key_from_params(g_c),
                    graph=g_c, lambdas=list(map(float, lambdas_c)),
                    cost_final_bins=cost_info_c_final, restart=restart, iter_index=t
                ),
                cache_policy
            )

            improve = cost_c < cost
            accept = improve or (math.exp((cost - cost_c) / max(1e-6, T)) > float(rng.random()))

            history.append({
                "restart": restart, "iter": t, "accepted": bool(accept), "improve": bool(improve),
                "temperature": float(T), "cost": float(cost), "cost_new": float(cost_c),
                "cost_detail": cost_info_init, "cost_detail_new": cost_info_c_init,
                "graph": asdict(g), "graph_new": asdict(g_c),
                "lambdas": list(map(float, lambdas)), "lambdas_new": list(map(float, lambdas_c)),
                "phase_bins": bins_use,
            })

            log.debug(f"[r{restart} t{t}] accept={accept} improve={improve} T={T:.4f} "
                      f"cost={cost:.6f}->{cost_c:.6f} "
                      f"comp={cost_info_c_init['comp_pen']:.3f} "
                      f"collapse={cost_info_c_init['collapse_pen']:.3f} "
                      f"spacing={cost_info_c_init['spacing_pen']:.3f} "
                      f"gap={cost_info_c_init['gap_reg']:.3f} "
                      f"dup={cost_info_c_init['dup_pen']:.3f} "
                      f"rep={cost_info_c_init['repulse_pen']:.3f} | g_new={asdict(g_c)}")

            if accept:
                g, lambdas, cost_info_init, cost_info_final, cost = g_c, lambdas_c, cost_info_c_init, cost_info_c_final, cost_c
            T *= cool

        elites.append({
            "restart": restart,
            "graph": asdict(g),
            "mean_cost": float(cost_info_final["total"]),
            "diag": cost_info_final,
            "lambdas": list(map(float, lambdas)),
        })

        cache = _prune_cache(cache, cache_policy)

    elites_sorted = sorted(elites, key=lambda z: (z["mean_cost"], z["diag"]["collapse_pen"]))
    best = elites_sorted[0]
    best_graph = GraphParams(**best["graph"])

    key_best = _key_from_params(best_graph)
    best_entry = next((c for c in cache if c.key == key_best), None)
    best_lam = best_entry.lambdas if best_entry else best["lambdas"]
    best_cost = best_entry.cost_final_bins["total"] if best_entry else best["mean_cost"]

    log.info(
        "Final best (pre-extrapolation) | "
        f"mean_cost={best_cost:.6f} diag={best_entry.cost_final_bins if best_entry else best['diag']} | "
        f"params={{eps:{best_graph.eps:.6g}, sigma:{best_graph.sigma:.6g}, "
        f"k:{best_graph.k}, topk:{best_graph.topk}, p:{best_graph.p}}}"
    )

    return LearnedResult(
        best_cost=float(best_cost),
        best_graph=best_graph,
        best_lambdas=list(map(float, best_lam)),
        history=history,
        elites=elites_sorted,
        cache=cache,
    )

# --------------------------------------------------------------------------------------
# Cached extrapolation (no recompute)
# --------------------------------------------------------------------------------------

def _in_neighbor(g: GraphParams, g_ref: GraphParams, radius: float) -> bool:
    a = abs(math.log(max(g.eps,1e-12)) - math.log(max(g_ref.eps,1e-12)))
    b = abs(math.log(max(g.sigma,1e-12)) - math.log(max(g_ref.sigma,1e-12)))
    return (a <= radius) and (b <= radius)

def _fit_local_quadratic(XY: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
    M = []; y = []
    def feat(x):
        a,b,c = x
        return np.array([1.0, a, b, c, 0.5*a*a, 0.5*b*b, 0.5*c*c, a*b, a*c, b*c], dtype=float)
    for x, cost in XY:
        M.append(feat(x)); y.append(cost)
    M = np.vstack(M); y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ beta; rmse = float(np.sqrt(np.mean(resid*resid))) if len(y) > 0 else float("inf")
    return beta, rmse

def _quadratic_minimizer(beta: np.ndarray) -> np.ndarray:
    b0, ba, bb, bc = beta[0], beta[1], beta[2], beta[3]
    haa, hbb, hcc = beta[4], beta[5], beta[6]
    hab, hac, hbc = beta[7], beta[8], beta[9]
    H = np.array([[haa, hab, hac], [hab, hbb, hbc], [hac, hbc, hcc]], dtype=float)
    g = np.array([ba, bb, bc], dtype=float)
    H_reg = H + 1e-6*np.eye(3)
    try:
        x_star = -np.linalg.solve(H_reg, g)
    except np.linalg.LinAlgError:
        x_star = -np.linalg.pinv(H_reg) @ g
    return x_star

def extrapolate_from_cache(
    cache: List[CacheEntry],
    best_graph: GraphParams,
    learn_cfg: LearnConfig,
    policy: CachePolicy,
    n_items: int
) -> Tuple[GraphParams, Dict[str, Any], List[float]]:
    neigh: List[CacheEntry] = [c for c in cache if _in_neighbor(c.graph, best_graph, policy.neighbor_log_radius)]
    if len(neigh) < 5:
        return best_graph, {"rmse_local_fit": float("inf"), "extrapolated_cost_mean": float("inf"),
                            "extrapolated_diag": {}, "eps_star": best_graph.eps,
                            "sigma_star": best_graph.sigma, "k_star": best_graph.k, "design_points": len(neigh)}, []

    XY: List[Tuple[np.ndarray, float]] = []
    for c in neigh:
        a = math.log(c.graph.eps); b = math.log(c.graph.sigma); d = float(c.graph.k)
        XY.append((np.array([a,b,d], dtype=float), float(c.cost_final_bins["total"])))

    beta, rmse = _fit_local_quadratic(XY)
    x_star = _quadratic_minimizer(beta)
    a_star, b_star, c_star = float(x_star[0]), float(x_star[1]), float(x_star[2])

    eps_star = _bounded(math.exp(a_star), learn_cfg.min_eps, learn_cfg.max_eps)
    sig_star = _bounded(math.exp(b_star), learn_cfg.min_sigma, learn_cfg.max_sigma)
    k_star = int(max(learn_cfg.min_k, min(round(c_star), max(2, n_items-1))))
    g_star = GraphParams(**asdict(best_graph))
    g_star.eps, g_star.sigma, g_star.k = eps_star, sig_star, k_star
    g_star.topk = max(g_star.topk, g_star.k)

    key_star = _key_from_params(g_star)
    c_star_entry = next((c for c in cache if c.key == key_star), None)
    if c_star_entry:
        mean_cost = float(c_star_entry.cost_final_bins["total"])
        diag = c_star_entry.cost_final_bins
        lam = c_star_entry.lambdas
    else:
        mean_cost = float("inf"); diag = {}; lam = []

    report = {
        "rmse_local_fit": rmse,
        "extrapolated_cost_mean": mean_cost,
        "extrapolated_diag": diag,
        "eps_star": eps_star,
        "sigma_star": sig_star,
        "k_star": k_star,
        "design_points": len(neigh),
    }
    return g_star, report, lam

def run_extrapolation_and_report_cached(
    X: np.ndarray,
    learn_cfg: LearnConfig,
    cache_policy: CachePolicy,
    result: LearnedResult
) -> None:
    g_best = result.best_graph
    g_star, rep, lam_star = extrapolate_from_cache(
        cache=result.cache,
        best_graph=g_best,
        learn_cfg=learn_cfg,
        policy=cache_policy,
        n_items=X.shape[0]
    )
    improved = (rep["extrapolated_cost_mean"] + 1e-6 < result.best_cost)
    log.info(
        "Extrapolation (cached) | "
        f"search_mean_cost={result.best_cost:.6f} "
        f"star_mean_cost={rep['extrapolated_cost_mean']:.6f} "
        f"improved={improved} | "
        f"g_best={{eps:{g_best.eps:.4g}, sigma:{g_best.sigma:.4g}, k:{g_best.k}}} "
        f"g_star={{eps:{rep['eps_star']:.4g}, sigma:{rep['sigma_star']:.4g}, k:{rep['k_star']}}} | "
        f"fit_rmse={rep['rmse_local_fit']:.4g} design_n={rep['design_points']}"
    )
    if lam_star:
        p = _safe_probs(np.asarray(lam_star, dtype=float), bins=learn_cfg.bins_final)
        log.info(f"Extrapolation λ bin probs (cached): {np.round(p, 4).tolist()}")

# --------------------------------------------------------------------------------------
# Example CLI usage
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    # OS-entropy rng by default; set LearnConfig.seed to fix reproducibility
    # X = np.random.default_rng().standard_normal((100, 1000)).astype(np.float64)

    ids, X = load_vectors_file("/datadisk/publish/pyarrowspace/tests/small_datasets/vectors_data_1500.txt")
    print(ids[0], X.shape, X.dtype)

    # Number of rows to pick
    n_samples = 10
    # Create a new random generator instance
    rng = np.random.default_rng()
    # Generate 100 unique random indices without replacement
    random_indices = rng.choice(X.shape[0], size=n_samples, replace=False)
    X = X[random_indices]

    cfg = LearnConfig(
        n_iter=30, restarts=16, seed=None,
        bins_init=4, bins_final=10,
        explore_frac=0.4, T_explore=3.0, cool_explore=0.92,
        T_exploit=0.7, cool_exploit=0.97
    )
    policy = CachePolicy(top_k=64, max_per_restart=8, max_dup_pen=0.2, max_collapse_pen=0.8)

    result = learn_eigenmaps_params(X, learn_cfg=cfg, cache_policy=policy)
    log.info(f"Best λ: {np.round(result.best_lambdas, 6).tolist()}")

    run_extrapolation_and_report_cached(X, cfg, policy, result)
