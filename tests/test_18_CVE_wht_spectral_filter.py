"""CVE semantic search with Graph-WHT spectral filter — end-to-end benchmark.

Extends test_17_CVE_semantic_recall.py with a fourth (and fifth) search mode
that inserts the Graph-WHT low-pass filter between query encoding and
ArrowSpace retrieval.  This makes WHT a first-class measurable step in the
pipeline rather than an isolated unit test.

Search modes
------------
  Cosine    tau=1.0   raw query embedding  (exact cosine baseline)
  Hybrid    tau=0.75  raw query embedding  (spectral-aware, mostly cosine)
  Taumode   tau=0.60  raw query embedding  (spectral-dominant)
  WHT-LP    tau=1.0   WHT low-pass pre-filtered query  (new)
  WHT-Tau   tau=0.60  WHT low-pass pre-filtered query  (new: WHT + spectral scoring)

For each mode the same metric suite as test_17 is computed:
  • Spearman-ρ / Kendall-τ rank correlation vs Cosine baseline
  • NDCG@10 vs Cosine baseline
  • Tail quality (head/tail ratio, CV, decay rate)
  • Semantic Recall (traditional / semantic / tolerant) — Kuffo et al., SIGIR '26

Additional WHT-specific diagnostics
  • WHT spectral energy distribution of the query (before / after filtering)
  • L2-norm ratio  ||q_wht|| / ||q_raw||  (energy retention under low-pass)
  • Cosine similarity between raw and WHT-filtered query vectors

Usage
-----
  python tests/test_18_CVE_wht_spectral_filter.py --dataset <cve_dir>

Requirements
------------
  pip install sentence-transformers numpy matplotlib scipy scikit-learn tqdm
  # arrowspace must be installed (Rust wheels)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from arrowspace import ArrowSpaceBuilder, set_debug
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# Graph-WHT filter — Python-layer, no Rust changes
from graph_wht_spectral_filter import GraphWHTFilter, heat_kernel, ideal_lowpass

logging.basicConfig(level=logging.INFO)
set_debug(True)

# ============================================================================
# Configuration
# ============================================================================
START_YEAR   = 1999
END_YEAR     = 2026
TAU_COSINE   = 1.0    # pure cosine
TAU_HYBRID   = 0.75   # hybrid
TAU_TAUMODE  = 0.60   # spectral-dominant
TAU_WHT_LP   = 1.0    # WHT-filtered query, pure cosine scoring
TAU_WHT_TAU  = 0.60   # WHT-filtered query, spectral scoring
K_TAIL_MAX   = 30
WHT_T        = 1.0    # diffusion time for heat kernel

graph_params = {
    "eps":   1.31,
    "k":     K_TAIL_MAX,
    "topk":  int(K_TAIL_MAX / 2),
    "p":     1.8,
    "sigma": 0.535,
}

# ============================================================================
# Data loading  (identical to test_17)
# ============================================================================

def iter_cve_json(root_dir, start=START_YEAR, end=END_YEAR):
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end + 1)):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    yield path, json.load(f)
                except Exception:
                    continue


def extract_text(j):
    cve_id  = j.get("cveMetadata", {}).get("cveId", "")
    cna     = j.get("containers", {}).get("cna", {})
    title   = cna.get("title", "") or ""
    descs   = []
    for d in cna.get("descriptions", []) or []:
        if isinstance(d, dict):
            val = d.get("value") or ""
            if val:
                descs.append(val)
    description = " ".join(descs)
    cwes = []
    for pt in cna.get("problemTypes", []) or []:
        for d in pt.get("descriptions", []) or []:
            cwe = d.get("cweId")
            if cwe:
                cwes.append(cwe)
    cwe_str  = " ".join(cwes)
    cvss_vec = ""
    for m in cna.get("metrics", []) or []:
        v31 = m.get("cvssV3_1")
        if isinstance(v31, dict):
            vs = v31.get("vectorString")
            if vs:
                cvss_vec = vs
                break
    affected = cna.get("affected", []) or []
    products = []
    for a in affected:
        vendor  = a.get("vendor")  or ""
        product = a.get("product") or ""
        if vendor or product:
            products.append(f"{vendor} {product}".strip())
    prod_str = " ".join(products)
    text = " | ".join(
        [s for s in [cve_id, title, description, cwe_str, cvss_vec, prod_str] if s]
    )
    return cve_id or "(unknown)", title or "(no title)", text


def build_embeddings(
    texts,
    model_path: str = str(Path(__file__).parent.parent / "domain_adapted_model"),
    cache_file: str = "cve_embeddings_cache.npy",
):
    if os.path.exists(cache_file):
        try:
            X = np.load(cache_file)
            if len(X) == len(texts):
                print(f"Loaded cached embeddings {X.shape} from {cache_file}")
                return X
        except Exception:
            pass
    print(f"Encoding {len(texts)} texts via {model_path} …")
    model = SentenceTransformer(model_path)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    X = X.astype(np.float64) * 1.2e1
    np.save(cache_file, X)
    return X


# ============================================================================
# Ranking / recall metrics  (identical helpers to test_17)
# ============================================================================

def compute_ranking_metrics(results_a, results_b):
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [indices_a.index(i) for i in shared]
    rank_b = [indices_b.index(i) for i in shared]
    rho, _  = spearmanr(rank_a, rank_b)
    tau, _  = kendalltau(rank_a, rank_b)
    return rho, tau


def compute_ndcg(results_pred, results_ref, k=10):
    ref_indices = [idx for idx, _ in results_ref[:k]]
    relevance_map = {idx: k - i for i, idx in enumerate(ref_indices)}
    pred_indices  = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]
    if sum(true_relevance) == 0:
        return 0.0
    try:
        pred_scores = np.array([sc for _, sc in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()
        return ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k,
        )
    except Exception:
        return 0.0


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    min_length = min(len(r) for r in results_list)
    if min_length <= k_head:
        return {}
    actual_k_tail = min(k_tail, min_length)
    metrics = {}
    for results, label in zip(results_list, labels):
        seg = results[:actual_k_tail]
        head_scores = [sc for _, sc in seg[:k_head]]
        tail_scores = [sc for _, sc in seg[k_head:actual_k_tail]]
        if not tail_scores or not head_scores:
            continue
        tail_mean = np.mean(tail_scores)
        tail_std  = np.std(tail_scores)
        head_mean = np.mean(head_scores)
        metrics[label] = {
            "head_mean":          head_mean,
            "tail_mean":          tail_mean,
            "tail_std":           tail_std,
            "tail_to_head_ratio": tail_mean / head_mean if head_mean > 1e-10 else 0.0,
            "tail_cv":            tail_std / tail_mean  if tail_mean > 1e-10 else 0.0,
            "tail_decay_rate":    (tail_scores[0] - tail_scores[-1]) / len(tail_scores)
                                  if len(tail_scores) > 1 else 0.0,
            "n_tail_items":       len(tail_scores),
            "total_items":        actual_k_tail,
        }
    return metrics


# ── Semantic Recall  (Kuffo et al., SIGIR '26) ────────────────────────────

def compute_traditional_recall(retrieved_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 0.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(set(ground_truth_ids))


def compute_semantic_recall(retrieved_ids, ground_truth_ids, semantic_neighbor_ids):
    sn_set = set(semantic_neighbor_ids) & set(ground_truth_ids)
    if not sn_set:
        return float("nan")
    return len(set(retrieved_ids) & sn_set) / len(sn_set)


def compute_tolerant_recall(
    retrieved_ids, retrieved_scores,
    ground_truth_ids, ground_truth_scores,
    tolerance_pct=1.0,
):
    if not ground_truth_ids:
        return 0.0
    gt_score_map = {idx: sc for idx, sc in zip(ground_truth_ids, ground_truth_scores)}
    matched_gt   = set()
    matched_count = 0
    for ret_idx, ret_score in zip(retrieved_ids, retrieved_scores):
        if ret_idx in gt_score_map and ret_idx not in matched_gt:
            matched_gt.add(ret_idx)
            matched_count += 1
        else:
            for gt_idx, gt_score in zip(ground_truth_ids, ground_truth_scores):
                if gt_idx in matched_gt:
                    continue
                if ret_score >= gt_score * (1.0 - tolerance_pct / 100.0):
                    matched_gt.add(gt_idx)
                    matched_count += 1
                    break
    return matched_count / len(ground_truth_ids)


def estimate_tolerance_threshold(ground_truth_scores, k):
    scores = list(ground_truth_scores)[:k]
    if len(scores) < 2:
        return 1.0
    max_score = max(scores) if max(scores) > 0 else 1.0
    two_thirds_k  = max(0, int(2 * k / 3) - 1)
    threshold_pct = abs(scores[two_thirds_k] - scores[-1]) / max_score * 100.0
    return max(0.1, threshold_pct)


def identify_semantic_neighbors(ground_truth_ids, ground_truth_scores,
                                score_gap_percentile=25.0):
    if not ground_truth_scores:
        return []
    scores    = np.array(ground_truth_scores)
    threshold = np.percentile(scores, 100 - score_gap_percentile)
    return [idx for idx, sc in zip(ground_truth_ids, ground_truth_scores)
            if sc >= threshold]


def compute_all_recall_metrics(
    retrieved_ids, retrieved_scores,
    ground_truth_ids, ground_truth_scores,
    tolerance_pct=None, sn_score_gap_percentile=25.0,
):
    k    = len(ground_truth_ids)
    trad = compute_traditional_recall(retrieved_ids, ground_truth_ids)
    sn_ids = identify_semantic_neighbors(
        ground_truth_ids, ground_truth_scores, sn_score_gap_percentile
    )
    sem = compute_semantic_recall(retrieved_ids, ground_truth_ids, sn_ids)
    if tolerance_pct is None:
        tolerance_pct = estimate_tolerance_threshold(ground_truth_scores, k)
    tol = compute_tolerant_recall(
        retrieved_ids, retrieved_scores,
        ground_truth_ids, ground_truth_scores,
        tolerance_pct=tolerance_pct,
    )
    return {
        "traditional_recall":   trad,
        "semantic_recall":      sem,
        "tolerant_recall":      tol,
        "n_semantic_neighbors": len(sn_ids),
        "tolerance_pct_used":   tolerance_pct,
    }


# ============================================================================
# WHT-specific diagnostics
# ============================================================================

def wht_query_diagnostics(
    q_raw: np.ndarray,
    q_wht: np.ndarray,
    wht_filter: GraphWHTFilter,
) -> dict:
    """Compute WHT-specific quality measures for a single query pair.

    Returns
    -------
    dict with:
        energy_retention  : ||q_wht||^2 / ||q_raw||^2   (1.0 = no energy lost)
        cosine_similarity : cos(q_raw, q_wht)           (1.0 = identical direction)
        low_freq_fraction : fraction of WHT energy below median lambda
        high_freq_fraction: fraction of WHT energy above median lambda
        spectral_centroid : energy-weighted mean lambda of the raw signal
        spectral_shift    : change in spectral centroid after filtering
    """
    lam_raw, e_raw = wht_filter.spectral_energy(q_raw)
    lam_wht, e_wht = wht_filter.spectral_energy(q_wht)

    norm_raw = np.linalg.norm(q_raw)
    norm_wht = np.linalg.norm(q_wht)

    energy_retention  = (norm_wht ** 2) / (norm_raw ** 2) if norm_raw > 1e-12 else 0.0
    cosine_sim        = float(np.dot(q_raw, q_wht) / (norm_raw * norm_wht + 1e-12))

    median_lam        = np.median(lam_raw)
    total_e_raw       = np.sum(e_raw) + 1e-30
    low_freq_fraction = float(np.sum(e_raw[lam_raw <= median_lam]) / total_e_raw)
    high_freq_fraction= float(np.sum(e_raw[lam_raw >  median_lam]) / total_e_raw)

    spectral_centroid_raw = float(np.sum(lam_raw * e_raw) / total_e_raw)
    total_e_wht           = np.sum(e_wht) + 1e-30
    spectral_centroid_wht = float(np.sum(lam_wht * e_wht) / total_e_wht)
    spectral_shift        = spectral_centroid_wht - spectral_centroid_raw

    return {
        "energy_retention":   energy_retention,
        "cosine_similarity":  cosine_sim,
        "low_freq_fraction":  low_freq_fraction,
        "high_freq_fraction": high_freq_fraction,
        "spectral_centroid_raw": spectral_centroid_raw,
        "spectral_centroid_wht": spectral_centroid_wht,
        "spectral_shift":     spectral_shift,
    }


# ============================================================================
# CSV export helpers
# ============================================================================

def _write_csv(output_file, fieldnames, rows):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {output_file}")


def save_wht_diagnostics_csv(all_wht_diag, queries, output_file="wht_diagnostics.csv"):
    """Per-query WHT spectral diagnostics."""
    fields = [
        "query_id", "query_text",
        "energy_retention", "cosine_similarity",
        "low_freq_fraction", "high_freq_fraction",
        "spectral_centroid_raw", "spectral_centroid_wht", "spectral_shift",
    ]
    rows = []
    for qi, (q, diag) in enumerate(zip(queries, all_wht_diag)):
        row = {"query_id": qi + 1, "query_text": q}
        row.update({k: f"{v:.6f}" for k, v in diag.items()})
        rows.append(row)
    _write_csv(output_file, fields, rows)


def save_search_results_csv(queries, all_results, mode_labels, ids, titles,
                            output_file="wht_search_results.csv"):
    fields = ["query_id", "query_text", "mode", "rank", "cve_id", "title", "score"]
    rows   = []
    for qi, query in enumerate(queries):
        for mode_label, results in zip(mode_labels, all_results[qi]):
            for rank, (idx, score) in enumerate(results[:20], 1):
                rows.append({
                    "query_id":   qi + 1,
                    "query_text": query,
                    "mode":       mode_label,
                    "rank":       rank,
                    "cve_id":     ids[idx],
                    "title":      titles[idx],
                    "score":      f"{score:.6f}",
                })
    _write_csv(output_file, fields, rows)


def save_ranking_metrics_csv(comparison_metrics, mode_labels,
                             output_file="wht_ranking_metrics.csv"):
    """Spearman / Kendall / NDCG of every mode vs Cosine baseline."""
    fields = ["query_id", "query_text", "mode",
              "spearman_vs_cosine", "kendall_vs_cosine", "ndcg_vs_cosine"]
    rows   = []
    for qi, m in enumerate(comparison_metrics):
        for mode_label, rho, tau, ndcg in zip(
            mode_labels[1:],    # skip Cosine vs Cosine
            m["spearman_vs_cosine"],
            m["kendall_vs_cosine"],
            m["ndcg_vs_cosine"],
        ):
            rows.append({
                "query_id":          qi + 1,
                "query_text":        m["query"],
                "mode":              mode_label,
                "spearman_vs_cosine": f"{rho:.6f}",
                "kendall_vs_cosine":  f"{tau:.6f}",
                "ndcg_vs_cosine":     f"{ndcg:.6f}",
            })
    _write_csv(output_file, fields, rows)


def save_recall_metrics_csv(comparison_metrics, mode_labels,
                            output_file="wht_recall_metrics.csv"):
    fields = [
        "query_id", "query_text", "mode",
        "traditional_recall", "semantic_recall", "tolerant_recall",
        "n_semantic_neighbors", "tolerance_pct_used", "semantic_minus_traditional",
    ]
    rows = []
    for qi, m in enumerate(comparison_metrics):
        for mode_label in mode_labels:
            rm = m.get("recall_metrics", {}).get(mode_label)
            if not rm:
                continue
            trad = rm["traditional_recall"]
            sem  = rm["semantic_recall"]
            is_nan = isinstance(sem, float) and sem != sem
            diff   = float("nan") if is_nan else sem - trad
            rows.append({
                "query_id":                  qi + 1,
                "query_text":                m["query"],
                "mode":                      mode_label,
                "traditional_recall":        f"{trad:.6f}",
                "semantic_recall":           "nan" if is_nan else f"{sem:.6f}",
                "tolerant_recall":           f"{rm['tolerant_recall']:.6f}",
                "n_semantic_neighbors":      rm["n_semantic_neighbors"],
                "tolerance_pct_used":        f"{rm['tolerance_pct_used']:.4f}",
                "semantic_minus_traditional": "nan" if (isinstance(diff, float) and diff != diff)
                                              else f"{diff:.6f}",
            })
    _write_csv(output_file, fields, rows)


def save_summary_csv(comparison_metrics, mode_labels,
                     output_file="wht_summary.csv"):
    fields = ["metric_type", "mode", "mean", "std_dev"]
    rows   = []

    # NDCG averages
    for i, mode_label in enumerate(mode_labels[1:]):
        vals = [m["ndcg_vs_cosine"][i] for m in comparison_metrics]
        rows.append({"metric_type": "NDCG@10 vs Cosine", "mode": mode_label,
                     "mean": f"{np.mean(vals):.6f}", "std_dev": f"{np.std(vals):.6f}"})

    # Recall averages
    for mode_label in mode_labels:
        for recall_type in ["traditional_recall", "semantic_recall", "tolerant_recall"]:
            vals = []
            for m in comparison_metrics:
                rm = m.get("recall_metrics", {}).get(mode_label)
                if rm:
                    v = rm[recall_type]
                    if not (isinstance(v, float) and v != v):
                        vals.append(v)
            if vals:
                rows.append({"metric_type": recall_type.replace("_", " ").title(),
                             "mode": mode_label,
                             "mean": f"{np.mean(vals):.6f}",
                             "std_dev": f"{np.std(vals):.6f}"})

    # WHT diagnostics averages
    for diag_key in ["energy_retention", "cosine_similarity",
                     "low_freq_fraction", "spectral_shift"]:
        vals = [m.get("wht_diagnostics", {}).get(diag_key, float("nan"))
                for m in comparison_metrics]
        clean = [v for v in vals if not (isinstance(v, float) and v != v)]
        if clean:
            rows.append({"metric_type": f"WHT {diag_key}", "mode": "WHT-LP",
                         "mean": f"{np.mean(clean):.6f}",
                         "std_dev": f"{np.std(clean):.6f}"})

    _write_csv(output_file, fields, rows)


# ============================================================================
# Visualisations
# ============================================================================

def plot_wht_spectral_energy(
    queries, all_wht_diag, wht_filter: GraphWHTFilter,
    sample_embeddings: np.ndarray,
    output_file="wht_spectral_energy.png",
):
    """One row per query: (a) raw WHT energy, (b) filtered WHT energy, (c) delta."""
    n = min(6, len(queries))   # cap at 6 rows for readability
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    h = heat_kernel(WH T_T)
    for qi in range(n):
        q_raw = sample_embeddings[qi]
        lam, e_raw = wht_filter.spectral_energy(q_raw)

        # filtered energy
        weights  = h(lam)
        e_filt   = (weights ** 2) * e_raw

        # (a) raw
        ax = axes[qi, 0]
        ax.semilogy(lam, e_raw + 1e-30, lw=1.0, color="#4c72b0")
        ax.set_title(f"Q{qi+1} raw WHT energy\n{queries[qi][:45]}…", fontsize=9)
        ax.set_xlabel("λ (WHT approx)"); ax.set_ylabel("energy"); ax.grid(alpha=0.3)

        # (b) filtered
        ax = axes[qi, 1]
        ax.semilogy(lam, e_filt + 1e-30, lw=1.0, color="#55a868")
        ax.set_title(f"Q{qi+1} WHT energy after heat-kernel (t={WH T_T})", fontsize=9)
        ax.set_xlabel("λ (WHT approx)"); ax.set_ylabel("energy"); ax.grid(alpha=0.3)

        # (c) delta
        ax = axes[qi, 2]
        delta = e_filt - e_raw
        ax.fill_between(lam, delta, where=(delta >= 0), alpha=0.6,
                        color="green", label="+")
        ax.fill_between(lam, delta, where=(delta < 0), alpha=0.6,
                        color="red",   label="-")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Q{qi+1} energy delta (filtered − raw)", fontsize=9)
        ax.set_xlabel("λ (WHT approx)"); ax.set_ylabel("Δ energy")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"WHT spectral energy plot saved to {output_file}")
    plt.close()


def plot_mode_comparison(
    queries, all_results, mode_labels, comparison_metrics,
    output_file="wht_mode_comparison.png",
):
    """Grouped bar: NDCG@10 and traditional recall per query for all 5 modes."""
    n_queries = len(queries)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, 2 * n_queries), 10))
    x      = np.arange(n_queries)
    n_modes = len(mode_labels)
    width  = 0.8 / n_modes
    colors = ["#4c72b0", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Panel 1: NDCG@10
    ax = axes[0]
    for i, mode_label in enumerate(mode_labels[1:]):
        ndcg_vals = [m["ndcg_vs_cosine"][i] for m in comparison_metrics]
        ax.bar(x + i * width, ndcg_vals, width, label=mode_label,
               color=colors[i + 1], alpha=0.85)
    ax.set_ylabel("NDCG@10 vs Cosine", fontsize=11)
    ax.set_title("NDCG@10 per query (Cosine = reference)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_modes - 2) / 2)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)], rotation=45, ha="right")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.2)

    # Panel 2: Traditional Recall
    ax = axes[1]
    for i, mode_label in enumerate(mode_labels):
        recall_vals = []
        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(mode_label)
            recall_vals.append(rm["traditional_recall"] if rm else 0.0)
        ax.bar(x + i * width, recall_vals, width, label=mode_label,
               color=colors[i], alpha=0.85)
    ax.set_ylabel("Traditional Recall@k", fontsize=11)
    ax.set_title("Traditional Recall per query across modes", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * (n_modes - 1) / 2)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)], rotation=45, ha="right")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Mode comparison plot saved to {output_file}")
    plt.close()


def plot_wht_recall_uplift(
    comparison_metrics, mode_labels,
    output_file="wht_recall_uplift.png",
):
    """Scatter: traditional recall (x) vs semantic recall (y) for all 5 modes."""
    colors = ["#4c72b0", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, mode_label in enumerate(mode_labels):
        trad_v, sem_v = [], []
        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(mode_label)
            if rm:
                sem = rm["semantic_recall"]
                if not (isinstance(sem, float) and sem != sem):
                    trad_v.append(rm["traditional_recall"])
                    sem_v.append(sem)
        if trad_v:
            ax.scatter(trad_v, sem_v, label=mode_label,
                       color=colors[i], s=70, alpha=0.75, edgecolors="white")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="y=x (equal)")
    ax.set_xlabel("Traditional Recall@k", fontsize=11)
    ax.set_ylabel("Semantic Recall@k",    fontsize=11)
    ax.set_title("Traditional vs Semantic Recall — all modes\n"
                 "(points above y=x benefit from WHT smoothing)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Recall uplift scatter saved to {output_file}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(dataset_root: str):
    # ------------------------------------------------------------------
    # 1. Load CVEs
    # ------------------------------------------------------------------
    ids, titles, docs = [], [], []
    print("Loading CVE JSON files…")
    for _, j in tqdm(iter_cve_json(dataset_root)):
        cve_id, title, text = extract_text(j)
        ids.append(cve_id); titles.append(title); docs.append(text)
    if not docs:
        raise SystemExit("No CVE files found in dataset directory.")
    print(f"Loaded {len(docs):,} CVEs")

    # ------------------------------------------------------------------
    # 2. Embeddings
    # ------------------------------------------------------------------
    emb = build_embeddings(docs)
    qemb = build_embeddings(
        QUERIES,
        cache_file=str(Path(__file__).parent.parent / "cve_queries_emb_cache.npy"),
    )

    # ------------------------------------------------------------------
    # 3. Build ArrowSpace
    # ------------------------------------------------------------------
    print("Building ArrowSpace…")
    t0 = time.perf_counter()
    aspace, gl = (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(enabled=False, eps=None)
        .with_sampling("simple", 1.0)
    ).build(graph_params, emb)
    print(f"ArrowSpace build time: {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Fit Graph-WHT filter on L_F
    # ------------------------------------------------------------------
    print("Fitting Graph-WHT filter on L_F…")
    t1 = time.perf_counter()
    wht_filter = GraphWHTFilter(gl)
    wht_filter.fit()
    print(
        f"WHT filter fitted: F={wht_filter.F}, P={wht_filter.P} "
        f"(pad={wht_filter.P - wht_filter.F}), "
        f"fit time={time.perf_counter() - t1:.2f}s"
    )
    print(
        f"  Laplacian spectral gap: "
        f"{wht_filter.eigenvalues[1] - wht_filter.eigenvalues[0]:.6f}"
    )

    # ------------------------------------------------------------------
    # 5. Define search modes
    # ------------------------------------------------------------------
    #   label      tau          uses WHT pre-filter?
    MODE_LABELS = ["Cosine", "Hybrid", "Taumode", "WHT-LP", "WHT-Tau"]
    MODE_TAUS   = [TAU_COSINE, TAU_HYBRID, TAU_TAUMODE, TAU_WHT_LP, TAU_WHT_TAU]
    MODE_WHT    = [False,      False,      False,        True,       True]

    h_lp = heat_kernel(WHT_T)   # low-pass heat kernel for query pre-filtering

    # ------------------------------------------------------------------
    # 6. Search loop
    # ------------------------------------------------------------------
    print(f"\nSearching {len(QUERIES)} queries over {len(MODE_LABELS)} modes…")
    all_results: list = []
    comparison_metrics: list = []
    all_wht_diag: list = []

    # Cosine is reference for NDCG / recall ground-truth
    for qi, q in enumerate(QUERIES):
        print(f"\n{'='*70}")
        print(f"Query {qi+1}/{len(QUERIES)}: {q}")
        print("="*70)

        q_raw = np.array(qemb[qi], dtype=np.float64)
        q_wht = wht_filter.apply(q_raw, h=h_lp)

        # WHT diagnostics for this query
        diag = wht_query_diagnostics(q_raw, q_wht, wht_filter)
        all_wht_diag.append(diag)
        print(
            f"  WHT diagnostics: energy_retention={diag['energy_retention']:.4f}  "
            f"cos_sim={diag['cosine_similarity']:.4f}  "
            f"spectral_shift={diag['spectral_shift']:+.4f}"
        )

        # Run all 5 modes
        mode_results = []
        for mode_label, tau, use_wht in zip(MODE_LABELS, MODE_TAUS, MODE_WHT):
            q_vec   = q_wht if use_wht else q_raw
            results = aspace.search(q_vec, gl, tau=tau)
            mode_results.append(results)

        # Trim to minimum length across modes
        min_len = min(len(r) for r in mode_results)
        mode_results = [r[:min_len] for r in mode_results]
        all_results.append(mode_results)

        # Ground truth = Cosine (mode 0)
        gt_ids    = [idx for idx, _  in mode_results[0]]
        gt_scores = [sc  for _,  sc  in mode_results[0]]

        # Ranking correlation vs Cosine
        spearman_vs_cosine = []
        kendall_vs_cosine  = []
        ndcg_vs_cosine     = []
        for mode_label, results in zip(MODE_LABELS[1:], mode_results[1:]):
            rho, tau_k = compute_ranking_metrics(results, mode_results[0])
            ndcg       = compute_ndcg(results, mode_results[0], k=min(10, min_len))
            spearman_vs_cosine.append(rho)
            kendall_vs_cosine.append(tau_k)
            ndcg_vs_cosine.append(ndcg)

        # Tail quality
        tail_metrics = analyze_tail_distribution(
            mode_results, MODE_LABELS, k_head=3, k_tail=K_TAIL_MAX
        )

        # Semantic recall
        recall_metrics_per_mode = {}
        for mode_label, results in zip(MODE_LABELS, mode_results):
            ret_ids    = [idx for idx, _  in results]
            ret_scores = [sc  for _,  sc  in results]
            recall_metrics_per_mode[mode_label] = compute_all_recall_metrics(
                ret_ids, ret_scores, gt_ids, gt_scores
            )

        comparison_metrics.append({
            "query":              q,
            "min_length":         min_len,
            "spearman_vs_cosine": spearman_vs_cosine,
            "kendall_vs_cosine":  kendall_vs_cosine,
            "ndcg_vs_cosine":     ndcg_vs_cosine,
            "tail_metrics":       tail_metrics,
            "recall_metrics":     recall_metrics_per_mode,
            "wht_diagnostics":    diag,
        })

        # Print per-query console summary
        print(f"\n  {'Mode':<12} {'ρ vs cos':>10} {'τ vs cos':>10} {'NDCG@10':>10}")
        print(f"  {'-'*46}")
        for mode_label, rho, tau_k, ndcg in zip(
            MODE_LABELS[1:], spearman_vs_cosine, kendall_vs_cosine, ndcg_vs_cosine
        ):
            print(f"  {mode_label:<12} {rho:>10.4f} {tau_k:>10.4f} {ndcg:>10.4f}")

        print(f"\n  Recall Metrics  (Kuffo et al., SIGIR '26)")
        print(f"  {'Mode':<12} {'Trad':>8} {'Sem':>8} {'Tol':>8} {'#SN':>6}")
        print(f"  {'-'*46}")
        for mode_label in MODE_LABELS:
            rm  = recall_metrics_per_mode[mode_label]
            sem = rm["semantic_recall"]
            is_nan = isinstance(sem, float) and sem != sem
            sem_s  = "    n/a" if is_nan else f"{sem:8.4f}"
            print(
                f"  {mode_label:<12} "
                f"{rm['traditional_recall']:8.4f} {sem_s} "
                f"{rm['tolerant_recall']:8.4f} {rm['n_semantic_neighbors']:6d}"
            )

        # Show top-5 for each mode
        for mode_label, results in zip(MODE_LABELS, mode_results):
            print(f"\n  [{mode_label}] top-5:")
            for rank, (idx, score) in enumerate(results[:5], 1):
                print(f"    {rank}. {ids[idx]:<18} {titles[idx]:<38} [{score:.4f}]")

    # ------------------------------------------------------------------
    # 7. Aggregated console summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("AGGREGATED SUMMARY")
    print("="*70)

    print(f"\n  Mode NDCG@10 vs Cosine (avg ± std):")
    for i, mode_label in enumerate(MODE_LABELS[1:]):
        vals = [m["ndcg_vs_cosine"][i] for m in comparison_metrics]
        print(f"    {mode_label:<12}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Average Recall Metrics:")
    print(f"  {'Mode':<12} {'Trad':>10} {'Sem':>10} {'Tol':>10}")
    print(f"  {'-'*46}")
    for mode_label in MODE_LABELS:
        trad_v, sem_v, tol_v = [], [], []
        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(mode_label)
            if rm:
                trad_v.append(rm["traditional_recall"])
                s = rm["semantic_recall"]
                if not (isinstance(s, float) and s != s):
                    sem_v.append(s)
                tol_v.append(rm["tolerant_recall"])
        print(
            f"  {mode_label:<12} "
            f"{np.mean(trad_v) if trad_v else float('nan'):10.4f} "
            f"{np.mean(sem_v)  if sem_v  else float('nan'):10.4f} "
            f"{np.mean(tol_v)  if tol_v  else float('nan'):10.4f}"
        )

    print(f"\n  WHT diagnostics (avg over queries):")
    for key in ["energy_retention", "cosine_similarity",
                "low_freq_fraction", "spectral_shift"]:
        vals = [m["wht_diagnostics"][key] for m in comparison_metrics]
        print(f"    {key:<30}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ------------------------------------------------------------------
    # 8. Visualisations
    # ------------------------------------------------------------------
    print("\nGenerating plots…")
    plot_wht_spectral_energy(QUERIES, all_wht_diag, wht_filter, qemb,
                             "wht_spectral_energy.png")
    plot_mode_comparison(QUERIES, all_results, MODE_LABELS, comparison_metrics,
                         "wht_mode_comparison.png")
    plot_wht_recall_uplift(comparison_metrics, MODE_LABELS,
                           "wht_recall_uplift.png")

    # ------------------------------------------------------------------
    # 9. CSV exports
    # ------------------------------------------------------------------
    print("\nExporting CSVs…")
    save_wht_diagnostics_csv(all_wht_diag, QUERIES)
    save_search_results_csv(QUERIES, all_results, MODE_LABELS, ids, titles)
    save_ranking_metrics_csv(comparison_metrics, MODE_LABELS)
    save_recall_metrics_csv(comparison_metrics, MODE_LABELS)
    save_summary_csv(comparison_metrics, MODE_LABELS)
    print("\nDone.")


# ============================================================================
# Queries  (same set as test_17)
# ============================================================================
QUERIES = [
    "authenticated arbitrary file read path traversal",
    "remote code execution in ERP web component",
    "SQL injection in login endpoint",
    "stored cross-site scripting XSS in user profile page",
    "server-side request forgery SSRF in URL preview feature",
    "XML external entity XXE injection in SOAP parser",
    "insecure direct object reference IDOR in invoice download",
    "heap buffer overflow in image processing library",
    "local privilege escalation via race condition in kernel",
    "use-after-free vulnerability in browser rendering engine",
    "integer overflow leading to heap corruption in video codec",
    "authentication bypass via JWT token manipulation",
    "unsafe deserialization in Java RMI service",
    "improper access control in REST API DELETE method",
    "command injection in router web administration interface",
    "hardcoded credentials in firmware update mechanism",
    "denial of service via malformed network packets",
    "sensitive information disclosure in cloud metadata service",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CVE end-to-end benchmark with Graph-WHT spectral filter"
    )
    parser.add_argument("--dataset", required=True,
                        help="Root directory containing CVE JSON files")
    args = parser.parse_args()
    main(args.dataset)
