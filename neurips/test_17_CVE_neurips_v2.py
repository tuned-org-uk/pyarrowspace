"""
CVE semantic search with pyarrowspace - peer-review ready multi-metric evaluation.

This test compares three ranking modes on a CVE corpus:
- Cosine baseline            (tau = 1.0)
- Hybrid ArrowSpace mode     (tau = 0.72)
- Taumode ArrowSpace mode    (tau = 0.42)

The benchmark evaluates:
1. Ranking agreement metrics (Spearman, Kendall)
2. NDCG@10 against cosine reference
3. Head-tail quality metrics
4. Semantic recall style metrics inspired by:
   Kuffo et al., "Semantic Recall for Vector Search"
   https://doi.org/10.1145/3805712.3809894
5. Sensitivity of head-tail analysis to the head definition HEAD_K

Important methodological note
-----------------------------
In this script, cosine ranking is used as the internal reference ranking G for
traditional / semantic / tolerant recall-style comparisons. This is a benchmark
design choice for comparative analysis, not a claim that cosine is the external
ground truth of semantic relevance.

Outputs:
- cve_search_results.csv
- cve_comparison_metrics.csv
- cve_tail_metrics.csv
- cve_semantic_recall_metrics.csv
- cve_summary.csv
- query_comparison.txt
- cve_top25_comparison.png
- cve_tail_analysis.png
- cve_semantic_recall_comparison.png

Additional outputs:
- cve_run_metadata.json
- cve_metric_deltas.png
- cve_win_loss_heatmap.png
- cve_pareto_tradeoff.png
- cve_headk_sweep.csv
- cve_headk_sweep.png

Usage:
    python test_17_CVE_semantic_recall.py --dataset /path/to/cvelistV5
"""

import argparse
import csv
import glob
import json
import logging
import math
import os
import time
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from arrowspace import ArrowSpaceBuilder, set_debug
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
set_debug(True)

# ============================================================================
# Configuration
# ============================================================================

START_YEAR = 1999
END_YEAR = 2026

TAU_COSINE = 1.0
TAU_HYBRID = 0.72
TAU_TAUMODE = 0.42

NDCG_K = 25
RESULTS_K = 25
NEIGHBOUR_K = 30
HEAD_K = 3
HEAD_K_SWEEP = [3, 5, 10]
QUERY_EMB_CACHE = Path(__file__).parent.parent / "cve_queries_emb_cache.npy"
OUTPUT_DIR = Path(__file__).parent / "output"

# as computed by the script 02
graph_params = {
    "eps": 1.31,
    "k": NEIGHBOUR_K,
    "topk": RESULTS_K,
    "p": 1.8,
    "sigma": 0.535,
}

TAU_METHOD_KEYS = ["Cosine", "Hybrid", "Taumode"]
TAU_DISPLAY = {
    "Cosine": f"Cosine (τ={TAU_COSINE})",
    "Hybrid": f"hybrid (τ={TAU_HYBRID})",
    "Taumode": f"taumode (τ={TAU_TAUMODE})",
}
TAU_VALUES = {
    "Cosine": TAU_COSINE,
    "Hybrid": TAU_HYBRID,
    "Taumode": TAU_TAUMODE,
}
METHOD_COLORS = {
    "Cosine": "#1f77b4",
    "Hybrid": "#ff7f0e",
    "Taumode": "#2ca02c",
}

# ============================================================================
# Data Loading
# ============================================================================

def iter_cve_json(root_dir, start=START_YEAR, end=END_YEAR):
    """Iterate over CVE JSON files in date range."""
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end + 1)):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    yield path, json.load(f)
                except Exception:
                    continue


def extract_text(j):
    """Extract searchable text from CVE JSON."""
    cve_id = j.get("cveMetadata", {}).get("cveId", "")
    cna = j.get("containers", {}).get("cna", {})
    title = cna.get("title", "") or ""

    descs = []
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
    cwe_str = " ".join(cwes)

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
        vendor = a.get("vendor") or ""
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
    model_path=str(Path(__file__).parent.parent / "domain_adapted_model"),
    cache_file="cve_embeddings_cache.npy",
):
    """
    Generate embeddings using fine-tuned model.
    Loads from disk if cache_file exists; otherwise generates and saves.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        try:
            X = np.load(cache_file)
            if len(X) != len(texts):
                print(
                    f"Warning: Cache size ({len(X)}) does not match text size ({len(texts)}). Regenerating..."
                )
            else:
                print(f"Embeddings loaded. Shape: {X.shape}")
                return X
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating...")

    print(f"Cache not found. Loading model from: {model_path}")
    model = SentenceTransformer(model_path)

    print("Encoding texts...")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    X_scaled = X.astype(np.float64) * 1.2e1

    print(f"Saving embeddings to {cache_file}...")
    np.save(cache_file, X_scaled)

    print(
        f"Embeddings generated. Shape: {X_scaled.shape}, sample: {X_scaled[0][:5]}..."
    )
    return X_scaled

# ============================================================================
# Core Metrics
# ============================================================================

def compute_ranking_metrics(results_a, results_b):
    """Compute Spearman rho and Kendall tau between two rankings."""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]

    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0

    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]

    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau, _ = kendalltau(rank_a, rank_b)

    spearman_rho = 0.0 if np.isnan(spearman_rho) else float(spearman_rho)
    kendall_tau = 0.0 if np.isnan(kendall_tau) else float(kendall_tau)
    return spearman_rho, kendall_tau


def compute_ndcg(results_pred, results_ref, k=NDCG_K):
    """Compute NDCG@k treating results_ref as the reference ranking."""
    ref_indices = [idx for idx, _ in results_ref[:k]]
    relevance_map = {idx: k - i for i, idx in enumerate(ref_indices)}

    pred_indices = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]

    if sum(true_relevance) == 0:
        return 0.0

    try:
        pred_scores = np.array([score for _, score in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()

        return float(
            ndcg_score(
                np.array([true_relevance]).reshape(1, -1),
                np.array([pred_scores]).reshape(1, -1),
                k=k,
            )
        )
    except Exception:
        return 0.0


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    """
    Analyze ranking-shape quality by comparing the head and tail score profiles.

    Metrics:
    - head_mean
    - tail_mean
    - tail_std
    - tail_to_head_ratio
    - tail_cv
    - tail_decay_rate
    """
    min_length = min(len(results) for results in results_list)
    if min_length <= k_head:
        return {}

    actual_k_tail = min(k_tail, min_length)
    metrics = {}

    for results, label in zip(results_list, labels):
        results_segment = results[:actual_k_tail]
        head_scores = [score for _, score in results_segment[:k_head]]
        tail_scores = [score for _, score in results_segment[k_head:actual_k_tail]]

        if not tail_scores or not head_scores:
            continue

        tail_mean = float(np.mean(tail_scores))
        tail_std = float(np.std(tail_scores))
        head_mean = float(np.mean(head_scores))

        tail_to_head_ratio = tail_mean / head_mean if head_mean > 1e-10 else 0.0
        tail_cv = tail_std / tail_mean if tail_mean > 1e-10 else 0.0
        tail_decay = (
            (tail_scores[0] - tail_scores[-1]) / len(tail_scores)
            if len(tail_scores) > 1
            else 0.0
        )

        metrics[label] = {
            "head_mean": head_mean,
            "tail_mean": tail_mean,
            "tail_std": tail_std,
            "tail_to_head_ratio": float(tail_to_head_ratio),
            "tail_cv": float(tail_cv),
            "tail_decay_rate": float(tail_decay),
            "n_tail_items": len(tail_scores),
            "total_items": actual_k_tail,
        }

    return metrics


# ============================================================================
# Semantic Recall Metrics
# ============================================================================

def compute_traditional_recall(retrieved_ids: list, ground_truth_ids: list) -> float:
    """Traditional recall@k = |R ∩ G| / |G|."""
    if not ground_truth_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)
    return len(retrieved_set & gt_set) / len(gt_set)


def compute_semantic_recall(
    retrieved_ids: list,
    ground_truth_ids: list,
    semantic_neighbor_ids: list,
) -> float:
    """
    Semantic recall@k = |R ∩ SN| / |SN|.

    Here SN is approximated via score-gap thresholding on the cosine reference ranking.
    """
    sn_set = set(semantic_neighbor_ids) & set(ground_truth_ids)
    if not sn_set:
        return float("nan")
    retrieved_set = set(retrieved_ids)
    return len(retrieved_set & sn_set) / len(sn_set)


def compute_tolerant_recall(
    retrieved_ids: list,
    retrieved_scores: list,
    ground_truth_ids: list,
    ground_truth_scores: list,
    tolerance_pct: float = 1.0,
) -> float:
    """
    Tolerant recall@k:
    allows score-close substitutions relative to the reference ranking.
    """
    if not ground_truth_ids:
        return 0.0

    k = len(ground_truth_ids)
    gt_score_map = {idx: sc for idx, sc in zip(ground_truth_ids, ground_truth_scores)}

    matched_gt = set()
    matched_count = 0

    for ret_idx, ret_score in zip(retrieved_ids, retrieved_scores):
        if ret_idx in gt_score_map and ret_idx not in matched_gt:
            matched_gt.add(ret_idx)
            matched_count += 1
        else:
            for gt_idx, gt_score in zip(ground_truth_ids, ground_truth_scores):
                if gt_idx in matched_gt:
                    continue
                threshold = gt_score * (1.0 - tolerance_pct / 100.0)
                if ret_score >= threshold:
                    matched_gt.add(gt_idx)
                    matched_count += 1
                    break

    return matched_count / k


def estimate_tolerance_threshold(ground_truth_scores: list, k: int) -> float:
    """
    Proxy tolerance threshold based on score spread near the lower tail of G.
    """
    scores = list(ground_truth_scores)[:k]
    if len(scores) < 2:
        return 1.0
    max_score = max(scores) if max(scores) > 0 else 1.0
    two_thirds_k = max(0, int(2 * k / 3) - 1)
    score_2k3 = scores[two_thirds_k]
    score_k = scores[-1]
    threshold_pct = abs(score_2k3 - score_k) / max_score * 100.0
    return max(0.1, threshold_pct)


def identify_semantic_neighbors(
    ground_truth_ids: list,
    ground_truth_scores: list,
    score_gap_percentile: float = 25.0,
) -> list:
    """
    Approximate semantic neighbors by taking the upper score band within G.
    """
    if not ground_truth_scores:
        return []
    scores = np.array(ground_truth_scores)
    threshold = np.percentile(scores, 100 - score_gap_percentile)
    return [
        idx
        for idx, sc in zip(ground_truth_ids, ground_truth_scores)
        if sc >= threshold
    ]


def compute_all_recall_metrics(
    retrieved_ids: list,
    retrieved_scores: list,
    ground_truth_ids: list,
    ground_truth_scores: list,
    tolerance_pct=None,
    sn_score_gap_percentile: float = 25.0,
) -> dict:
    """
    Compute traditional, semantic, and tolerant recall in one call.

    Inspired by Kuffo et al., "Semantic Recall for Vector Search", SIGIR '26
    (https://doi.org/10.1145/3805712.3809894). The cosine ranking is used as the
    internal reference G; semantic neighbors SN are identified via score-gap
    thresholding (proxy for an LLM judge).
    """
    k = len(ground_truth_ids)

    # 1. Traditional recall: |R ∩ G| / |G|
    trad = compute_traditional_recall(retrieved_ids, ground_truth_ids)

    # 2. Semantic recall: |R ∩ SN| / |SN|, with SN approximated via score-gap
    sn_ids = identify_semantic_neighbors(
        ground_truth_ids, ground_truth_scores, sn_score_gap_percentile
    )
    sem = compute_semantic_recall(retrieved_ids, ground_truth_ids, sn_ids)

    # 3. Tolerant recall: greedy match allowing score-close substitutions
    if tolerance_pct is None:
        tolerance_pct = estimate_tolerance_threshold(ground_truth_scores, k)

    tol = compute_tolerant_recall(
        retrieved_ids,
        retrieved_scores,
        ground_truth_ids,
        ground_truth_scores,
        tolerance_pct=tolerance_pct,
    )

    return {
        "traditional_recall": float(trad),
        "semantic_recall": sem,
        "tolerant_recall": float(tol),
        "n_semantic_neighbors": int(len(sn_ids)),
        "tolerance_pct_used": float(tolerance_pct),
    }


# ============================================================================
# Export Helpers
# ============================================================================

def save_run_metadata(
    output_file,
    dataset_root,
    n_docs,
    embedding_model_path,
    query_count,
    comparison_metrics,
):
    """Persist run metadata for auditability and peer review."""
    min_lengths = [m["min_length"] for m in comparison_metrics] if comparison_metrics else []
    payload = {
        "test_name": "test_17_CVE_semantic_recall",
        "timestamp_unix": time.time(),
        "dataset_root": str(dataset_root),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "n_cve_documents_loaded": int(n_docs),
        "embedding_model_path": str(embedding_model_path),
        "tau_values": TAU_VALUES,
        "head_k": HEAD_K,
        "head_k_sweep": HEAD_K_SWEEP,
        "neighbour_K": NEIGHBOUR_K,
        "graph_params": graph_params,
        "query_count": int(query_count),
        "ndcg_k": NDCG_K,
        "results_k": RESULTS_K,
        "min_result_length_observed": int(min(min_lengths)) if min_lengths else None,
        "max_result_length_observed": int(max(min_lengths)) if min_lengths else None,
        "note": (
            "Cosine ranking is used as the internal reference ranking for recall-style "
            "comparisons. Existing CSV and plot outputs are preserved for compatibility."
        ),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Run metadata saved to {output_file}")


def save_search_results_to_csv(queries, all_results, ids, titles, output_file=OUTPUT_DIR / "cve_search_results.csv"):
    """Save top search results for all queries and modes."""
    tau_labels = ["Cosine", "hybrid", "taumode"]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "query_id", "query_text", "tau_method", "rank", "cve_id", "title", "score"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, query in enumerate(queries):
            results_cosine, results_hybrid, results_taumode = all_results[qi]
            for tau_label, results in zip(
                tau_labels, [results_cosine, results_hybrid, results_taumode]
            ):
                for rank, (idx, score) in enumerate(results[:RESULTS_K], 1):
                    writer.writerow({
                        "query_id": qi + 1,
                        "query_text": query,
                        "tau_method": tau_label,
                        "rank": rank,
                        "cve_id": ids[idx],
                        "title": titles[idx],
                        "score": f"{score:.6f}",
                    })
    print(f"Search results saved to {output_file}")


def save_metrics_to_csv(comparison_metrics, output_file=OUTPUT_DIR / "cve_comparison_metrics.csv"):
    """Save ranking agreement and NDCG metrics to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "query_id",
            "query_text",
            "min_length",
            "spearman_cosine_hybrid",
            "spearman_cosine_taumode",
            "spearman_hybrid_taumode",
            "kendall_cosine_hybrid",
            "kendall_cosine_taumode",
            "kendall_hybrid_taumode",
            "ndcg_hybrid_vs_cosine",
            "ndcg_taumode_vs_cosine",
            "ndcg_taumode_vs_hybrid",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, metrics in enumerate(comparison_metrics):
            writer.writerow({
                "query_id": qi + 1,
                "query_text": metrics["query"],
                "min_length": metrics["min_length"],
                "spearman_cosine_hybrid": f"{metrics['spearman'][0]:.6f}",
                "spearman_cosine_taumode": f"{metrics['spearman'][1]:.6f}",
                "spearman_hybrid_taumode": f"{metrics['spearman'][2]:.6f}",
                "kendall_cosine_hybrid": f"{metrics['kendall'][0]:.6f}",
                "kendall_cosine_taumode": f"{metrics['kendall'][1]:.6f}",
                "kendall_hybrid_taumode": f"{metrics['kendall'][2]:.6f}",
                "ndcg_hybrid_vs_cosine": f"{metrics['ndcg'][0]:.6f}",
                "ndcg_taumode_vs_cosine": f"{metrics['ndcg'][1]:.6f}",
                "ndcg_taumode_vs_hybrid": f"{metrics['ndcg'][2]:.6f}",
            })
    print(f"Comparison metrics saved to {output_file}")


def save_tail_metrics_to_csv(comparison_metrics, output_file=OUTPUT_DIR / "cve_tail_metrics.csv"):
    """Save head-tail metrics to CSV."""
    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "query_id",
            "query_text",
            "tau_method",
            "head_mean",
            "tail_mean",
            "tail_std",
            "tail_to_head_ratio",
            "tail_cv",
            "tail_decay_rate",
            "n_tail_items",
            "total_items",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, metrics in enumerate(comparison_metrics):
            tail_metrics = metrics.get("tail_metrics", {})
            if not tail_metrics:
                continue

            for tau_label in tau_labels:
                if tau_label in tail_metrics:
                    m = tail_metrics[tau_label]
                    writer.writerow({
                        "query_id": qi + 1,
                        "query_text": metrics["query"],
                        "tau_method": tau_label,
                        "head_mean": f"{m['head_mean']:.6f}",
                        "tail_mean": f"{m['tail_mean']:.6f}",
                        "tail_std": f"{m['tail_std']:.6f}",
                        "tail_to_head_ratio": f"{m['tail_to_head_ratio']:.6f}",
                        "tail_cv": f"{m['tail_cv']:.6f}",
                        "tail_decay_rate": f"{m['tail_decay_rate']:.6f}",
                        "n_tail_items": m["n_tail_items"],
                        "total_items": m["total_items"],
                    })
    print(f"Tail metrics saved to {output_file}")


def save_semantic_recall_to_csv(comparison_metrics, output_file=OUTPUT_DIR / "cve_semantic_recall_metrics.csv"):
    """Save traditional / semantic / tolerant recall metrics to CSV."""
    tau_keys = ["Cosine", "Hybrid", "Taumode"]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "query_id",
            "query_text",
            "tau_method",
            "traditional_recall",
            "semantic_recall",
            "tolerant_recall",
            "n_semantic_neighbors",
            "tolerance_pct_used",
            "tolerant_minus_traditional",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, m in enumerate(comparison_metrics):
            recall_metrics = m.get("recall_metrics", {})
            if not recall_metrics:
                continue
            for tau_key in tau_keys:
                if tau_key not in recall_metrics:
                    continue
                rm = recall_metrics[tau_key]
                trad = rm["traditional_recall"]
                sem = rm["semantic_recall"]
                is_nan = isinstance(sem, float) and math.isnan(sem)
                sem_str = "nan" if is_nan else f"{sem:.6f}"
                tol = rm["tolerant_recall"]
                diff = tol - trad
                diff_str = f"{diff:.6f}"
                writer.writerow({
                    "query_id": qi + 1,
                    "query_text": m["query"],
                    "tau_method": tau_key,
                    "traditional_recall": f"{trad:.6f}",
                    "semantic_recall": sem_str,
                    "tolerant_recall": f"{rm['tolerant_recall']:.6f}",
                    "n_semantic_neighbors": rm["n_semantic_neighbors"],
                    "tolerance_pct_used": f"{rm['tolerance_pct_used']:.4f}",
                    "tolerant_minus_traditional": diff_str,
                })
    print(f"Semantic recall metrics saved to {output_file}")


def save_summary_to_csv(comparison_metrics, output_file=OUTPUT_DIR / "cve_summary.csv"):
    """Save aggregate summary statistics to CSV."""
    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]

    avg_ndcg_h_c = np.mean([m["ndcg"][0] for m in comparison_metrics])
    avg_ndcg_t_c = np.mean([m["ndcg"][1] for m in comparison_metrics])
    avg_ndcg_t_h = np.mean([m["ndcg"][2] for m in comparison_metrics])
    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["metric_type", "metric_name", "value", "std_dev"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            "metric_type": f"NDCG@{NDCG_K}",
            "metric_name": "hybrid vs Cosine",
            "value": f"{avg_ndcg_h_c:.6f}",
            "std_dev": f"{np.std([m['ndcg'][0] for m in comparison_metrics]):.6f}",
        })
        writer.writerow({
            "metric_type": f"NDCG@{NDCG_K}",
            "metric_name": "taumode vs Cosine",
            "value": f"{avg_ndcg_t_c:.6f}",
            "std_dev": f"{np.std([m['ndcg'][1] for m in comparison_metrics]):.6f}",
        })
        writer.writerow({
            "metric_type": f"NDCG@{NDCG_K}",
            "metric_name": "taumode vs hybrid",
            "value": f"{avg_ndcg_t_h:.6f}",
            "std_dev": f"{np.std([m['ndcg'][2] for m in comparison_metrics]):.6f}",
        })

        if valid_tail:
            for label in tau_labels:
                ratios = []
                for m in valid_tail:
                    if label in m["tail_metrics"]:
                        ratios.append(m["tail_metrics"][label]["tail_to_head_ratio"])
                if ratios:
                    writer.writerow({
                        "metric_type": "Tail/Head Ratio",
                        "metric_name": label,
                        "value": f"{np.mean(ratios):.6f}",
                        "std_dev": f"{np.std(ratios):.6f}",
                    })

        for tau_key in ["Cosine", "Hybrid", "Taumode"]:
            trad_vals, sem_vals, tol_vals = [], [], []
            for m in comparison_metrics:
                rm = m.get("recall_metrics", {}).get(tau_key)
                if rm:
                    trad_vals.append(rm["traditional_recall"])
                    s = rm["semantic_recall"]
                    if not (isinstance(s, float) and math.isnan(s)):
                        sem_vals.append(s)
                    tol_vals.append(rm["tolerant_recall"])

            if trad_vals:
                writer.writerow({
                    "metric_type": "Traditional Recall@k",
                    "metric_name": tau_key,
                    "value": f"{np.mean(trad_vals):.6f}",
                    "std_dev": f"{np.std(trad_vals):.6f}",
                })
            if sem_vals:
                writer.writerow({
                    "metric_type": "Semantic Recall@k",
                    "metric_name": tau_key,
                    "value": f"{np.mean(sem_vals):.6f}",
                    "std_dev": f"{np.std(sem_vals):.6f}",
                })
            if tol_vals:
                writer.writerow({
                    "metric_type": "Tolerant Recall@k",
                    "metric_name": tau_key,
                    "value": f"{np.mean(tol_vals):.6f}",
                    "std_dev": f"{np.std(tol_vals):.6f}",
                })

    print(f"Summary statistics saved to {output_file}")


def save_headk_sweep_to_csv(headk_sweep_rows, output_file=OUTPUT_DIR / "cve_headk_sweep.csv"):
    """Save HEAD_K sweep results to CSV."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "head_k",
            "query_id",
            "query_text",
            "tau_method",
            "head_mean",
            "tail_mean",
            "tail_std",
            "tail_to_head_ratio",
            "tail_cv",
            "tail_decay_rate",
            "n_tail_items",
            "total_items",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in headk_sweep_rows:
            writer.writerow(row)
    print(f"HEAD_K sweep saved to {output_file}")


def run_headk_sweep(queries, all_results, head_k_values):
    """
    Recompute tail metrics for multiple head definitions using the same rankings.

    This isolates the effect of the head/tail split from the retrieval process itself.
    """
    rows = []

    for h in head_k_values:
        for qi, query in enumerate(queries):
            results_cosine, results_hybrid, results_taumode = all_results[qi]
            min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))

            if min_len <= h or (min_len - h) < 2:
                continue

            results_trimmed = [
                results_cosine[:min_len],
                results_hybrid[:min_len],
                results_taumode[:min_len],
            ]

            tau_labels = [
                TAU_DISPLAY["Cosine"],
                TAU_DISPLAY["Hybrid"],
                TAU_DISPLAY["Taumode"],
            ]

            tail_metrics = analyze_tail_distribution(
                results_trimmed,
                tau_labels,
                k_head=h,
                k_tail=min_len,
            )

            for tau_label in tau_labels:
                if tau_label not in tail_metrics:
                    continue
                m = tail_metrics[tau_label]
                rows.append({
                    "head_k": h,
                    "query_id": qi + 1,
                    "query_text": query,
                    "tau_method": tau_label,
                    "head_mean": f"{m['head_mean']:.6f}",
                    "tail_mean": f"{m['tail_mean']:.6f}",
                    "tail_std": f"{m['tail_std']:.6f}",
                    "tail_to_head_ratio": f"{m['tail_to_head_ratio']:.6f}",
                    "tail_cv": f"{m['tail_cv']:.6f}",
                    "tail_decay_rate": f"{m['tail_decay_rate']:.6f}",
                    "n_tail_items": m["n_tail_items"],
                    "total_items": m["total_items"],
                })

    return rows


# ============================================================================
# Visualizations
# ============================================================================

def plot_comparison(queries, all_results, ids, titles, output_file=OUTPUT_DIR / "cve_top25_comparison.png"):
    """Plot top-k results across the three modes."""
    n_queries = len(queries)
    fig, axes = plt.subplots(n_queries, 3, figsize=(18, 6 * n_queries))
    if n_queries == 1:
        axes = axes.reshape(1, -1)

    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]
    colors = [METHOD_COLORS["Cosine"], METHOD_COLORS["Hybrid"], METHOD_COLORS["Taumode"]]

    for qi, query in enumerate(queries):
        results_cosine, results_hybrid, results_taumode = all_results[qi]
        k = min(RESULTS_K, min(len(results_cosine), len(results_hybrid), len(results_taumode)))

        for ti, (results, label, color) in enumerate(
            zip([results_cosine, results_hybrid, results_taumode], tau_labels, colors)
        ):
            ax = axes[qi, ti]
            scores = [score for _, score in results[:k]]
            ranks = list(range(1, k + 1))

            ax.bar(ranks, scores, alpha=0.75, color=color)
            ax.set_xlabel("Rank", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"Q{qi + 1}: {label}\n{query[:50]}...", fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            for i, (idx, score) in enumerate(results[:k]):
                ax.text(
                    i + 1,
                    score + 0.01 * max(scores) if scores else 0,
                    ids[idx].split("-")[-1],
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Top-{RESULTS_K} plot saved to {output_file}")
    plt.close()


def plot_tail_comparison(queries, all_results, ids, titles, output_file=OUTPUT_DIR / "cve_tail_analysis.png"):
    """Create per-query tail analysis panels."""
    n_queries = len(queries)
    fig = plt.figure(figsize=(20, 5 * n_queries))
    gs = fig.add_gridspec(n_queries, 4, hspace=0.3, wspace=0.3)

    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]
    colors = [METHOD_COLORS["Cosine"], METHOD_COLORS["Hybrid"], METHOD_COLORS["Taumode"]]

    for qi, query in enumerate(queries):
        results_cosine, results_hybrid, results_taumode = all_results[qi]
        k = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        results_trimmed = [results_cosine[:k], results_hybrid[:k], results_taumode[:k]]

        ax1 = fig.add_subplot(gs[qi, 0])
        ranks = list(range(1, k + 1))
        for results, label, color in zip(results_trimmed, tau_labels, colors):
            scores = [score for _, score in results]
            ax1.plot(ranks, scores, marker="o", label=label, color=color, alpha=0.75, markersize=4, linewidth=2)
        ax1.axvline(x=HEAD_K + 0.5, color="red", linestyle="--", alpha=0.5, linewidth=2, label="Head/Tail split")
        ax1.set_xlabel("Rank", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax1.set_title(f"Q{qi + 1}: Score Distribution (n={k})\n{query[:45]}...", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8, loc="best")
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[qi, 1])
        if k > HEAD_K:
            tail_ranks = list(range(HEAD_K + 1, k + 1))
            for results, label, color in zip(results_trimmed, tau_labels, colors):
                tail_scores = [score for _, score in results[HEAD_K:]]
                ax2.plot(tail_ranks, tail_scores, marker="s", label=label, color=color, alpha=0.75, markersize=5, linewidth=2)
            ax2.set_xlabel("Rank", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax2.set_title(f"Q{qi + 1}: Tail (Ranks {HEAD_K + 1}-{k})", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=8, loc="best")
            ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[qi, 2])
        if k > HEAD_K:
            tail_data = [[score for _, score in r[HEAD_K:]] for r in results_trimmed]
            bp = ax3.boxplot(
                tail_data,
                labels=["Cosine", "hybrid", "taumode"],
                patch_artist=True,
                widths=0.6,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax3.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax3.set_title(f"Q{qi + 1}: Tail Variability", fontsize=10, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

        ax4 = fig.add_subplot(gs[qi, 3])
        if k > HEAD_K:
            tail_metrics = analyze_tail_distribution(results_trimmed, tau_labels, k_head=HEAD_K, k_tail=k)
            metrics_names = ["Tail Mean", "T/H Ratio", "Stability"]
            x_pos = np.arange(len(metrics_names))
            width = 0.25

            for i, (label, color) in enumerate(zip(tau_labels, colors)):
                if label in tail_metrics:
                    m = tail_metrics[label]
                    values = [
                        m["tail_mean"],
                        m["tail_to_head_ratio"],
                        1.0 / (1.0 + m["tail_cv"]) if m["tail_cv"] > 0 else 1.0,
                    ]
                    ax4.bar(x_pos + i * width, values, width, label=label, color=color, alpha=0.75)

            ax4.set_ylabel("Value", fontsize=11, fontweight="bold")
            ax4.set_title(f"Q{qi + 1}: Tail Metrics", fontsize=10, fontweight="bold")
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(metrics_names, fontsize=9, rotation=15, ha="right")
            ax4.legend(fontsize=8, loc="best")
            ax4.grid(axis="y", alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Tail analysis plot saved to {output_file}")
    plt.close()


def plot_semantic_recall_comparison(comparison_metrics, output_file=OUTPUT_DIR / "cve_semantic_recall_comparison.png"):
    """Plot traditional / semantic / tolerant recall per method."""
    tau_methods = ["Cosine", "Hybrid", "Taumode"]
    tau_display = [TAU_DISPLAY[k] for k in tau_methods]
    method_colors = {
        "traditional": "#4c72b0",
        "semantic": "#55a868",
        "tolerant": "#dd8452",
    }

    n_methods = len(tau_methods)
    fig, axes = plt.subplots(n_methods, 3, figsize=(20, 6 * n_methods))
    fig.suptitle(
        "Semantic Recall Analysis — Traditional vs Semantic vs Tolerant Recall\n"
        "(proxy implementation inspired by Kuffo et al., SIGIR '26)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    n_queries = len(comparison_metrics)
    query_ids = list(range(1, n_queries + 1))

    for ti, (tau_key, tau_label) in enumerate(zip(tau_methods, tau_display)):
        trad_vals, sem_vals, tol_vals, n_sn_vals, valid_mask = [], [], [], [], []

        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(tau_key)
            if rm:
                trad_vals.append(rm["traditional_recall"])
                sem_raw = rm["semantic_recall"]
                is_nan = isinstance(sem_raw, float) and math.isnan(sem_raw)
                sem_vals.append(0.0 if is_nan else sem_raw)
                tol_vals.append(rm["tolerant_recall"])
                n_sn_vals.append(rm["n_semantic_neighbors"])
                valid_mask.append(not is_nan)
            else:
                trad_vals.append(0.0)
                sem_vals.append(0.0)
                tol_vals.append(0.0)
                n_sn_vals.append(0)
                valid_mask.append(False)

        x = np.arange(n_queries)
        bar_w = 0.28

        ax0 = axes[ti, 0]
        ax0.bar(x - bar_w, trad_vals, bar_w, label="Traditional", color=method_colors["traditional"], alpha=0.85)
        ax0.bar(x, sem_vals, bar_w, label="Semantic", color=method_colors["semantic"], alpha=0.85)
        ax0.bar(x + bar_w, tol_vals, bar_w, label="Tolerant", color=method_colors["tolerant"], alpha=0.85)
        ax0.set_xlabel("Query", fontsize=10)
        ax0.set_ylabel("Recall@k", fontsize=10)
        ax0.set_title(f"{tau_label}\nRecall per Query", fontsize=11, fontweight="bold")
        ax0.set_xticks(x)
        ax0.set_xticklabels([f"Q{i}" for i in query_ids], rotation=45, ha="right", fontsize=8)
        ax0.set_ylim(0, 1.15)
        ax0.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax0.legend(fontsize=8)
        ax0.grid(axis="y", alpha=0.3)

        ax1 = axes[ti, 1]
        valid_trad = [v for v, ok in zip(trad_vals, valid_mask) if ok]
        valid_sem = [v for v, ok in zip(sem_vals, valid_mask) if ok]
        valid_sn = [v for v, ok in zip(n_sn_vals, valid_mask) if ok]

        if valid_trad:
            sc = ax1.scatter(
                valid_trad,
                valid_sem,
                c=valid_sn,
                cmap="viridis",
                s=60,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
            )
            plt.colorbar(sc, ax=ax1, label="#Semantic Neighbors", shrink=0.8)
            ax1.plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.7, label="y = x")
            ax1.set_xlabel("Traditional Recall", fontsize=10)
            ax1.set_ylabel("Semantic Recall", fontsize=10)
            ax1.set_title(f"{tau_label}\nTraditional vs Semantic Recall", fontsize=11, fontweight="bold")
            ax1.set_xlim(-0.05, 1.1)
            ax1.set_ylim(-0.05, 1.1)
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

        ax2 = axes[ti, 2]
        uplift = [tol - t for tol, t in zip(tol_vals, trad_vals)]
        if uplift:
            ax2.hist(uplift, bins=min(15, len(uplift)), color=method_colors["tolerant"], alpha=0.8, edgecolor="white")
            ax2.axvline(0, color="red", linewidth=1.5, linestyle="--", label="No uplift")
            ax2.axvline(np.mean(uplift), color="orange", linewidth=1.5, linestyle="-", label=f"Mean: {np.mean(uplift):+.3f}")
            ax2.set_xlabel("Tolerant − Traditional Recall", fontsize=10)
            ax2.set_ylabel("Query Count", fontsize=10)
            ax2.set_title(f"{tau_label}\nTolerant Recall Uplift Distribution", fontsize=11, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Semantic recall comparison plot saved to {output_file}")
    plt.close()


def plot_metric_deltas(comparison_metrics, output_file=OUTPUT_DIR / "cve_metric_deltas.png"):
    """Plot per-query deltas against cosine for key reviewer-facing metrics."""
    n_queries = len(comparison_metrics)
    query_labels = [f"Q{i+1}" for i in range(n_queries)]

    hybrid_t_h_delta = []
    taumode_t_h_delta = []
    hybrid_sem_delta = []
    taumode_sem_delta = []
    hybrid_tol_delta = []
    taumode_tol_delta = []

    for m in comparison_metrics:
        tm = m["tail_metrics"]
        cos_label = TAU_DISPLAY["Cosine"]
        hyb_label = TAU_DISPLAY["Hybrid"]
        tau_label = TAU_DISPLAY["Taumode"]

        cos_th = tm[cos_label]["tail_to_head_ratio"]
        hyb_th = tm[hyb_label]["tail_to_head_ratio"]
        tau_th = tm[tau_label]["tail_to_head_ratio"]

        hybrid_t_h_delta.append(hyb_th - cos_th)
        taumode_t_h_delta.append(tau_th - cos_th)

        cos_rm = m["recall_metrics"]["Cosine"]
        hyb_rm = m["recall_metrics"]["Hybrid"]
        tau_rm = m["recall_metrics"]["Taumode"]

        cos_sem = np.nan if (isinstance(cos_rm["semantic_recall"], float) and math.isnan(cos_rm["semantic_recall"])) else cos_rm["semantic_recall"]
        hyb_sem = np.nan if (isinstance(hyb_rm["semantic_recall"], float) and math.isnan(hyb_rm["semantic_recall"])) else hyb_rm["semantic_recall"]
        tau_sem = np.nan if (isinstance(tau_rm["semantic_recall"], float) and math.isnan(tau_rm["semantic_recall"])) else tau_rm["semantic_recall"]

        hybrid_sem_delta.append((hyb_sem - cos_sem) if not (np.isnan(hyb_sem) or np.isnan(cos_sem)) else 0.0)
        taumode_sem_delta.append((tau_sem - cos_sem) if not (np.isnan(tau_sem) or np.isnan(cos_sem)) else 0.0)
        hybrid_tol_delta.append(hyb_rm["tolerant_recall"] - cos_rm["tolerant_recall"])
        taumode_tol_delta.append(tau_rm["tolerant_recall"] - cos_rm["tolerant_recall"])

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    x = np.arange(n_queries)
    width = 0.36

    panels = [
        ("Δ Tail/Head Ratio vs Cosine", hybrid_t_h_delta, taumode_t_h_delta),
        ("Δ Semantic Recall vs Cosine", hybrid_sem_delta, taumode_sem_delta),
        ("Δ Tolerant Recall vs Cosine", hybrid_tol_delta, taumode_tol_delta),
    ]

    for ax, (title, hyb_vals, tau_vals) in zip(axes, panels):
        ax.bar(x - width / 2, hyb_vals, width=width, color=METHOD_COLORS["Hybrid"], alpha=0.8, label="hybrid - Cosine")
        ax.bar(x + width / 2, tau_vals, width=width, color=METHOD_COLORS["Taumode"], alpha=0.8, label="taumode - Cosine")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(query_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Query")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Metric delta plot saved to {output_file}")
    plt.close()


def plot_win_loss_heatmap(comparison_metrics, output_file=OUTPUT_DIR / "cve_win_loss_heatmap.png"):
    """
    Heatmap-like reviewer plot showing which method wins per query/metric.
    Metrics:
    - Tail/Head Ratio (higher better)
    - Tail CV (lower better)
    - Tail Decay (lower better)
    - Semantic Recall (higher better)
    - Tolerant Recall (higher better)
    """
    metric_names = [
        "T/H Ratio",
        "Tail CV",
        "Tail Decay",
        "Semantic Recall",
        "Tolerant Recall",
    ]
    method_to_value = {"Cosine": 0, "Hybrid": 1, "Taumode": 2}
    n_queries = len(comparison_metrics)

    win_matrix = np.zeros((len(metric_names), n_queries))
    annotations = [["" for _ in range(n_queries)] for _ in range(len(metric_names))]

    for qi, m in enumerate(comparison_metrics):
        tm = m["tail_metrics"]
        rm = m["recall_metrics"]

        values = {
            "T/H Ratio": {
                "Cosine": tm[TAU_DISPLAY["Cosine"]]["tail_to_head_ratio"],
                "Hybrid": tm[TAU_DISPLAY["Hybrid"]]["tail_to_head_ratio"],
                "Taumode": tm[TAU_DISPLAY["Taumode"]]["tail_to_head_ratio"],
            },
            "Tail CV": {
                "Cosine": tm[TAU_DISPLAY["Cosine"]]["tail_cv"],
                "Hybrid": tm[TAU_DISPLAY["Hybrid"]]["tail_cv"],
                "Taumode": tm[TAU_DISPLAY["Taumode"]]["tail_cv"],
            },
            "Tail Decay": {
                "Cosine": tm[TAU_DISPLAY["Cosine"]]["tail_decay_rate"],
                "Hybrid": tm[TAU_DISPLAY["Hybrid"]]["tail_decay_rate"],
                "Taumode": tm[TAU_DISPLAY["Taumode"]]["tail_decay_rate"],
            },
            "Semantic Recall": {
                "Cosine": -1 if (isinstance(rm["Cosine"]["semantic_recall"], float) and math.isnan(rm["Cosine"]["semantic_recall"])) else rm["Cosine"]["semantic_recall"],
                "Hybrid": -1 if (isinstance(rm["Hybrid"]["semantic_recall"], float) and math.isnan(rm["Hybrid"]["semantic_recall"])) else rm["Hybrid"]["semantic_recall"],
                "Taumode": -1 if (isinstance(rm["Taumode"]["semantic_recall"], float) and math.isnan(rm["Taumode"]["semantic_recall"])) else rm["Taumode"]["semantic_recall"],
            },
            "Tolerant Recall": {
                "Cosine": rm["Cosine"]["tolerant_recall"],
                "Hybrid": rm["Hybrid"]["tolerant_recall"],
                "Taumode": rm["Taumode"]["tolerant_recall"],
            },
        }

        higher_better = {"T/H Ratio", "Semantic Recall", "Tolerant Recall"}

        for mi, metric in enumerate(metric_names):
            mv = values[metric]
            if metric in higher_better:
                winner = max(mv, key=mv.get)
            else:
                winner = min(mv, key=mv.get)
            win_matrix[mi, qi] = method_to_value[winner]
            annotations[mi][qi] = winner[0]

    cmap = plt.matplotlib.colors.ListedColormap(
        [METHOD_COLORS["Cosine"], METHOD_COLORS["Hybrid"], METHOD_COLORS["Taumode"]]
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(win_matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)

    ax.set_xticks(np.arange(n_queries))
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_title("Per-query metric winners (C=Cosine, H=hybrid, T=taumode)", fontweight="bold")

    for i in range(len(metric_names)):
        for j in range(n_queries):
            ax.text(j, i, annotations[i][j], ha="center", va="center", color="white", fontweight="bold")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLORS["Cosine"], markersize=12, label="Cosine"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLORS["Hybrid"], markersize=12, label="hybrid"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLORS["Taumode"], markersize=12, label="taumode"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Win/loss heatmap saved to {output_file}")
    plt.close()


def plot_pareto_tradeoff(comparison_metrics, output_file=OUTPUT_DIR / "cve_pareto_tradeoff.png"):
    """
    Plot Tail/Head ratio vs Tolerant Recall to visualise ranking-shape vs recall trade-offs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for method in ["Cosine", "Hybrid", "Taumode"]:
        xs, ys = [], []
        for qi, m in enumerate(comparison_metrics, start=1):
            xs.append(m["tail_metrics"][TAU_DISPLAY[method]]["tail_to_head_ratio"])
            ys.append(m["recall_metrics"][method]["tolerant_recall"])
        ax.scatter(xs, ys, s=70, alpha=0.8, color=METHOD_COLORS[method], label=TAU_DISPLAY[method])

        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.text(x, y, f"Q{i}", fontsize=8, alpha=0.75)

    ax.set_xlabel("Tail/Head Ratio", fontweight="bold")
    ax.set_ylabel("Tolerant Recall@k", fontweight="bold")
    ax.set_title("Pareto-style view: ranking-shape quality vs tolerant recall", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Pareto trade-off plot saved to {output_file}")
    plt.close()


def plot_headk_sweep(headk_sweep_rows, output_file=OUTPUT_DIR / "cve_headk_sweep.png"):
    """
    Plot aggregate head-tail metrics across multiple HEAD_K definitions.

    Panels:
    1. Mean tail/head ratio vs HEAD_K
    2. Mean tail CV vs HEAD_K
    3. Mean tail decay vs HEAD_K
    """
    if not headk_sweep_rows:
        print("No HEAD_K sweep data available; skipping plot.")
        return

    grouped = {}
    for row in headk_sweep_rows:
        h = int(row["head_k"])
        tau_method = row["tau_method"]
        grouped.setdefault(h, {}).setdefault(tau_method, {
            "tail_to_head_ratio": [],
            "tail_cv": [],
            "tail_decay_rate": [],
        })
        grouped[h][tau_method]["tail_to_head_ratio"].append(float(row["tail_to_head_ratio"]))
        grouped[h][tau_method]["tail_cv"].append(float(row["tail_cv"]))
        grouped[h][tau_method]["tail_decay_rate"].append(float(row["tail_decay_rate"]))

    head_ks = sorted(grouped.keys())
    tau_methods = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_specs = [
        ("tail_to_head_ratio", "Mean Tail/Head Ratio", "higher is better"),
        ("tail_cv", "Mean Tail CV", "lower is better"),
        ("tail_decay_rate", "Mean Tail Decay", "lower is better"),
    ]

    color_map = {
        TAU_DISPLAY["Cosine"]: METHOD_COLORS["Cosine"],
        TAU_DISPLAY["Hybrid"]: METHOD_COLORS["Hybrid"],
        TAU_DISPLAY["Taumode"]: METHOD_COLORS["Taumode"],
    }

    for ax, (metric_key, title, subtitle) in zip(axes, metric_specs):
        for tau_method in tau_methods:
            means = []
            stds = []
            for h in head_ks:
                vals = grouped.get(h, {}).get(tau_method, {}).get(metric_key, [])
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)

            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)

            ax.plot(
                head_ks,
                means,
                marker="o",
                linewidth=2,
                color=color_map[tau_method],
                label=tau_method,
            )
            ax.fill_between(
                head_ks,
                means - stds,
                means + stds,
                color=color_map[tau_method],
                alpha=0.15,
            )

        ax.set_title(f"{title}\n({subtitle})", fontweight="bold")
        ax.set_xlabel("HEAD_K")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Metric value")
    axes[0].legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"HEAD_K sweep plot saved to {output_file}")
    plt.close()


def save_query_comparison(queries, all_results, titles, docs, output_file=OUTPUT_DIR / "query_comparison.txt"):
    """
    Save a readable cosine vs taumode comparison for best / worst / median query.

    Best and worst are determined by the top taumode score.
    """
    print(f"Generating human-readable comparison to {output_file}...")

    query_metrics = []
    for qi, q in enumerate(queries):
        res_cosine, _, res_taumode = all_results[qi]
        top_score = res_taumode[0][1] if res_taumode else 0.0
        query_metrics.append({
            "qi": qi,
            "query": q,
            "score": top_score,
            "res_cosine": res_cosine,
            "res_taumode": res_taumode,
        })

    sorted_queries = sorted(query_metrics, key=lambda x: x["score"], reverse=True)
    if not sorted_queries:
        return

    selected = []
    if len(sorted_queries) <= 3:
        labels = ["Best (Highest Confidence)", "Sample (Middle)", "Worst (Lowest Confidence)"]
        for i, q_data in enumerate(sorted_queries):
            label = labels[i] if i < len(labels) else "Query"
            selected.append((label, q_data))
    else:
        selected.append(("BEST QUERY (Highest Top Score)", sorted_queries[0]))
        selected.append(("WORST QUERY (Lowest Top Score)", sorted_queries[-1]))
        mid_idx = len(sorted_queries) // 2
        selected.append(("SAMPLE QUERY (Median Score)", sorted_queries[mid_idx]))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write(" QUERY RESULT COMPARISON: COSINE vs TAUMODE\n")
        f.write("================================================================================\n\n")

        for label, q_data in selected:
            query_text = q_data["query"]
            res_c = q_data["res_cosine"]
            res_e = q_data["res_taumode"]

            f.write(f"QUERY TYPE: {label}\n")
            f.write(f"QUERY TEXT: {query_text}\n")
            f.write("-" * 80 + "\n")

            k_show = min(RESULTS_K, max(len(res_c), len(res_e)))
            for i in range(k_show):
                f.write(f"RANK {i + 1}:\n")

                if i < len(res_c):
                    idx, score = res_c[i]
                    title = titles[idx]
                    text_snippet = docs[idx][:300].replace("\n", " ") + "..."
                    f.write(f"  [Cosine] Score: {score:.4f}\n")
                    f.write(f"           Title: {title}\n")
                    f.write(f"           Text:  {text_snippet}\n")
                else:
                    f.write("  [Cosine] No result\n")

                f.write("\n")

                if i < len(res_e):
                    idx, score = res_e[i]
                    title = titles[idx]
                    text_snippet = docs[idx][:300].replace("\n", " ") + "..."
                    f.write(f"  [taumode] Score: {score:.4f}\n")
                    f.write(f"            Title: {title}\n")
                    f.write(f"            Text:  {text_snippet}\n")
                else:
                    f.write("  [Taumode] No result\n")

                f.write("-" * 40 + "\n")

            f.write("=" * 80 + "\n\n")

    print(f"Comparison saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main(dataset_root):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, titles, docs = [], [], []

    print("Loading CVE JSON files...")
    for _, j in tqdm(iter_cve_json(dataset_root)):
        cve_id, title, text = extract_text(j)
        ids.append(cve_id)
        titles.append(title)
        docs.append(text)

    if not docs:
        raise SystemExit("No CVE files found")

    print(f"Loaded {len(docs)} CVEs")

    print("Generating embeddings...")
    emb = build_embeddings(docs)

    print("Building ArrowSpace...")
    start = time.perf_counter()
    aspace, gl = (
        ArrowSpaceBuilder()
        .with_seed(42)
        .with_dims_reduction(enabled=False, eps=None)
        .with_sampling("simple", 1.0)
    ).build(graph_params, emb)
    print(f"Build time: {time.perf_counter() - start:.2f}s")

    queries = [
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
        "directory traversal in backup restore endpoint",
        "cross-site request forgery CSRF in admin password change form",
        "out-of-bounds read in PDF rendering engine",
        "format string vulnerability in logging daemon",
        "prototype pollution in Node.js configuration merge utility",
        "arbitrary file upload leading to remote code execution",
        "open redirect in single sign-on logout flow",
        "improper certificate validation in mobile banking application",
        "sandbox escape in browser extension messaging interface",
        "null pointer dereference in Bluetooth service",
        "double free vulnerability in archive extraction library",
        "buffer underflow in font parsing component",
        "OS command injection in diagnostic CGI script",
        "LDAP injection in enterprise directory search feature",
        "NoSQL injection in user search API",
        "path canonicalization bypass in file download servlet",
        "improper neutralization of CRLF in mail gateway",
        "privilege escalation via writable system service configuration",
        "race condition in temporary file creation",
        "authorization bypass in GraphQL mutation resolver",
        "memory disclosure via uninitialized stack buffer",
        "heap use-after-free in media streaming server",
        "stack-based buffer overflow in DHCP client",
        "improper signature verification in software update channel",
        "symlink attack in installer cleanup routine",
        "cross-tenant data exposure in multi-tenant storage service",
        "request smuggling in reverse proxy parser",
        "host header injection in password reset workflow",
        "business logic bypass in coupon redemption API",
        "improper input validation in SAML assertion processing",
        "authentication brute force due to missing rate limiting",
        "insecure default permissions in container runtime socket",
    ]

    shuffle(queries)

    print(f"\nSearching {len(queries)} queries...")
    qemb = build_embeddings(queries, cache_file=QUERY_EMB_CACHE)

    tau_labels = [
        TAU_DISPLAY["Cosine"],
        TAU_DISPLAY["Hybrid"],
        TAU_DISPLAY["Taumode"],
    ]

    all_results = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        print(f"\n{'=' * 70}")
        print(f"Query {qi + 1}: {q}")
        print("=" * 70)

        results_cosine = aspace.search(qemb[qi], gl, tau=TAU_COSINE)
        results_hybrid = aspace.search(qemb[qi], gl, tau=TAU_HYBRID)
        results_taumode = aspace.search(qemb[qi], gl, tau=TAU_TAUMODE)

        min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        print(
            f"Results: cosine={len(results_cosine)}, "
            f"hybrid-{TAU_HYBRID}={len(results_hybrid)}, "
            f"taumode-{TAU_TAUMODE}={len(results_taumode)}, using min={min_len}"
        )

        results_cosine = results_cosine[:min_len]
        results_hybrid = results_hybrid[:min_len]
        results_taumode = results_taumode[:min_len]

        all_results.append((results_cosine, results_hybrid, results_taumode))

        spear_c_h, kendall_c_h = compute_ranking_metrics(results_cosine, results_hybrid)
        spear_c_t, kendall_c_t = compute_ranking_metrics(results_cosine, results_taumode)
        spear_h_t, kendall_h_t = compute_ranking_metrics(results_hybrid, results_taumode)

        k_ndcg = min(NDCG_K, min_len)
        ndcg_h_c = compute_ndcg(results_hybrid, results_cosine, k=k_ndcg)
        ndcg_t_c = compute_ndcg(results_taumode, results_cosine, k=k_ndcg)
        ndcg_t_h = compute_ndcg(results_taumode, results_hybrid, k=k_ndcg)

        tail_metrics = analyze_tail_distribution(
            [results_cosine, results_hybrid, results_taumode],
            tau_labels,
            k_head=HEAD_K,
            k_tail=RESULTS_K,
        )

        comparison_metrics.append({
            "query": q,
            "min_length": min_len,
            "spearman": (spear_c_h, spear_c_t, spear_h_t),
            "kendall": (kendall_c_h, kendall_c_t, kendall_h_t),
            "ndcg": (ndcg_h_c, ndcg_t_c, ndcg_t_h),
            "tail_metrics": tail_metrics,
        })

        gt_ids = [idx for idx, _ in results_cosine]
        gt_scores = [sc for _, sc in results_cosine]

        recall_metrics_per_tau = {}
        for tau_key, ret_results in [
            ("Cosine", results_cosine),
            ("Hybrid", results_hybrid),
            ("Taumode", results_taumode),
        ]:
            ret_ids = [idx for idx, _ in ret_results]
            ret_scores = [sc for _, sc in ret_results]
            recall_metrics_per_tau[tau_key] = compute_all_recall_metrics(
                retrieved_ids=ret_ids,
                retrieved_scores=ret_scores,
                ground_truth_ids=gt_ids,
                ground_truth_scores=gt_scores,
            )

        comparison_metrics[-1]["recall_metrics"] = recall_metrics_per_tau

        for label, results in zip(
            tau_labels, [results_cosine, results_hybrid, results_taumode]
        ):
            print(f"\n{label}")
            print("-" * 70)
            for rank, (idx, score) in enumerate(results[:RESULTS_K], 1):
                print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<40} [{score:.4f}]")

        print(f"\nCorrelations:")
        print(f" Cosine vs hybrid-{TAU_HYBRID}: ρ={spear_c_h:.3f}, τ={kendall_c_h:.3f}")
        print(f" Cosine vs taumode-{TAU_TAUMODE}: ρ={spear_c_t:.3f}, τ={kendall_c_t:.3f}")
        print(f" hybrid-{TAU_HYBRID} vs taumode-{TAU_TAUMODE}: ρ={spear_h_t:.3f}, τ={kendall_h_t:.3f}")

        print(f"\nNDCG@{k_ndcg}:")
        print(f" hybrid-{TAU_HYBRID} vs Cosine: {ndcg_h_c:.4f}")
        print(f" taumode-{TAU_TAUMODE} vs Cosine: {ndcg_t_c:.4f}")
        print(f" taumode-{TAU_TAUMODE} vs hybrid-{TAU_HYBRID}: {ndcg_t_h:.4f}")

        if tail_metrics:
            k_tail = tail_metrics[tau_labels[0]]["total_items"]
            print(f"\nTail Quality (Head=top-{HEAD_K}, Tail=ranks {HEAD_K + 1}-{k_tail}):")
            for label in tau_labels:
                if label in tail_metrics:
                    m = tail_metrics[label]
                    print(f" {label}:")
                    print(f"   T/H ratio: {m['tail_to_head_ratio']:.4f}")
                    print(f"   CV:        {m['tail_cv']:.4f}")
                    print(f"   Decay:     {m['tail_decay_rate']:.6f}")

        print(f"\nRecall Metrics (proxy implementation inspired by Kuffo et al., SIGIR '26):")
        print(f" {'Method':<20} {'Traditional':>14} {'Semantic':>12} {'Tolerant':>12} {'#SN':>6} {'Tol%':>8}")
        print(f" {'-' * 74}")
        for tau_key in ["Cosine", "Hybrid", "Taumode"]:
            rm = recall_metrics_per_tau[tau_key]
            sem = rm["semantic_recall"]
            is_nan = isinstance(sem, float) and math.isnan(sem)
            sem_str = " n/a" if is_nan else f"{sem:12.4f}"
            print(
                f" {tau_key:<20} {rm['traditional_recall']:>14.4f} {sem_str} "
                f"{rm['tolerant_recall']:>12.4f} {rm['n_semantic_neighbors']:>6} "
                f"{rm['tolerance_pct_used']:>7.2f}%"
            )

    plot_comparison(queries, all_results, ids, titles)

    if all(min(len(r[0]), len(r[1]), len(r[2])) > HEAD_K for r in all_results):
        plot_tail_comparison(queries, all_results, ids, titles)

    plot_semantic_recall_comparison(comparison_metrics)
    plot_metric_deltas(comparison_metrics)
    plot_win_loss_heatmap(comparison_metrics)
    plot_pareto_tradeoff(comparison_metrics)

    headk_sweep_rows = run_headk_sweep(queries, all_results, HEAD_K_SWEEP)
    save_headk_sweep_to_csv(headk_sweep_rows)
    plot_headk_sweep(headk_sweep_rows)

    print(f"\n{'=' * 70}")
    print("EXPORTING RESULTS TO CSV")
    print("=" * 70)

    save_search_results_to_csv(queries, all_results, ids, titles)
    save_metrics_to_csv(comparison_metrics)
    save_tail_metrics_to_csv(comparison_metrics)
    save_semantic_recall_to_csv(comparison_metrics)
    save_summary_to_csv(comparison_metrics)
    save_query_comparison(queries, all_results, titles, docs)
    save_run_metadata(
        OUTPUT_DIR / "cve_run_metadata.json",
        dataset_root=dataset_root,
        n_docs=len(docs),
        embedding_model_path=Path(__file__).parent.parent / "domain_adapted_model",
        query_count=len(queries),
        comparison_metrics=comparison_metrics,
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    avg_ndcg_h_c = np.mean([m["ndcg"][0] for m in comparison_metrics])
    avg_ndcg_t_c = np.mean([m["ndcg"][1] for m in comparison_metrics])

    print(f"\nAverage NDCG@{NDCG_K}:")
    print(f" hybrid-{TAU_HYBRID} vs Cosine: {avg_ndcg_h_c:.4f}")
    print(f" taumode-{TAU_TAUMODE} vs Cosine: {avg_ndcg_t_c:.4f}")

    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
    if valid_tail:
        print(f"\nAverage Tail/Head Ratios:")
        for label in tau_labels:
            ratios = []
            for m in valid_tail:
                if label in m["tail_metrics"]:
                    ratios.append(m["tail_metrics"][label]["tail_to_head_ratio"])
            if ratios:
                print(f" {label}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

    print(f"\n→ Higher T/H ratio indicates stronger long-tail score retention.")
    print(f"→ Lower tail CV and lower tail decay indicate more stable tail behaviour.")

    if headk_sweep_rows:
        print(f"\nHEAD_K sensitivity sweep completed for: {HEAD_K_SWEEP}")
        grouped_summary = {}
        for row in headk_sweep_rows:
            h = int(row["head_k"])
            tm = row["tau_method"]
            grouped_summary.setdefault(h, {}).setdefault(tm, []).append(float(row["tail_to_head_ratio"]))
        for h in sorted(grouped_summary):
            print(f" Head={h}:")
            for tm in [TAU_DISPLAY["Cosine"], TAU_DISPLAY["Hybrid"], TAU_DISPLAY["Taumode"]]:
                vals = grouped_summary[h].get(tm, [])
                if vals:
                    print(f"   {tm}: mean T/H ratio = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nAverage Recall Metrics:")
    print(f" {'Method':<20} {'Traditional':>14} {'Semantic':>12} {'Tolerant':>12}")
    print(f" {'-' * 62}")
    for tau_key in ["Cosine", "Hybrid", "Taumode"]:
        trad_v, sem_v, tol_v = [], [], []
        for m in comparison_metrics:
            rm = m.get("recall_metrics", {}).get(tau_key)
            if rm:
                trad_v.append(rm["traditional_recall"])
                s = rm["semantic_recall"]
                if not (isinstance(s, float) and math.isnan(s)):
                    sem_v.append(s)
                tol_v.append(rm["tolerant_recall"])

        trad_mean = np.mean(trad_v) if trad_v else float("nan")
        sem_mean = np.mean(sem_v) if sem_v else float("nan")
        tol_mean = np.mean(tol_v) if tol_v else float("nan")
        sem_str = " n/a" if (isinstance(sem_mean, float) and math.isnan(sem_mean)) else f"{sem_mean:12.4f}"
        print(f" {tau_key:<20} {trad_mean:>14.4f} {sem_str} {tol_mean:>12.4f}")

    print("\nSaving test query comparisons complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVE search with peer-review oriented tail and semantic recall analysis")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    args = parser.parse_args()
    main(args.dataset)
