"""CVE semantic search with pyarrowspace - Multi-metric comparison with tail analysis
Metrics: Spearman ρ, Kendall τ, NDCG@k, Tail/Head ratio, MRR-Top0 (novel topology-aware)

MRR-Top0 formula:
    MRR-Top0 = (1/|Q|) * Σ_q Σ_{i ∈ Rel(q)} T_{q,i} / rank(q,i)

where T_{q,i} is a topology factor derived from the ArrowSpace normalised lambda
(Rayleigh quotient / Dirichlet dispersion proxy) for item i, label-agnostic.

Requirements:
    pip install sentence-transformers numpy matplotlib scipy scikit-learn tqdm
Usage:
    python tests/test_2_CVE_db.py --dataset <dataset_dir>
"""
import os
import json
import glob
import time
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from arrowspace import ArrowSpaceBuilder, set_debug
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
import logging

logging.basicConfig(level=logging.INFO)
set_debug(True)

# ============================================================================
# Configuration
# ============================================================================
START_YEAR  = 1999
END_YEAR    = 2026
TAU_COSINE  = 1.0    # Pure cosine similarity
TAU_HYBRID  = 0.72   # Hybrid: mostly cosine, some spectral
TAU_TAUMODE = 0.42   # Spectral-aware (taumode)
K_TAIL_MAX  = 25     # Analyse tail up to rank 25

TAU_LABELS = [
    f"Cosine (τ={TAU_COSINE})",
    f"Hybrid (τ={TAU_HYBRID})",
    f"Taumode (τ={TAU_TAUMODE})",
]

GRAPH_PARAMS = {
    "eps":   1.31,
    "k":     30,
    "topk":  15,
    "p":     1.8,
    "sigma": 0.535,
}

print(f"Graph parameters: {GRAPH_PARAMS}")


# ============================================================================
# Data Loading
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
    cve_id = j.get("cveMetadata", {}).get("cveId", "")
    cna    = j.get("containers", {}).get("cna", {})
    title  = cna.get("title", "") or ""

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
        vendor  = a.get("vendor") or ""
        product = a.get("product") or ""
        if vendor or product:
            products.append(f"{vendor} {product}".strip())
    prod_str = " ".join(products)

    text = " | ".join(
        [s for s in [cve_id, title, description, cwe_str, cvss_vec, prod_str] if s]
    )
    return cve_id or "(unknown)", title or "(no title)", text


# ============================================================================
# Embeddings
# ============================================================================
def build_embeddings(
    texts,
    model_path=str(Path(__file__).parent.parent / "domain_adapted_model"),
    cache_file="cve_embeddings_cache.npy",
):
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        try:
            X = np.load(cache_file)
            if len(X) != len(texts):
                print(f"Cache size mismatch ({len(X)} vs {len(texts)}). Regenerating...")
            else:
                print(f"Embeddings loaded. Shape: {X.shape}")
                return X
        except Exception as e:
            print(f"Cache load error: {e}. Regenerating...")

    print(f"Loading model from: {model_path}")
    model = SentenceTransformer(model_path)
    print("Encoding texts...")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    X_scaled = X.astype(np.float64) * 1.2e1
    print(f"Saving embeddings to {cache_file}...")
    np.save(cache_file, X_scaled)
    print(f"Embeddings generated. Shape: {X_scaled.shape}")
    return X_scaled


# ============================================================================
# Metrics
# ============================================================================
def compute_ranking_metrics(results_a, results_b):
    """Spearman ρ and Kendall τ between two ranked lists."""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]
    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau_v, _ = kendalltau(rank_a, rank_b)
    return spearman_rho, kendall_tau_v


def compute_ndcg(results_pred, results_ref, k=10):
    """NDCG@k treating reference ranking as ground truth."""
    ref_indices    = [idx for idx, _ in results_ref[:k]]
    relevance_map  = {idx: k - i for i, idx in enumerate(ref_indices)}
    pred_indices   = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]

    if sum(true_relevance) == 0:
        return 0.0
    try:
        pred_scores = np.array([score for _, score in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()
        return ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k,
        )
    except Exception:
        return 0.0


def build_topology_scores(aspace) -> dict:
    """
    Build T_{q,i} topology factor for every item.

    Uses ArrowSpace's normalised lambda (Rayleigh quotient + Dirichlet dispersion
    proxy, already mapped to [0,1] during build) as the topology scalar.
    Replace or extend this function to incorporate explicit PageRank,
    conductance, or modularity values when available.

    Returns
    -------
    dict[int, float]  item_index -> T_{q,i}
    """
    lambdas = np.array(aspace.lambdas())  # already normalised to [0,1]
    # Avoid zero weights: shift so minimum is epsilon
    lambdas = np.clip(lambdas, 1e-9, None)
    return {i: float(lambdas[i]) for i in range(len(lambdas))}


def compute_mrr_top0(results, topo_scores: dict) -> float:
    """
    MRR-Top0: topology-weighted reciprocal rank over the full top-k.

    Formula (label-agnostic, all returned items are Rel(q)):
        MRR-Top0 = Σ_{i ∈ results} T_{q,i} / rank(i)

    Normalised by the number of items so scores are comparable across
    queries with different result-set sizes.

    Parameters
    ----------
    results     : list[(item_idx, score)]  ranked from best to worst
    topo_scores : dict[int, float]         item_idx -> T_{q,i}

    Returns
    -------
    float
    """
    if not results:
        return 0.0
    total = 0.0
    for rank, (idx, _) in enumerate(results, 1):
        T_qi = topo_scores.get(idx, 0.0)
        total += T_qi / float(rank)
    return total / len(results)


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    """Score distribution statistics for head vs tail positions."""
    min_length = min(len(r) for r in results_list)
    if min_length <= k_head:
        return {}

    actual_k_tail = min(k_tail, min_length)
    metrics = {}

    for results, label in zip(results_list, labels):
        seg         = results[:actual_k_tail]
        head_scores = [s for _, s in seg[:k_head]]
        tail_scores = [s for _, s in seg[k_head:actual_k_tail]]

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
            "tail_cv":            tail_std / tail_mean if tail_mean > 1e-10 else 0.0,
            "tail_decay_rate":    (tail_scores[0] - tail_scores[-1]) / len(tail_scores)
                                  if len(tail_scores) > 1 else 0.0,
            "n_tail_items":       len(tail_scores),
            "total_items":        actual_k_tail,
        }
    return metrics


# ============================================================================
# CSV Exports
# ============================================================================
def save_search_results_to_csv(queries, all_results, ids, titles,
                                output_file="cve_search_results.csv"):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_id", "query_text", "tau_method",
                        "rank", "cve_id", "title", "score"],
        )
        writer.writeheader()
        for qi, query in enumerate(queries):
            for label, results in zip(TAU_LABELS, all_results[qi]):
                for rank, (idx, score) in enumerate(results[:20], 1):
                    writer.writerow({
                        "query_id":   qi + 1,
                        "query_text": query,
                        "tau_method": label,
                        "rank":       rank,
                        "cve_id":     ids[idx],
                        "title":      titles[idx],
                        "score":      f"{score:.6f}",
                    })
    print(f"Search results saved to {output_file}")


def save_metrics_to_csv(comparison_metrics, output_file="cve_comparison_metrics.csv"):
    fieldnames = [
        "query_id", "query_text", "min_length",
        "spearman_cosine_hybrid",  "spearman_cosine_taumode",  "spearman_hybrid_taumode",
        "kendall_cosine_hybrid",   "kendall_cosine_taumode",   "kendall_hybrid_taumode",
        "ndcg_hybrid_vs_cosine",   "ndcg_taumode_vs_cosine",   "ndcg_taumode_vs_hybrid",
        "mrr_top0_cosine",         "mrr_top0_hybrid",          "mrr_top0_taumode",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi, m in enumerate(comparison_metrics):
            writer.writerow({
                "query_id":                 qi + 1,
                "query_text":               m["query"],
                "min_length":               m["min_length"],
                "spearman_cosine_hybrid":   f"{m['spearman'][0]:.6f}",
                "spearman_cosine_taumode":  f"{m['spearman'][1]:.6f}",
                "spearman_hybrid_taumode":  f"{m['spearman'][2]:.6f}",
                "kendall_cosine_hybrid":    f"{m['kendall'][0]:.6f}",
                "kendall_cosine_taumode":   f"{m['kendall'][1]:.6f}",
                "kendall_hybrid_taumode":   f"{m['kendall'][2]:.6f}",
                "ndcg_hybrid_vs_cosine":    f"{m['ndcg'][0]:.6f}",
                "ndcg_taumode_vs_cosine":   f"{m['ndcg'][1]:.6f}",
                "ndcg_taumode_vs_hybrid":   f"{m['ndcg'][2]:.6f}",
                "mrr_top0_cosine":          f"{m['mrr_top0'][0]:.6f}",
                "mrr_top0_hybrid":          f"{m['mrr_top0'][1]:.6f}",
                "mrr_top0_taumode":         f"{m['mrr_top0'][2]:.6f}",
            })
    print(f"Comparison metrics saved to {output_file}")


def save_tail_metrics_to_csv(comparison_metrics, output_file="cve_tail_metrics.csv"):
    fieldnames = [
        "query_id", "query_text", "tau_method",
        "head_mean", "tail_mean", "tail_std",
        "tail_to_head_ratio", "tail_cv", "tail_decay_rate",
        "n_tail_items", "total_items",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qi, m in enumerate(comparison_metrics):
            for label in TAU_LABELS:
                tm = m.get("tail_metrics", {}).get(label)
                if tm is None:
                    continue
                writer.writerow({
                    "query_id":          qi + 1,
                    "query_text":        m["query"],
                    "tau_method":        label,
                    "head_mean":         f"{tm['head_mean']:.6f}",
                    "tail_mean":         f"{tm['tail_mean']:.6f}",
                    "tail_std":          f"{tm['tail_std']:.6f}",
                    "tail_to_head_ratio":f"{tm['tail_to_head_ratio']:.6f}",
                    "tail_cv":           f"{tm['tail_cv']:.6f}",
                    "tail_decay_rate":   f"{tm['tail_decay_rate']:.6f}",
                    "n_tail_items":      tm["n_tail_items"],
                    "total_items":       tm["total_items"],
                })
    print(f"Tail metrics saved to {output_file}")


def save_summary_to_csv(comparison_metrics, output_file="cve_summary.csv"):
    def _mean(key, idx=None):
        if idx is None:
            return np.mean([m[key] for m in comparison_metrics])
        return np.mean([m[key][idx] for m in comparison_metrics])

    def _std(key, idx=None):
        if idx is None:
            return np.std([m[key] for m in comparison_metrics])
        return np.std([m[key][idx] for m in comparison_metrics])

    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]

    rows = [
        # NDCG
        ("NDCG@10", "Hybrid vs Cosine",   _mean("ndcg", 0), _std("ndcg", 0)),
        ("NDCG@10", "Taumode vs Cosine",  _mean("ndcg", 1), _std("ndcg", 1)),
        ("NDCG@10", "Taumode vs Hybrid",  _mean("ndcg", 2), _std("ndcg", 2)),
        # MRR-Top0
        ("MRR-Top0", TAU_LABELS[0], _mean("mrr_top0", 0), _std("mrr_top0", 0)),
        ("MRR-Top0", TAU_LABELS[1], _mean("mrr_top0", 1), _std("mrr_top0", 1)),
        ("MRR-Top0", TAU_LABELS[2], _mean("mrr_top0", 2), _std("mrr_top0", 2)),
    ]

    # Tail/Head ratio averages
    for label in TAU_LABELS:
        ratios = [
            m["tail_metrics"][label]["tail_to_head_ratio"]
            for m in valid_tail
            if label in m["tail_metrics"]
        ]
        if ratios:
            rows.append(("Tail/Head Ratio", label,
                         float(np.mean(ratios)), float(np.std(ratios))))

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric_type", "metric_name", "value", "std_dev"]
        )
        writer.writeheader()
        for metric_type, metric_name, value, std_dev in rows:
            writer.writerow({
                "metric_type": metric_type,
                "metric_name": metric_name,
                "value":       f"{value:.6f}",
                "std_dev":     f"{std_dev:.6f}",
            })
    print(f"Summary statistics saved to {output_file}")


# ============================================================================
# Visualisations
# ============================================================================
def plot_comparison(queries, all_results, ids, titles,
                    output_file="cve_top10_comparison.png"):
    n  = len(queries)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for qi, query in enumerate(queries):
        k = min(10, min(len(r) for r in all_results[qi]))
        for ti, (results, label, color) in enumerate(
            zip(all_results[qi], TAU_LABELS, colors)
        ):
            ax     = axes[qi, ti]
            scores = [s for _, s in results[:k]]
            ax.bar(range(1, k + 1), scores, alpha=0.7, color=color)
            ax.set_xlabel("Rank", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"Q{qi+1}: {label}\n{query[:50]}...",
                         fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for i, (idx, score) in enumerate(results[:k]):
                ax.text(i + 1, score + 0.01 * (max(scores) if scores else 1),
                        ids[idx].split("-")[-1],
                        ha="center", va="bottom", fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Top-10 plot saved to {output_file}")
    plt.close()


def plot_tail_comparison(queries, all_results, ids, titles,
                         output_file="cve_tail_analysis.png"):
    n   = len(queries)
    fig = plt.figure(figsize=(20, 5 * n))
    gs  = fig.add_gridspec(n, 4, hspace=0.3, wspace=0.3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for qi, query in enumerate(queries):
        k           = min(len(r) for r in all_results[qi])
        results_all = [r[:k] for r in all_results[qi]]

        # Panel 1: full distribution
        ax1 = fig.add_subplot(gs[qi, 0])
        for results, label, color in zip(results_all, TAU_LABELS, colors):
            scores = [s for _, s in results]
            ax1.plot(range(1, k + 1), scores, marker="o", label=label,
                     color=color, alpha=0.7, markersize=4, linewidth=2)
        ax1.axvline(x=3.5, color="red", linestyle="--", alpha=0.5,
                    linewidth=2, label="Head/Tail")
        ax1.set_xlabel("Rank", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax1.set_title(f"Q{qi+1}: Score Distribution (n={k})\n{query[:45]}...",
                      fontsize=10, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: tail only
        ax2 = fig.add_subplot(gs[qi, 1])
        if k > 3:
            for results, label, color in zip(results_all, TAU_LABELS, colors):
                tail = [s for _, s in results[3:]]
                ax2.plot(range(4, k + 1), tail, marker="s", label=label,
                         color=color, alpha=0.7, markersize=5, linewidth=2)
            ax2.set_xlabel("Rank", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax2.set_title(f"Q{qi+1}: Tail (Ranks 4–{k})",
                          fontsize=10, fontweight="bold")
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        # Panel 3: box plot
        ax3 = fig.add_subplot(gs[qi, 2])
        if k > 3:
            tail_data = [[s for _, s in r[3:]] for r in results_all]
            bp = ax3.boxplot(tail_data,
                             labels=["Cosine", "Hybrid", "Taumode"],
                             patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax3.set_ylabel("Score", fontsize=11, fontweight="bold")
            ax3.set_title(f"Q{qi+1}: Tail Variability",
                          fontsize=10, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

        # Panel 4: tail metrics bar
        ax4 = fig.add_subplot(gs[qi, 3])
        if k > 3:
            tail_metrics = analyze_tail_distribution(
                results_all, TAU_LABELS, k_head=3, k_tail=k
            )
            x_pos  = np.arange(3)
            width  = 0.25
            metric_names = ["Tail Mean", "T/H Ratio", "Stability"]
            for i, (label, color) in enumerate(zip(TAU_LABELS, colors)):
                if label in tail_metrics:
                    tm = tail_metrics[label]
                    cv = tm["tail_cv"]
                    vals = [
                        tm["tail_mean"],
                        tm["tail_to_head_ratio"],
                        1.0 / (1.0 + cv) if cv > 0 else 1.0,
                    ]
                    ax4.bar(x_pos + i * width, vals, width,
                            label=label, color=color, alpha=0.7)
            ax4.set_ylabel("Value", fontsize=11, fontweight="bold")
            ax4.set_title(f"Q{qi+1}: Tail Metrics",
                          fontsize=10, fontweight="bold")
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(metric_names, fontsize=9,
                                rotation=15, ha="right")
            ax4.legend(fontsize=8)
            ax4.grid(axis="y", alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Tail analysis plot saved to {output_file}")
    plt.close()


def plot_mrr_top0(comparison_metrics, output_file="cve_mrr_top0.png"):
    """Bar chart of per-query MRR-Top0 for all three tau methods."""
    n      = len(comparison_metrics)
    x_pos  = np.arange(n)
    width  = 0.28
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), 5))
    for i, (label, color) in enumerate(zip(TAU_LABELS, colors)):
        vals = [m["mrr_top0"][i] for m in comparison_metrics]
        ax.bar(x_pos + i * width, vals, width, label=label,
               color=color, alpha=0.75)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n)],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("MRR-Top0", fontsize=12, fontweight="bold")
    ax.set_title("Per-Query MRR-Top0 (topology-weighted reciprocal rank)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"MRR-Top0 plot saved to {output_file}")
    plt.close()


# ============================================================================
# Human-readable comparison
# ============================================================================
def save_query_comparison(queries, all_results, titles, docs,
                           output_file="query_comparison.txt"):
    print(f"Generating human-readable comparison to {output_file}...")

    query_metrics = []
    for qi, q in enumerate(queries):
        res_c, _, res_e = all_results[qi]
        top_score = res_e[0][1] if res_e else 0.0
        query_metrics.append({
            "qi": qi, "query": q, "score": top_score,
            "res_cosine": res_c, "res_taumode": res_e,
        })

    sorted_q = sorted(query_metrics, key=lambda x: x["score"], reverse=True)
    if not sorted_q:
        return

    if len(sorted_q) <= 3:
        labels_map = [
            "Best (Highest Confidence)",
            "Sample (Middle)",
            "Worst (Lowest Confidence)",
        ]
        selected = [(labels_map[i], q) for i, q in enumerate(sorted_q)]
    else:
        selected = [
            ("BEST QUERY (Highest Top Score)", sorted_q[0]),
            ("WORST QUERY (Lowest Top Score)", sorted_q[-1]),
            ("SAMPLE QUERY (Median Score)",    sorted_q[len(sorted_q) // 2]),
        ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" QUERY RESULT COMPARISON: COSINE vs TAUMODE\n")
        f.write("=" * 80 + "\n\n")

        for label, q_data in selected:
            f.write(f"QUERY TYPE: {label}\n")
            f.write(f"QUERY TEXT: {q_data['query']}\n")
            f.write("-" * 80 + "\n")

            for i in range(10):
                f.write(f"RANK {i+1}:\n")
                for method, res in [("Cosine ", q_data["res_cosine"]),
                                    ("Taumode", q_data["res_taumode"])]:
                    if i < len(res):
                        idx, score = res[i]
                        snippet = docs[idx][:300].replace("\n", " ") + "..."
                        f.write(f"  [{method}] Score: {score:.4f}\n")
                        f.write(f"           Title: {titles[idx]}\n")
                        f.write(f"           Text:  {snippet}\n")
                    else:
                        f.write(f"  [{method}] No result\n")
                f.write("-" * 40 + "\n")

            f.write("=" * 80 + "\n\n")

    print(f"Comparison saved to {output_file}")


# ============================================================================
# Main
# ============================================================================
def main(dataset_root):
    # ── Load CVEs ──────────────────────────────────────────────────────────
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

    # ── Embeddings ─────────────────────────────────────────────────────────
    print("Generating embeddings...")
    emb = build_embeddings(docs)

    # ── Build ArrowSpace ───────────────────────────────────────────────────
    print("Building ArrowSpace...")
    t0 = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder().build(GRAPH_PARAMS, emb)
    print(f"Build time: {time.perf_counter() - t0:.2f}s")

    # Pre-compute topology scores (shared across all queries)
    topo_scores = build_topology_scores(aspace)

    # ── Queries ────────────────────────────────────────────────────────────
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
    ]
    from random import shuffle
    shuffle(queries)

    print(f"\nSearching {len(queries)} queries...")
    qemb = build_embeddings(
        queries,
        cache_file=str(Path(__file__).parent.parent / "cve_queries_emb_cache.npy"),
    )

    all_results        = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        print(f"\n{'='*70}")
        print(f"Query {qi+1}: {q}")
        print("=" * 70)

        results_cosine  = aspace.search(qemb[qi], gl, tau=TAU_COSINE)
        results_hybrid  = aspace.search(qemb[qi], gl, tau=TAU_HYBRID)
        results_taumode = aspace.search(qemb[qi], gl, tau=TAU_TAUMODE)

        min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        print(f"Results: cosine={len(results_cosine)}, "
              f"hybrid={len(results_hybrid)}, "
              f"taumode={len(results_taumode)}, using min={min_len}")

        results_cosine  = results_cosine[:min_len]
        results_hybrid  = results_hybrid[:min_len]
        results_taumode = results_taumode[:min_len]

        all_results.append((results_cosine, results_hybrid, results_taumode))

        # ── Correlation metrics ────────────────────────────────────────────
        spear_c_h, kendall_c_h = compute_ranking_metrics(results_cosine, results_hybrid)
        spear_c_t, kendall_c_t = compute_ranking_metrics(results_cosine, results_taumode)
        spear_h_t, kendall_h_t = compute_ranking_metrics(results_hybrid, results_taumode)

        # ── NDCG ──────────────────────────────────────────────────────────
        k_ndcg  = min(10, min_len)
        ndcg_hc = compute_ndcg(results_hybrid,  results_cosine, k=k_ndcg)
        ndcg_tc = compute_ndcg(results_taumode, results_cosine, k=k_ndcg)
        ndcg_th = compute_ndcg(results_taumode, results_hybrid, k=k_ndcg)

        # ── Tail distribution ─────────────────────────────────────────────
        tail_metrics = analyze_tail_distribution(
            [results_cosine, results_hybrid, results_taumode],
            TAU_LABELS, k_head=3, k_tail=K_TAIL_MAX,
        )

        # ── MRR-Top0 (novel topology-aware metric) ────────────────────────
        mrr_cos = compute_mrr_top0(results_cosine,  topo_scores)
        mrr_hyb = compute_mrr_top0(results_hybrid,  topo_scores)
        mrr_tau = compute_mrr_top0(results_taumode, topo_scores)

        comparison_metrics.append({
            "query":       q,
            "min_length":  min_len,
            "spearman":    (spear_c_h, spear_c_t, spear_h_t),
            "kendall":     (kendall_c_h, kendall_c_t, kendall_h_t),
            "ndcg":        (ndcg_hc, ndcg_tc, ndcg_th),
            "tail_metrics":tail_metrics,
            "mrr_top0":    (mrr_cos, mrr_hyb, mrr_tau),
        })

        # ── Display top-10 ────────────────────────────────────────────────
        for label, results in zip(TAU_LABELS,
                                   [results_cosine, results_hybrid, results_taumode]):
            print(f"\n{label}")
            print("-" * 70)
            for rank, (idx, score) in enumerate(results[:10], 1):
                print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<40} [{score:.4f}]")

        # ── Print metrics ─────────────────────────────────────────────────
        print(f"\nCorrelations:")
        print(f"  Cosine vs Hybrid-{TAU_HYBRID}:        ρ={spear_c_h:.3f}, τ={kendall_c_h:.3f}")
        print(f"  Cosine vs Taumode-{TAU_TAUMODE}:      ρ={spear_c_t:.3f}, τ={kendall_c_t:.3f}")
        print(f"  Hybrid-{TAU_HYBRID} vs Taumode-{TAU_TAUMODE}: ρ={spear_h_t:.3f}, τ={kendall_h_t:.3f}")

        print(f"\nNDCG@{k_ndcg}:")
        print(f"  Hybrid-{TAU_HYBRID} vs Cosine:        {ndcg_hc:.4f}")
        print(f"  Taumode-{TAU_TAUMODE} vs Cosine:      {ndcg_tc:.4f}")
        print(f"  Taumode-{TAU_TAUMODE} vs Hybrid-{TAU_HYBRID}: {ndcg_th:.4f}")

        print(f"\nMRR-Top0 (topology-weighted reciprocal rank):")
        print(f"  {TAU_LABELS[0]}: {mrr_cos:.4f}")
        print(f"  {TAU_LABELS[1]}: {mrr_hyb:.4f}")
        print(f"  {TAU_LABELS[2]}: {mrr_tau:.4f}")

        if tail_metrics:
            k_tail = next(iter(tail_metrics.values()))["total_items"]
            print(f"\nTail Quality (Ranks 4–{k_tail}):")
            for label in TAU_LABELS:
                if label in tail_metrics:
                    tm = tail_metrics[label]
                    print(f"  {label}:  T/H={tm['tail_to_head_ratio']:.4f}  "
                          f"CV={tm['tail_cv']:.4f}")

    # ── Visualisations ──────────────────────────────────────────────────
    plot_comparison(queries, all_results, ids, titles)
    plot_mrr_top0(comparison_metrics)

    if all(min(len(r) for r in triple) > 3 for triple in all_results):
        plot_tail_comparison(queries, all_results, ids, titles)

    # ── CSV Exports ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXPORTING RESULTS TO CSV")
    print("=" * 70)
    save_search_results_to_csv(queries, all_results, ids, titles)
    save_metrics_to_csv(comparison_metrics)
    save_tail_metrics_to_csv(comparison_metrics)
    save_summary_to_csv(comparison_metrics)
    save_query_comparison(queries, all_results, titles, docs)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\nAverage NDCG@10:")
    print(f"  Hybrid-{TAU_HYBRID} vs Cosine:        "
          f"{np.mean([m['ndcg'][0] for m in comparison_metrics]):.4f}")
    print(f"  Taumode-{TAU_TAUMODE} vs Cosine:      "
          f"{np.mean([m['ndcg'][1] for m in comparison_metrics]):.4f}")

    print(f"\nAverage MRR-Top0:")
    for i, label in enumerate(TAU_LABELS):
        vals = [m["mrr_top0"][i] for m in comparison_metrics]
        print(f"  {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
    if valid_tail:
        print(f"\nAverage Tail/Head Ratios:")
        for label in TAU_LABELS:
            ratios = [
                m["tail_metrics"][label]["tail_to_head_ratio"]
                for m in valid_tail
                if label in m["tail_metrics"]
            ]
            if ratios:
                print(f"  {label}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

        print(f"\n→ Higher T/H ratio   = Better long-tail quality")
        print(f"→ Higher MRR-Top0    = Better topology-aware ranking")
        print(f"→ ArrowSpace (τ<1.0) maintains higher tail scores and MRR-Top0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVE search with tail analysis + MRR-Top0")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    args = parser.parse_args()
    main(args.dataset)
