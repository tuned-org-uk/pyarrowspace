"""CVE semantic search with pyarrowspace + pplx-embed (1B) embeddings from file

Metrics: Spearman ρ, Kendall τ, NDCG@k, Tail/Head ratio, MRR-Top0 (topology-aware).

Embeddings are produced externally by embeddings_ppx_1B.py:
    from embeddings_ppx_1B import build_embeddings

This script only:
    - loads CVE JSON,
    - asks embeddings_ppx_1B for document + query embeddings,
    - builds ArrowSpace graph,
    - evaluates search quality (incl. MRR-Top0),
    - exports CSV and plots.

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

import matplotlib.pyplot as plt
import numpy as np
from arrowspace import ArrowSpaceBuilder, set_debug
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
from arrowspace import ArrowSpaceBuilder, set_debug
import logging

logging.basicConfig(level=logging.INFO)
set_debug(True)

# ============================================================================
# Configuration
# ============================================================================
START_YEAR  = 1999
END_YEAR    = 2025
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
    "k":     120,
    "topk":  50,
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
# Metrics
# ============================================================================
def compute_ranking_metrics(results_a, results_b):
    """Spearman ρ and Kendall τ between two ranked lists."""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    shared    = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]
    rho, _ = spearmanr(rank_a, rank_b)
    tau, _ = kendalltau(rank_a, rank_b)
    return rho, tau


def compute_ndcg(results_pred, results_ref, k=10):
    """NDCG@k treating reference ranking as ground truth."""
    ref_indices    = [idx for idx, _ in results_ref[:k]]
    relevance_map  = {idx: k - i for i, idx in enumerate(ref_indices)}
    pred_indices   = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]
    if sum(true_relevance) == 0:
        return 0.0
    try:
        pred_scores = np.array([s for _, s in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()
        return ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k,
        )
    except Exception:
        return 0.0


def build_embeddings(
    texts,
    model_path=str(Path(__file__).parent.parent / "domain_adapted_model_ppx" / "pplx_model_snapshot"),
    cache_file="cve_embeddings_pplx1b.npy",
    encode_precision="float32",   # match what the Colab script saved: float64-cast float32
    batch_size=4,                  # small batches to survive MPS memory limits
    max_seq_length=1024,            # truncate long CVE texts to avoid OOM
):
    """
    Load embeddings from cache if available, else generate with pplx-embed snapshot.
    
    IMPORTANT: encode_precision must match whatever the training script used.
    - embeddings_ppx_0_6B.py saves: model.encode(..., precision="int8").astype(float64) * 1.2e1
    - If cache exists, it is already scaled. Do NOT rescale on load.
    - If regenerating, we encode float32 here and apply the same 1.2e1 scale.
    """
    # ── 1. Try cache first ───────────────────────────────────────────────
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        try:
            X = np.load(cache_file, allow_pickle=False)
            if len(X) != len(texts):
                print(f"  Cache size mismatch ({len(X)} vs {len(texts)}). Regenerating...")
            else:
                print(f"  Loaded. Shape: {X.shape}  dtype: {X.dtype}")
                return X
        except Exception as e:
            print(f"  Cache load failed: {e}. Regenerating...")

    # ── 2. Generate from snapshot ────────────────────────────────────────
    print(f"Cache not found. Loading model from: {model_path}")
    model = SentenceTransformer(
        model_path,
        trust_remote_code=True,
        tokenizer_kwargs={"fix_mistral_regex": True} 
    )

    # Truncate inputs to avoid OOM on long CVE descriptions
    model.max_seq_length = max_seq_length

    # Set MPS memory watermark before any forward pass
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

    print(f"Encoding {len(texts):,} texts  "
          f"batch_size={batch_size}  max_seq={max_seq_length}  precision={encode_precision}...")
    
    X = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        precision=encode_precision,        # "float32" — no int8 batch-calibration instability
        normalize_embeddings=False,        # pplx-embed: unnormalised for cosine
    )

    # Apply the same scaling used by embeddings_ppx_0_6B.py
    X_scaled = X.astype(np.float64) * 1.2e1

    print(f"Saving to {cache_file}...")
    np.save(cache_file, X_scaled)
    print(f"Done. Shape: {X_scaled.shape}  dtype: {X_scaled.dtype}")
    return X_scaled

def build_topology_scores(aspace) -> dict:
    """
    Topology factor T_{q,i} for MRR-Top0.

    Uses ArrowSpace normalised lambda ∈ [0,1] (Rayleigh quotient + Dirichlet
    proxy, computed during build). Clip zeros to avoid degenerate weights.
    """
    lambdas = np.array(aspace.lambdas(), dtype=np.float64)
    lambdas = np.clip(lambdas, 1e-9, None)
    return {i: float(lambdas[i]) for i in range(len(lambdas))}


def compute_mrr_top0(results: list, topo_scores: dict) -> float:
    """
    MRR-Top0: label-agnostic, topology-weighted reciprocal rank over full top-k.

      MRR-Top0 = (1/|results|) * Σ_{i ∈ results} T_{q,i} / rank(i)
    """
    if not results:
        return 0.0
    total = sum(
        topo_scores.get(idx, 0.0) / float(rank)
        for rank, (idx, _) in enumerate(results, 1)
    )
    return total / len(results)


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    """Score distribution statistics for head vs tail positions."""
    min_length = min(len(r) for r in results_list)
    if min_length <= k_head:
        return {}

    actual_k = min(k_tail, min_length)
    metrics  = {}

    for results, label in zip(results_list, labels):
        seg         = results[:actual_k]
        head_scores = [s for _, s in seg[:k_head]]
        tail_scores = [s for _, s in seg[k_head:actual_k]]
        if not tail_scores or not head_scores:
            continue

        tail_mean = float(np.mean(tail_scores))
        tail_std  = float(np.std(tail_scores))
        head_mean = float(np.mean(head_scores))

        metrics[label] = {
            "head_mean":          head_mean,
            "tail_mean":          tail_mean,
            "tail_std":           tail_std,
            "tail_to_head_ratio": tail_mean / head_mean if head_mean > 1e-10 else 0.0,
            "tail_cv":            tail_std / tail_mean if tail_mean > 1e-10 else 0.0,
            "tail_decay_rate":    (tail_scores[0] - tail_scores[-1]) / len(tail_scores)
                                  if len(tail_scores) > 1 else 0.0,
            "n_tail_items":       len(tail_scores),
            "total_items":        actual_k,
        }
    return metrics

# ============================================================================
# CSV Exports
# ============================================================================
def save_search_results_to_csv(queries, all_results, ids, titles,
                               output_file="cve_search_results.csv"):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["query_id", "query_text", "tau_method",
                           "rank", "cve_id", "title", "score"]
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
                "query_id":                qi + 1,
                "query_text":              m["query"],
                "min_length":              m["min_length"],
                "spearman_cosine_hybrid":  f"{m['spearman'][0]:.6f}",
                "spearman_cosine_taumode": f"{m['spearman'][1]:.6f}",
                "spearman_hybrid_taumode": f"{m['spearman'][2]:.6f}",
                "kendall_cosine_hybrid":   f"{m['kendall'][0]:.6f}",
                "kendall_cosine_taumode":  f"{m['kendall'][1]:.6f}",
                "kendall_hybrid_taumode":  f"{m['kendall'][2]:.6f}",
                "ndcg_hybrid_vs_cosine":   f"{m['ndcg'][0]:.6f}",
                "ndcg_taumode_vs_cosine":  f"{m['ndcg'][1]:.6f}",
                "ndcg_taumode_vs_hybrid":  f"{m['ndcg'][2]:.6f}",
                "mrr_top0_cosine":         f"{m['mrr_top0'][0]:.6f}",
                "mrr_top0_hybrid":         f"{m['mrr_top0'][1]:.6f}",
                "mrr_top0_taumode":        f"{m['mrr_top0'][2]:.6f}",
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
                    "query_id":           qi + 1,
                    "query_text":         m["query"],
                    "tau_method":         label,
                    "head_mean":          f"{tm['head_mean']:.6f}",
                    "tail_mean":          f"{tm['tail_mean']:.6f}",
                    "tail_std":           f"{tm['tail_std']:.6f}",
                    "tail_to_head_ratio": f"{tm['tail_to_head_ratio']:.6f}",
                    "tail_cv":            f"{tm['tail_cv']:.6f}",
                    "tail_decay_rate":    f"{tm['tail_decay_rate']:.6f}",
                    "n_tail_items":       tm["n_tail_items"],
                    "total_items":        tm["total_items"],
                })
    print(f"Tail metrics saved to {output_file}")


def save_summary_to_csv(comparison_metrics, output_file="cve_summary.csv"):
    def _agg(key, idx):
        vals = [m[key][idx] for m in comparison_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    valid_tail = [m for m in comparison_metrics if m["tail_metrics"]]
    rows = []

    for i, label in enumerate(["Hybrid vs Cosine", "Taumode vs Cosine", "Taumode vs Hybrid"]):
        mu, sd = _agg("ndcg", i)
        rows.append(("NDCG@10", label, mu, sd))

    for i, label in enumerate(TAU_LABELS):
        mu, sd = _agg("mrr_top0", i)
        rows.append(("MRR-Top0", label, mu, sd))

    for label in TAU_LABELS:
        ratios = [
            m["tail_metrics"][label]["tail_to_head_ratio"]
            for m in valid_tail if label in m["tail_metrics"]
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
    n       = len(queries)
    fig, ax = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        ax = ax.reshape(1, -1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for qi, query in enumerate(queries):
        k = min(10, min(len(r) for r in all_results[qi]))
        for ti, (results, label, color) in enumerate(
            zip(all_results[qi], TAU_LABELS, colors)
        ):
            a      = ax[qi, ti]
            scores = [s for _, s in results[:k]]
            a.bar(range(1, k + 1), scores, alpha=0.7, color=color)
            a.set_xlabel("Rank", fontsize=10)
            a.set_ylabel("Score", fontsize=10)
            a.set_title(f"Q{qi+1}: {label}\n{query[:50]}...",
                        fontsize=9, fontweight="bold")
            a.grid(axis="y", alpha=0.3)
            for i, (idx, score) in enumerate(results[:k]):
                a.text(i + 1, score + 0.01 * (max(scores) if scores else 1),
                       ids[idx].split("-")[-1],
                       ha="center", va="bottom", fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Top-10 plot saved to {output_file}")
    plt.close()


def plot_tail_comparison(queries, all_results, ids, titles,
                         output_file="cve_tail_analysis.png"):
    n      = len(queries)
    fig    = plt.figure(figsize=(20, 5 * n))
    gs     = fig.add_gridspec(n, 4, hspace=0.3, wspace=0.3)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for qi, query in enumerate(queries):
        k           = min(len(r) for r in all_results[qi])
        results_all = [r[:k] for r in all_results[qi]]

        ax1 = fig.add_subplot(gs[qi, 0])
        for results, label, color in zip(results_all, TAU_LABELS, colors):
            scores = [s for _, s in results]
            ax1.plot(range(1, k + 1), scores, marker="o", label=label,
                     color=color, alpha=0.7, markersize=4, linewidth=2)
        ax1.axvline(x=3.5, color="red", linestyle="--", alpha=0.5, linewidth=2)
        ax1.set_title(f"Q{qi+1}: Distribution (n={k})\n{query[:40]}...",
                      fontsize=9, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[qi, 1])
        if k > 3:
            for results, label, color in zip(results_all, TAU_LABELS, colors):
                tail = [s for _, s in results[3:]]
                ax2.plot(range(4, k + 1), tail, marker="s", label=label,
                         color=color, alpha=0.7, markersize=5, linewidth=2)
            ax2.set_title(f"Q{qi+1}: Tail (Ranks 4–{k})",
                          fontsize=9, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[qi, 2])
        if k > 3:
            bp = ax3.boxplot(
                [[s for _, s in r[3:]] for r in results_all],
                labels=["Cos", "Hyb", "Tau"],
                patch_artist=True, widths=0.6,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax3.set_title(f"Q{qi+1}: Tail Variability",
                          fontsize=9, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

        ax4 = fig.add_subplot(gs[qi, 3])
        if k > 3:
            tm = analyze_tail_distribution(results_all, TAU_LABELS, k_head=3, k_tail=k)
            x  = np.arange(3)
            w  = 0.25
            for i, (label, color) in enumerate(zip(TAU_LABELS, colors)):
                if label in tm:
                    m  = tm[label]
                    cv = m["tail_cv"]
                    ax4.bar(x + i * w,
                            [m["tail_mean"], m["tail_to_head_ratio"],
                             1.0 / (1.0 + cv) if cv > 0 else 1.0],
                            w, label=label, color=color, alpha=0.7)
            ax4.set_xticks(x + w)
            ax4.set_xticklabels(["Mean", "T/H", "Stab"], fontsize=9)
            ax4.set_title(f"Q{qi+1}: Tail Metrics",
                          fontsize=9, fontweight="bold")
            ax4.legend(fontsize=7)
            ax4.grid(axis="y", alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Tail analysis plot saved to {output_file}")
    plt.close()


def plot_mrr_top0(comparison_metrics, output_file="cve_mrr_top0.png"):
    """Per-query MRR-Top0 grouped bar chart for all three tau methods."""
    n      = len(comparison_metrics)
    x      = np.arange(n)
    w      = 0.28
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), 5))
    for i, (label, color) in enumerate(zip(TAU_LABELS, colors)):
        vals = [m["mrr_top0"][i] for m in comparison_metrics]
        ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.75)

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n)],
                       fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("MRR-Top0", fontsize=12, fontweight="bold")
    ax.set_title("Per-Query MRR-Top0  (pplx-embed 1B via embeddings_ppx_1B.py)",
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
    print(f"Generating comparison to {output_file}...")

    ranked = sorted(
        [{"qi": qi, "query": q,
          "score": all_results[qi][2][0][1] if all_results[qi][2] else 0.0,
          "res_cosine": all_results[qi][0],
          "res_taumode": all_results[qi][2]}
         for qi, q in enumerate(queries)],
        key=lambda x: x["score"], reverse=True,
    )

    selected = (
        [("BEST (Highest Confidence)",  ranked[0]),
         ("WORST (Lowest Confidence)",  ranked[-1]),
         ("SAMPLE (Median Score)",      ranked[len(ranked) // 2])]
        if len(ranked) > 3
        else list(zip(["Best", "Sample", "Worst"][:len(ranked)], ranked))
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(" QUERY COMPARISON: COSINE vs TAUMODE  [pplx-embed 1B embeddings]\n")
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
                        snippet    = docs[idx][:300].replace("\n", " ") + "..."
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

    # ── Embeddings (via embeddings_ppx_1B.py) ─────────────────────────────
    print("\nBuilding document embeddings (pplx-embed 1B via embeddings_ppx_1B.py)...")
    emb = build_embeddings(docs, cache_file="cve_embeddings_pplx1b.npy")

    # ── Build ArrowSpace ───────────────────────────────────────────────────
    print("\nBuilding ArrowSpace...")
    t0 = time.perf_counter()
    builder = (ArrowSpaceBuilder()
        .with_dims_reduction(False, None)
        .with_sampling("simple", 1.0)
    )
    aspace, gl = builder.build(GRAPH_PARAMS, emb)
    print(f"Build time: {time.perf_counter() - t0:.2f}s")

    # Pre-compute topology scores (shared, query-independent)
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

    print(f"\nBuilding query embeddings ({len(queries)})...")
    qemb = build_embeddings(queries, cache_file="cve_queries_pplx1b.npy")

    all_results        = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        print(f"\n{'='*70}")
        print(f"Query {qi+1}/{len(queries)}: {q}")
        print("=" * 70)

        results_cosine  = aspace.search(qemb[qi], gl, tau=TAU_COSINE)
        results_hybrid  = aspace.search(qemb[qi], gl, tau=TAU_HYBRID)
        results_taumode = aspace.search(qemb[qi], gl, tau=TAU_TAUMODE)

        min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        print(f"Results  cosine={len(results_cosine)}  "
              f"hybrid={len(results_hybrid)}  "
              f"taumode={len(results_taumode)}  → using min={min_len}")

        results_cosine  = results_cosine[:min_len]
        results_hybrid  = results_hybrid[:min_len]
        results_taumode = results_taumode[:min_len]
        all_results.append((results_cosine, results_hybrid, results_taumode))

        # Correlation
        spear_ch, kend_ch = compute_ranking_metrics(results_cosine,  results_hybrid)
        spear_ct, kend_ct = compute_ranking_metrics(results_cosine,  results_taumode)
        spear_ht, kend_ht = compute_ranking_metrics(results_hybrid,  results_taumode)

        # NDCG
        k_ndcg  = min(10, min_len)
        ndcg_hc = compute_ndcg(results_hybrid,  results_cosine,  k=k_ndcg)
        ndcg_tc = compute_ndcg(results_taumode, results_cosine,  k=k_ndcg)
        ndcg_th = compute_ndcg(results_taumode, results_hybrid,  k=k_ndcg)

        # Tail
        tail_metrics = analyze_tail_distribution(
            [results_cosine, results_hybrid, results_taumode],
            TAU_LABELS, k_head=3, k_tail=K_TAIL_MAX,
        )

        # MRR-Top0
        mrr_cos = compute_mrr_top0(results_cosine,  topo_scores)
        mrr_hyb = compute_mrr_top0(results_hybrid,  topo_scores)
        mrr_tau = compute_mrr_top0(results_taumode, topo_scores)

        comparison_metrics.append({
            "query":       q,
            "min_length":  min_len,
            "spearman":    (spear_ch, spear_ct, spear_ht),
            "kendall":     (kend_ch,  kend_ct,  kend_ht),
            "ndcg":        (ndcg_hc,  ndcg_tc,  ndcg_th),
            "tail_metrics":tail_metrics,
            "mrr_top0":    (mrr_cos,  mrr_hyb,  mrr_tau),
        })

        # Top-10
        for label, results in zip(TAU_LABELS,
                                   [results_cosine, results_hybrid, results_taumode]):
            print(f"\n{label}")
            print("-" * 70)
            for rank, (idx, score) in enumerate(results[:10], 1):
                print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<40} [{score:.4f}]")

        print(f"\nCorrelations:")
        print(f"  Cosine vs Hybrid-{TAU_HYBRID}:        "
              f"ρ={spear_ch:.3f}  τ={kend_ch:.3f}")
        print(f"  Cosine vs Taumode-{TAU_TAUMODE}:      "
              f"ρ={spear_ct:.3f}  τ={kend_ct:.3f}")
        print(f"  Hybrid-{TAU_HYBRID} vs Taumode-{TAU_TAUMODE}: "
              f"ρ={spear_ht:.3f}  τ={kend_ht:.3f}")

        print(f"\nNDCG@{k_ndcg}:")
        print(f"  Hybrid-{TAU_HYBRID} vs Cosine:        {ndcg_hc:.4f}")
        print(f"  Taumode-{TAU_TAUMODE} vs Cosine:      {ndcg_tc:.4f}")
        print(f"  Taumode-{TAU_TAUMODE} vs Hybrid-{TAU_HYBRID}: {ndcg_th:.4f}")

        print(f"\nMRR-Top0 (topology-weighted):")
        for label, mrr in zip(TAU_LABELS, [mrr_cos, mrr_hyb, mrr_tau]):
            print(f"  {label}: {mrr:.4f}")

        if tail_metrics:
            k_tail = next(iter(tail_metrics.values()))["total_items"]
            print(f"\nTail Quality (Ranks 4–{k_tail}):")
            for label in TAU_LABELS:
                if label in tail_metrics:
                    tm = tail_metrics[label]
                    print(f"  {label}:  T/H={tm['tail_to_head_ratio']:.4f}  "
                          f"CV={tm['tail_cv']:.4f}")

    # Visualisations
    plot_comparison(queries, all_results, ids, titles)
    plot_mrr_top0(comparison_metrics)
    if all(min(len(r) for r in triple) > 3 for triple in all_results):
        plot_tail_comparison(queries, all_results, ids, titles)

    # CSV
    print(f"\n{'='*70}")
    print("EXPORTING RESULTS TO CSV")
    print("=" * 70)
    save_search_results_to_csv(queries, all_results, ids, titles)
    save_metrics_to_csv(comparison_metrics)
    save_tail_metrics_to_csv(comparison_metrics)
    save_summary_to_csv(comparison_metrics)
    save_query_comparison(queries, all_results, titles, docs)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY  [pplx-embed 1B via embeddings_ppx_1B.py]")
    print("=" * 70)

    print(f"\nAverage NDCG@10:")
    print(f"  Hybrid-{TAU_HYBRID} vs Cosine:   "
          f"{np.mean([m['ndcg'][0] for m in comparison_metrics]):.4f}")
    print(f"  Taumode-{TAU_TAUMODE} vs Cosine: "
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
                for m in valid_tail if label in m["tail_metrics"]
            ]
            if ratios:
                print(f"  {label}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

    print("\n→ Higher MRR-Top0 = better topology-aware ranking quality.")
    print("→ ArrowSpace τ<1.0 leverages spectral proximity for long-tail gains.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CVE search: pplx-embed (1B via embeddings_ppx_1B.py) + ArrowSpace + MRR-Top0"
    )
    parser.add_argument("--dataset", required=True, help="CVE dataset directory")
    args = parser.parse_args()
    main(args.dataset)
