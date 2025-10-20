"""CVE semantic search with pyarrowspace - Multi-metric comparison with tail analysis
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
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from arrowspace import ArrowSpaceBuilder, set_debug
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

set_debug(True)

# Configuration
START_YEAR = 2001
END_YEAR = 2001
TAU_COSINE = 1.0    # Pure cosine similarity
TAU_HYBRID = 0.8    # Hybrid: mostly cosine, some spectral
TAU_TAUMODE = 0.62  # Spectral-aware (taumode)
K_TAIL_MAX = 20     # Analyze tail up to rank 20


# Build ArrowSpace
graph_params = {
    "eps": 1.0,
    "k": 10,
    "topk": 10,
    "p": 2.0,
    "sigma": None
}


# # Scale × Magnitude Matrix
# Combined effect on effective bandwidth:
# ----------------------------------------
# n_items  |  avg=0.1  |  avg=0.7  |  avg=2.0  |  avg=10.0
# ---------+-----------+-----------+-----------+----------
# 1K       |  0.8      |  5.8      |  16.5     |  82.7    
# 10K      |  1.9      |  13.2     |  37.7     |  188.6   
# 100K     |  4.2      |  29.6     |  84.7     |  423.3   
# 1M       |  9.4      |  65.7     |  187.7    |  938.4   
# 10M      |  20.6     |  144.2    |  411.9    |  2059.6  
#
# # Impact of Data Magnitude
# For n=10,000 items with f_dimensions=512:
# avg_value  |  eps     |  scaling  |  sigma   |  eff_bw  |  magnitude_factor  |  Needs Rescaling?
# -----------+----------+-----------+----------+----------+--------------------+------------------
# 0.01       |  0.016   |  12.00    |  0.014   |  0.19    |  0.014             |  ⚠️ YES          
# 0.10       |  0.157   |  12.00    |  0.143   |  1.89    |  0.143             |  ✓ No            
# 0.70       |  1.100   |  12.00    |  1.000   |  13.20   |  1.000             |  ✓ No            
# 1.00       |  1.571   |  12.00    |  1.429   |  18.86   |  1.429             |  ✓ No            
# 2.00       |  3.143   |  12.00    |  2.857   |  37.71   |  2.857             |  ✓ No            
# 5.00       |  7.857   |  12.00    |  7.143   |  94.29   |  7.143             |  ✓ No            
# 10.00      |  15.714  |  12.00    |  14.286  |  188.57  |  14.286            |  ✓ No            
# 50.00      |  78.571  |  12.00    |  71.429  |  942.86  |  71.429            |  ⚠️ YES          

# ============================================================================
# Data Loading
# ============================================================================
def iter_cve_json(root_dir, start=START_YEAR, end=END_YEAR):
    """Iterate over CVE JSON files in date range."""
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end+1)):
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

    # Descriptions
    descs = []
    for d in cna.get("descriptions", []) or []:
        if isinstance(d, dict):
            val = d.get("value") or ""
            if val:
                descs.append(val)
    description = " ".join(descs)

    # CWE IDs
    cwes = []
    for pt in cna.get("problemTypes", []) or []:
        for d in pt.get("descriptions", []) or []:
            cwe = d.get("cweId")
            if cwe:
                cwes.append(cwe)
    cwe_str = " ".join(cwes)

    # CVSS vector
    cvss_vec = ""
    for m in cna.get("metrics", []) or []:
        v31 = m.get("cvssV3_1")
        if isinstance(v31, dict):
            vs = v31.get("vectorString")
            if vs:
                cvss_vec = vs
                break

    # Affected products
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

def build_embeddings(texts, model_path="./domain_adapted_model"):
    """Generate embeddings using fine-tuned model."""
    model = SentenceTransformer(model_path)
    print(f"Model loaded from: {model_path}")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # if len(X) > 3:
    #     np.savetxt(f"cve{START_YEAR}-{END_YEAR}.csv", X, delimiter=",")
    print(f"Embeddings shape: {X.shape}, sample: {X[0][:5]}...")
    return X.astype(np.float64) * 1.2e1

# ============================================================================
# Metrics
# ============================================================================
def compute_ranking_metrics(results_a, results_b):
    """Compute Spearman and Kendall-tau correlations between two rankings."""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]

    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0

    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]

    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau, _ = kendalltau(rank_a, rank_b)

    return spearman_rho, kendall_tau

def compute_ndcg(results_pred, results_ref, k=10):
    """Compute NDCG@k treating reference ranking as ground truth."""
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

        score = ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k
        )
        return score
    except:
        return 0.0

def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    """
    Analyze score distribution in head vs tail.

    Assumes all results in results_list have been pre-trimmed to same length.
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

        tail_mean = np.mean(tail_scores)
        tail_std = np.std(tail_scores)
        head_mean = np.mean(head_scores)

        tail_to_head_ratio = tail_mean / head_mean if head_mean > 1e-10 else 0
        tail_cv = tail_std / tail_mean if tail_mean > 1e-10 else 0
        tail_decay = (tail_scores[0] - tail_scores[-1]) / len(tail_scores) if len(tail_scores) > 1 else 0

        metrics[label] = {
            'head_mean': head_mean,
            'tail_mean': tail_mean,
            'tail_std': tail_std,
            'tail_to_head_ratio': tail_to_head_ratio,
            'tail_cv': tail_cv,
            'tail_decay_rate': tail_decay,
            'n_tail_items': len(tail_scores),
            'total_items': actual_k_tail,
        }

    return metrics

# ============================================================================
# CSV Export Functions
# ============================================================================
def save_search_results_to_csv(queries, all_results, ids, titles, output_file="cve_search_results.csv"):
    """Save search results for all queries and tau values to CSV."""
    tau_labels = ["Cosine", "Hybrid", "Taumode"]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query_id', 'query_text', 'tau_method', 'rank', 'cve_id', 'title', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, query in enumerate(queries):
            results_cosine, results_hybrid, results_taumode = all_results[qi]

            for tau_label, results in zip(tau_labels, [results_cosine, results_hybrid, results_taumode]):
                for rank, (idx, score) in enumerate(results[:20], 1):  # Save top 20
                    writer.writerow({
                        'query_id': qi + 1,
                        'query_text': query,
                        'tau_method': tau_label,
                        'rank': rank,
                        'cve_id': ids[idx],
                        'title': titles[idx],
                        'score': f"{score:.6f}"
                    })

    print(f"Search results saved to {output_file}")

def save_metrics_to_csv(comparison_metrics, output_file="cve_comparison_metrics.csv"):
    """Save comparison metrics to CSV."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'query_id', 'query_text', 'min_length',
            'spearman_cosine_hybrid', 'spearman_cosine_taumode', 'spearman_hybrid_taumode',
            'kendall_cosine_hybrid', 'kendall_cosine_taumode', 'kendall_hybrid_taumode',
            'ndcg_hybrid_vs_cosine', 'ndcg_taumode_vs_cosine', 'ndcg_taumode_vs_hybrid'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, metrics in enumerate(comparison_metrics):
            writer.writerow({
                'query_id': qi + 1,
                'query_text': metrics['query'],
                'min_length': metrics['min_length'],
                'spearman_cosine_hybrid': f"{metrics['spearman'][0]:.6f}",
                'spearman_cosine_taumode': f"{metrics['spearman'][1]:.6f}",
                'spearman_hybrid_taumode': f"{metrics['spearman'][2]:.6f}",
                'kendall_cosine_hybrid': f"{metrics['kendall'][0]:.6f}",
                'kendall_cosine_taumode': f"{metrics['kendall'][1]:.6f}",
                'kendall_hybrid_taumode': f"{metrics['kendall'][2]:.6f}",
                'ndcg_hybrid_vs_cosine': f"{metrics['ndcg'][0]:.6f}",
                'ndcg_taumode_vs_cosine': f"{metrics['ndcg'][1]:.6f}",
                'ndcg_taumode_vs_hybrid': f"{metrics['ndcg'][2]:.6f}"
            })

    print(f"Comparison metrics saved to {output_file}")

def save_tail_metrics_to_csv(comparison_metrics, output_file="cve_tail_metrics.csv"):
    """Save tail analysis metrics to CSV."""
    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'query_id', 'query_text', 'tau_method',
            'head_mean', 'tail_mean', 'tail_std',
            'tail_to_head_ratio', 'tail_cv', 'tail_decay_rate',
            'n_tail_items', 'total_items'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for qi, metrics in enumerate(comparison_metrics):
            tail_metrics = metrics.get('tail_metrics', {})
            if not tail_metrics:
                continue

            for tau_label in tau_labels:
                if tau_label in tail_metrics:
                    m = tail_metrics[tau_label]
                    writer.writerow({
                        'query_id': qi + 1,
                        'query_text': metrics['query'],
                        'tau_method': tau_label,
                        'head_mean': f"{m['head_mean']:.6f}",
                        'tail_mean': f"{m['tail_mean']:.6f}",
                        'tail_std': f"{m['tail_std']:.6f}",
                        'tail_to_head_ratio': f"{m['tail_to_head_ratio']:.6f}",
                        'tail_cv': f"{m['tail_cv']:.6f}",
                        'tail_decay_rate': f"{m['tail_decay_rate']:.6f}",
                        'n_tail_items': m['n_tail_items'],
                        'total_items': m['total_items']
                    })

    print(f"Tail metrics saved to {output_file}")

def save_summary_to_csv(comparison_metrics, output_file="cve_summary.csv"):
    """Save summary statistics to CSV."""
    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]

    # Calculate averages
    avg_ndcg_h_c = np.mean([m['ndcg'][0] for m in comparison_metrics])
    avg_ndcg_t_c = np.mean([m['ndcg'][1] for m in comparison_metrics])
    avg_ndcg_t_h = np.mean([m['ndcg'][2] for m in comparison_metrics])

    valid_tail = [m for m in comparison_metrics if m['tail_metrics']]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['metric_type', 'metric_name', 'value', 'std_dev']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # NDCG averages
        writer.writerow({
            'metric_type': 'NDCG@10',
            'metric_name': 'Hybrid vs Cosine',
            'value': f"{avg_ndcg_h_c:.6f}",
            'std_dev': f"{np.std([m['ndcg'][0] for m in comparison_metrics]):.6f}"
        })
        writer.writerow({
            'metric_type': 'NDCG@10',
            'metric_name': 'Taumode vs Cosine',
            'value': f"{avg_ndcg_t_c:.6f}",
            'std_dev': f"{np.std([m['ndcg'][1] for m in comparison_metrics]):.6f}"
        })
        writer.writerow({
            'metric_type': 'NDCG@10',
            'metric_name': 'Taumode vs Hybrid',
            'value': f"{avg_ndcg_t_h:.6f}",
            'std_dev': f"{np.std([m['ndcg'][2] for m in comparison_metrics]):.6f}"
        })

        # Tail/Head ratios
        if valid_tail:
            for label in tau_labels:
                ratios = []
                for m in valid_tail:
                    if label in m['tail_metrics']:
                        ratios.append(m['tail_metrics'][label]['tail_to_head_ratio'])
                if ratios:
                    writer.writerow({
                        'metric_type': 'Tail/Head Ratio',
                        'metric_name': label,
                        'value': f"{np.mean(ratios):.6f}",
                        'std_dev': f"{np.std(ratios):.6f}"
                    })

    print(f"Summary statistics saved to {output_file}")

# ============================================================================
# Visualization
# ============================================================================
def plot_comparison(queries, all_results, ids, titles, output_file="cve_top10_comparison.png"):
    """Plot top-10 comparison across tau values."""
    n_queries = len(queries)
    fig, axes = plt.subplots(n_queries, 3, figsize=(18, 6*n_queries))
    if n_queries == 1:
        axes = axes.reshape(1, -1)

    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for qi, query in enumerate(queries):
        results_cosine, results_hybrid, results_taumode = all_results[qi]
        k = min(10, min(len(results_cosine), len(results_hybrid), len(results_taumode)))

        for ti, (results, label, color) in enumerate(zip(
            [results_cosine, results_hybrid, results_taumode],
            tau_labels, colors
        )):
            ax = axes[qi, ti]
            scores = [score for _, score in results[:k]]
            ranks = list(range(1, k+1))

            ax.bar(ranks, scores, alpha=0.7, color=color)
            ax.set_xlabel("Rank", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"Q{qi+1}: {label}\n{query[:50]}...",
                        fontsize=9, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            for i, (idx, score) in enumerate(results[:k]):
                ax.text(i+1, score + 0.01*max(scores) if scores else 0,
                       ids[idx].split('-')[-1],
                       ha='center', va='bottom', fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Top-10 plot saved to {output_file}")
    plt.close()

def plot_tail_comparison(queries, all_results, ids, titles, output_file="cve_tail_analysis.png"):
    """Create tail analysis visualization."""
    n_queries = len(queries)
    fig = plt.figure(figsize=(20, 5*n_queries))
    gs = fig.add_gridspec(n_queries, 4, hspace=0.3, wspace=0.3)

    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for qi, query in enumerate(queries):
        results_cosine, results_hybrid, results_taumode = all_results[qi]
        k = min(len(results_cosine), len(results_hybrid), len(results_taumode))

        results_trimmed = [
            results_cosine[:k],
            results_hybrid[:k],
            results_taumode[:k]
        ]

        # Panel 1: Full distribution
        ax1 = fig.add_subplot(gs[qi, 0])
        ranks = list(range(1, k+1))

        for results, label, color in zip(results_trimmed, tau_labels, colors):
            scores = [score for _, score in results]
            ax1.plot(ranks, scores, marker='o', label=label, color=color,
                    alpha=0.7, markersize=4, linewidth=2)

        ax1.axvline(x=3.5, color='red', linestyle='--', alpha=0.5, linewidth=2,
                   label='Head/Tail')
        ax1.set_xlabel("Rank", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Score", fontsize=11, fontweight='bold')
        ax1.set_title(f"Q{qi+1}: Score Distribution (n={k})\n{query[:45]}...",
                     fontsize=10, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(alpha=0.3)

        # Panel 2: Tail only
        ax2 = fig.add_subplot(gs[qi, 1])
        if k > 3:
            tail_ranks = list(range(4, k+1))
            for results, label, color in zip(results_trimmed, tau_labels, colors):
                tail_scores = [score for _, score in results[3:]]
                ax2.plot(tail_ranks, tail_scores, marker='s', label=label,
                        color=color, alpha=0.7, markersize=5, linewidth=2)

            ax2.set_xlabel("Rank", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Score", fontsize=11, fontweight='bold')
            ax2.set_title(f"Q{qi+1}: Tail (Ranks 4-{k})",
                         fontsize=10, fontweight='bold')
            ax2.legend(fontsize=9, loc='best')
            ax2.grid(alpha=0.3)

        # Panel 3: Box plot
        ax3 = fig.add_subplot(gs[qi, 2])
        if k > 3:
            tail_data = [[score for _, score in r[3:]] for r in results_trimmed]
            bp = ax3.boxplot(tail_data, labels=['Cosine', 'Hybrid', 'Taumode'],
                           patch_artist=True, widths=0.6)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax3.set_ylabel("Score", fontsize=11, fontweight='bold')
            ax3.set_title(f"Q{qi+1}: Tail Variability",
                         fontsize=10, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)

        # Panel 4: Metrics
        ax4 = fig.add_subplot(gs[qi, 3])
        if k > 3:
            tail_metrics = analyze_tail_distribution(results_trimmed, tau_labels, k_head=3, k_tail=k)

            metrics_names = ['Tail Mean', 'T/H Ratio', 'Stability']
            x_pos = np.arange(len(metrics_names))
            width = 0.25

            for i, (label, color) in enumerate(zip(tau_labels, colors)):
                if label in tail_metrics:
                    m = tail_metrics[label]
                    values = [
                        m['tail_mean'],
                        m['tail_to_head_ratio'],
                        1.0 / (1.0 + m['tail_cv']) if m['tail_cv'] > 0 else 1.0
                    ]
                    ax4.bar(x_pos + i*width, values, width, label=label,
                           color=color, alpha=0.7)

            ax4.set_ylabel("Value", fontsize=11, fontweight='bold')
            ax4.set_title(f"Q{qi+1}: Tail Metrics",
                         fontsize=10, fontweight='bold')
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(metrics_names, fontsize=9, rotation=15, ha='right')
            ax4.legend(fontsize=8, loc='best')
            ax4.grid(axis='y', alpha=0.3)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Tail analysis plot saved to {output_file}")
    plt.close()

# ============================================================================
# Main
# ============================================================================
def main(dataset_root):
    # Load CVEs
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

    # Build embeddings
    print("Generating embeddings...")
    emb = build_embeddings(docs)

    print("Building ArrowSpace...")
    start = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder.build(graph_params, emb)
    print(f"Build time: {time.perf_counter() - start:.2f}s")

    # Queries
    queries = [
        "authenticated arbitrary file read path traversal",
        "remote code execution in ERP web component",
        "SQL injection in login endpoint",
    ]

    print(f"\nSearching {len(queries)} queries...")
    qemb = build_embeddings(queries)

    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]
    all_results = []
    comparison_metrics = []

    for qi, q in enumerate(queries):
        print(f"\n{'='*70}")
        print(f"Query {qi+1}: {q}")
        print('='*70)

        # Search with three tau values
        results_cosine  = aspace.search(qemb[qi], gl, tau=TAU_COSINE)
        results_hybrid  = aspace.search(qemb[qi], gl, tau=TAU_HYBRID)
        results_taumode = aspace.search(qemb[qi], gl, tau=TAU_TAUMODE)

        # Trim to minimum length
        min_len = min(len(results_cosine), len(results_hybrid), len(results_taumode))
        print(f"Results: cosine={len(results_cosine)}, hybrid={len(results_hybrid)}, "
              f"taumode={len(results_taumode)}, using min={min_len}")

        results_cosine = results_cosine[:min_len]
        results_hybrid = results_hybrid[:min_len]
        results_taumode = results_taumode[:min_len]

        all_results.append((results_cosine, results_hybrid, results_taumode))

        # Metrics
        spear_c_h, kendall_c_h = compute_ranking_metrics(results_cosine, results_hybrid)
        spear_c_t, kendall_c_t = compute_ranking_metrics(results_cosine, results_taumode)
        spear_h_t, kendall_h_t = compute_ranking_metrics(results_hybrid, results_taumode)

        k_ndcg = min(10, min_len)
        ndcg_h_c = compute_ndcg(results_hybrid, results_cosine, k=k_ndcg)
        ndcg_t_c = compute_ndcg(results_taumode, results_cosine, k=k_ndcg)
        ndcg_t_h = compute_ndcg(results_taumode, results_hybrid, k=k_ndcg)

        tail_metrics = analyze_tail_distribution(
            [results_cosine, results_hybrid, results_taumode],
            tau_labels, k_head=3, k_tail=K_TAIL_MAX
        )

        comparison_metrics.append({
            'query': q,
            'min_length': min_len,
            'spearman': (spear_c_h, spear_c_t, spear_h_t),
            'kendall': (kendall_c_h, kendall_c_t, kendall_h_t),
            'ndcg': (ndcg_h_c, ndcg_t_c, ndcg_t_h),
            'tail_metrics': tail_metrics,
        })

        # Display top-10
        for label, results in zip(tau_labels, [results_cosine, results_hybrid, results_taumode]):
            print(f"\n{label}")
            print('-'*70)
            for rank, (idx, score) in enumerate(results[:10], 1):
                print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<40} [{score:.4f}]")

        # Print metrics
        print(f"\nCorrelations:")
        print(f"  Cosine vs Hybrid:   ρ={spear_c_h:.3f}, τ={kendall_c_h:.3f}")
        print(f"  Cosine vs Taumode:  ρ={spear_c_t:.3f}, τ={kendall_c_t:.3f}")
        print(f"  Hybrid vs Taumode:  ρ={spear_h_t:.3f}, τ={kendall_h_t:.3f}")

        print(f"\nNDCG@{k_ndcg}:")
        print(f"  Hybrid vs Cosine:   {ndcg_h_c:.4f}")
        print(f"  Taumode vs Cosine:  {ndcg_t_c:.4f}")
        print(f"  Taumode vs Hybrid:  {ndcg_t_h:.4f}")

        if tail_metrics:
            k_tail = tail_metrics[tau_labels[0]]['total_items']
            print(f"\nTail Quality (Ranks 4-{k_tail}):")
            for label in tau_labels:
                if label in tail_metrics:
                    m = tail_metrics[label]
                    print(f"  {label}:")
                    print(f"    T/H ratio: {m['tail_to_head_ratio']:.4f}")
                    print(f"    CV: {m['tail_cv']:.4f}")

    # Visualizations
    plot_comparison(queries, all_results, ids, titles, "cve_top10_comparison.png")

    if all(min(len(r[0]), len(r[1]), len(r[2])) > 3 for r in all_results):
        plot_tail_comparison(queries, all_results, ids, titles, "cve_tail_analysis.png")

    # ============================================================================
    # CSV EXPORTS
    # ============================================================================
    print(f"\n{'='*70}")
    print("EXPORTING RESULTS TO CSV")
    print('='*70)

    save_search_results_to_csv(queries, all_results, ids, titles, "cve_search_results.csv")
    save_metrics_to_csv(comparison_metrics, "cve_comparison_metrics.csv")
    save_tail_metrics_to_csv(comparison_metrics, "cve_tail_metrics.csv")
    save_summary_to_csv(comparison_metrics, "cve_summary.csv")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)

    avg_ndcg_h_c = np.mean([m['ndcg'][0] for m in comparison_metrics])
    avg_ndcg_t_c = np.mean([m['ndcg'][1] for m in comparison_metrics])

    print(f"\nAverage NDCG@10:")
    print(f"  Hybrid vs Cosine:   {avg_ndcg_h_c:.4f}")
    print(f"  Taumode vs Cosine:  {avg_ndcg_t_c:.4f}")

    valid_tail = [m for m in comparison_metrics if m['tail_metrics']]
    if valid_tail:
        print(f"\nAverage Tail/Head Ratios:")
        for label in tau_labels:
            ratios = []
            for m in valid_tail:
                if label in m['tail_metrics']:
                    ratios.append(m['tail_metrics'][label]['tail_to_head_ratio'])
            if ratios:
                print(f"  {label}: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")

        print(f"\n→ Higher T/H ratio = Better long-tail quality")
        print(f"→ ArrowSpace (τ<1.0) maintains higher tail scores")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVE search with tail analysis")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    args = parser.parse_args()
    main(args.dataset)
