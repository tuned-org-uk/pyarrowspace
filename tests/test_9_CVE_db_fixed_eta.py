"""CVE semantic search with energy-only pipeline + energy clustered search - Diffusion parameter sweep with MRR
Requirements:
    pip install sentence-transformers numpy matplotlib scipy scikit-learn tqdm
Usage:
    python tests/test_cve_energy.py --dataset <dataset_dir>

**********************
Energymaps with sweeps of energy parameters and search_energy
*********************
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
START_YEAR = 1999
END_YEAR = 2025
K_RESULTS = 25
K_TAIL_MAX = 25

# Energy parameters for sweep
ETA_VALUES = [0.22, 0.535, 1.0]
STEPS_VALUES = [2, 4, 6, 8, 10]

# Standard graph params (used for both standard and energy builds)
graph_params = {
    "eps": 0.8,   # eps for 100K
    "k": 25,
    "topk": 15,
    "p": 2.0,
    "sigma": 0.2  # sigma for 100K
}

# ============================================================================
# Data Loading (unchanged from original)
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

def build_embeddings(texts, model_path="./domain_adapted_model"):
    """Generate embeddings using fine-tuned model."""
    model = SentenceTransformer(model_path)
    print(f"Model loaded from: {model_path}")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {X.shape}, sample: {X[0][:5]}...")
    return X.astype(np.float64) * 1.2e1

# ============================================================================
# NEW: MRR and Additional Metrics
# ============================================================================
def compute_mrr(results, relevant_set):
    """Compute Mean Reciprocal Rank for a single query.
    
    Args:
        results: List of (idx, score) tuples
        relevant_set: Set of indices considered relevant
    
    Returns:
        MRR score (1/rank of first relevant item, or 0 if none found)
    """
    for rank, (idx, _) in enumerate(results, 1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0

def compute_map(results, relevant_set, k=20):
    """Compute Mean Average Precision@k.
    
    Args:
        results: List of (idx, score) tuples
        relevant_set: Set of indices considered relevant
        k: Cutoff for evaluation
    
    Returns:
        MAP@k score
    """
    results_k = results[:k]
    relevant_count = 0
    precision_sum = 0.0
    
    for rank, (idx, _) in enumerate(results_k, 1):
        if idx in relevant_set:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            precision_sum += precision_at_rank
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / min(len(relevant_set), k)

def compute_recall_at_k(results, relevant_set, k=20):
    """Compute Recall@k."""
    results_k = results[:k]
    found = sum(1 for idx, _ in results_k if idx in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0

# ============================================================================
# Energy-Only Build and Search
# ============================================================================
def build_energy_index(emb, eta, steps, optical_tokens=None):
    """Build energy-only index with specified diffusion parameters."""
    energy_params = {
        "optical_tokens": None, # ← Rust computes 2√N automatically
        "trim_quantile": 0.1,
        "eta": eta,
        "steps": steps,
        "split_quantile": 0.9,
        "neighbor_k": 12,
        "split_tau": 0.15,
        "w_lambda": 1.0,
        "w_disp": 0.5,
        "w_dirichlet": 0.25,
        "candidate_m": 40,
    }
    
    print(f"Building energy index: η={eta}, steps={steps}, optical_tokens={optical_tokens}")
    start = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder.build_energy(
        emb,
        energy_params=energy_params,
        graph_params=graph_params
    )
    build_time = time.perf_counter() - start
    print(f"Energy build time: {build_time:.2f}s")
    
    return aspace, gl, build_time

# ============================================================================
# Diffusion Parameter Sweep
# ============================================================================
def sweep_diffusion_params(emb, qemb, queries, ids, titles):
    """Sweep eta and steps parameters and compute metrics."""
    results_sweep = {}
    
    # Build baseline standard index for comparison
    print("\n" + "="*70)
    print("Building BASELINE standard index...")
    print("="*70)
    start = time.perf_counter()
    aspace_std, gl_std = ArrowSpaceBuilder.build(graph_params, emb)
    std_build_time = time.perf_counter() - start
    print(f"Standard build time: {std_build_time:.2f}s")
    
    # Get baseline results
    baseline_results = []
    for qi, q in enumerate(queries):
        results = aspace_std.search(qemb[qi], gl_std, tau=0.7)
        baseline_results.append(results)
    
    del aspace_std, gl_std
    
    # Sweep diffusion parameters
    for eta in ETA_VALUES:
        for steps in STEPS_VALUES:
            config_key = f"eta{eta}_steps{steps}"
            print(f"\n{'='*70}")
            print(f"Testing: η={eta}, steps={steps}")
            print('='*70)
            
            try:
                aspace_energy, gl_energy, build_time = build_energy_index(
                    emb, eta, steps
                )
                
                # Test queries
                query_metrics = []
                for qi, q in enumerate(queries):
                    print(f"\nQuery {qi+1}: {q[:50]}...")
                    
                    # Energy search
                    results_energy = aspace_energy.search_energy(
                        qemb[qi], gl_energy, k=K_RESULTS
                    )
                    
                    # Define relevant set as top-10 from baseline
                    relevant_set = set(idx for idx, _ in baseline_results[qi][:10])
                    
                    # Compute metrics
                    mrr = compute_mrr(results_energy, relevant_set)
                    map_score = compute_map(results_energy, relevant_set, k=K_RESULTS)
                    recall_10 = compute_recall_at_k(results_energy, relevant_set, k=10)
                    recall_20 = compute_recall_at_k(results_energy, relevant_set, k=20)
                    
                    # Ranking correlation with baseline
                    spear, kendall = compute_ranking_metrics(
                        results_energy, baseline_results[qi]
                    )
                    
                    ndcg = compute_ndcg(results_energy, baseline_results[qi], k=10)
                    
                    query_metrics.append({
                        'query': q,
                        'mrr': mrr,
                        'map': map_score,
                        'recall@10': recall_10,
                        'recall@20': recall_20,
                        'spearman': spear,
                        'kendall': kendall,
                        'ndcg': ndcg,
                        'results': results_energy,
                    })
                    
                    print(f"  MRR: {mrr:.4f}")
                    print(f"  MAP@{K_RESULTS}: {map_score:.4f}")
                    print(f"  Recall@10: {recall_10:.4f}")
                    print(f"  Recall@20: {recall_20:.4f}")
                    print(f"  NDCG@10: {ndcg:.4f}")
                
                # Aggregate metrics
                avg_mrr = np.mean([m['mrr'] for m in query_metrics])
                avg_map = np.mean([m['map'] for m in query_metrics])
                avg_ndcg = np.mean([m['ndcg'] for m in query_metrics])
                avg_recall10 = np.mean([m['recall@10'] for m in query_metrics])
                
                results_sweep[config_key] = {
                    'eta': eta,
                    'steps': steps,
                    'build_time': build_time,
                    'avg_mrr': avg_mrr,
                    'avg_map': avg_map,
                    'avg_ndcg': avg_ndcg,
                    'avg_recall@10': avg_recall10,
                    'query_metrics': query_metrics,
                }
                
                print(f"\nAggregated metrics:")
                print(f"  Avg MRR: {avg_mrr:.4f}")
                print(f"  Avg MAP: {avg_map:.4f}")
                print(f"  Avg NDCG@10: {avg_ndcg:.4f}")
                print(f"  Avg Recall@10: {avg_recall10:.4f}")

                del aspace_energy, gl_energy 
            except Exception as e:
                print(f"ERROR: Failed for η={eta}, steps={steps}: {e}")
                results_sweep[config_key] = None
    
    return results_sweep, baseline_results, std_build_time

# ============================================================================
# Metrics (from original, kept for compatibility)
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
    kendall_tau_val, _ = kendalltau(rank_a, rank_b)
    
    return spearman_rho, kendall_tau_val

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

# ============================================================================
# Visualization: Heatmaps for parameter sweep
# ============================================================================
def plot_sweep_heatmaps(results_sweep, output_file="diffusion_sweep_heatmaps.png"):
    """Create heatmaps showing metric performance across eta/steps grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Prepare grids
    eta_vals = sorted(set(r['eta'] for r in results_sweep.values() if r))
    steps_vals = sorted(set(r['steps'] for r in results_sweep.values() if r))
    
    metrics_to_plot = [
        ('avg_mrr', 'Mean Reciprocal Rank'),
        ('avg_map', 'Mean Average Precision'),
        ('avg_ndcg', 'NDCG@10'),
        ('avg_recall@10', 'Recall@10'),
        ('build_time', 'Build Time (s)'),
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        # Build matrix
        matrix = np.zeros((len(steps_vals), len(eta_vals)))
        for i, steps in enumerate(steps_vals):
            for j, eta in enumerate(eta_vals):
                config_key = f"eta{eta}_steps{steps}"
                if config_key in results_sweep and results_sweep[config_key]:
                    matrix[i, j] = results_sweep[config_key][metric_key]
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(eta_vals)))
        ax.set_yticks(np.arange(len(steps_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in eta_vals])
        ax.set_yticklabels([f'{v}' for v in steps_vals])
        ax.set_xlabel('η (eta)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Steps', fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        
        # Annotate cells
        for i in range(len(steps_vals)):
            for j in range(len(eta_vals)):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Sweep heatmaps saved to {output_file}")
    plt.close()

# ============================================================================
# CSV Export for Sweep Results
# ============================================================================
def save_sweep_results_to_csv(results_sweep, output_file="diffusion_sweep_results.csv"):
    """Save parameter sweep results to CSV."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'eta', 'steps', 'build_time', 
            'avg_mrr', 'avg_map', 'avg_ndcg', 'avg_recall@10', 'avg_recall@20'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for config_key, data in results_sweep.items():
            if data:
                writer.writerow({
                    'eta': data['eta'],
                    'steps': data['steps'],
                    'build_time': f"{data['build_time']:.2f}",
                    'avg_mrr': f"{data['avg_mrr']:.4f}",
                    'avg_map': f"{data['avg_map']:.4f}",
                    'avg_ndcg': f"{data['avg_ndcg']:.4f}",
                    'avg_recall@10': f"{data['avg_recall@10']:.4f}",
                    'avg_recall@20': f"{data.get('avg_recall@20', 0):.4f}",
                })
    
    print(f"Sweep results saved to {output_file}")

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
    
    # Queries
    queries = [
        "authenticated arbitrary file read path traversal",
        "remote code execution in ERP web component",
        "SQL injection in login endpoint",
    ]
    
    print(f"\nGenerating query embeddings...")
    qemb = build_embeddings(queries)
    
    # Run parameter sweep
    print(f"\n{'='*70}")
    print("DIFFUSION PARAMETER SWEEP")
    print('='*70)
    
    results_sweep, baseline_results, std_build_time = sweep_diffusion_params(
        emb, qemb, queries, ids, titles
    )
    
    # Find best configuration
    valid_configs = {k: v for k, v in results_sweep.items() if v}
    if valid_configs:
        best_mrr_config = max(valid_configs.items(), key=lambda x: x[1]['avg_mrr'])
        best_map_config = max(valid_configs.items(), key=lambda x: x[1]['avg_map'])
        
        print(f"\n{'='*70}")
        print("BEST CONFIGURATIONS")
        print('='*70)
        print(f"\nBest MRR: η={best_mrr_config[1]['eta']}, steps={best_mrr_config[1]['steps']}")
        print(f"  MRR: {best_mrr_config[1]['avg_mrr']:.4f}")
        print(f"  MAP: {best_mrr_config[1]['avg_map']:.4f}")
        print(f"  NDCG: {best_mrr_config[1]['avg_ndcg']:.4f}")
        
        print(f"\nBest MAP: η={best_map_config[1]['eta']}, steps={best_map_config[1]['steps']}")
        print(f"  MRR: {best_map_config[1]['avg_mrr']:.4f}")
        print(f"  MAP: {best_map_config[1]['avg_map']:.4f}")
        print(f"  NDCG: {best_map_config[1]['avg_ndcg']:.4f}")
    
    # Visualizations
    plot_sweep_heatmaps(results_sweep, "diffusion_sweep_heatmaps.png")
    save_sweep_results_to_csv(results_sweep, "diffusion_sweep_results.csv")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print('='*70)
    print(f"Standard build time: {std_build_time:.2f}s")
    print(f"Energy builds tested: {len([r for r in results_sweep.values() if r])}/{len(ETA_VALUES) * len(STEPS_VALUES)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVE energy search with diffusion sweep")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    args = parser.parse_args()
    main(args.dataset)
