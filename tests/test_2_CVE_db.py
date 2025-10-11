# Example: CVE semantic search with pyarrowspace
# Requirements:
#   pip install sentence-transformers numpy
#   pyarrowspace built/installed and importable
# ##################################################
# Need to download the CVE database in a local directory
# Directory structure:
#   dataset/
#     2025/
#       0xxx/
#         CVE-2025-0001.json
####################################################

import os, json, glob
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from arrowspace import ArrowSpaceBuilder, set_debug
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

set_debug(True)

START_YEAR = 1999
END_YEAR = 1999

def iter_cve_json(root_dir, start=START_YEAR, end=END_YEAR):
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end+1)):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    j = json.load(f)
                except Exception:
                    continue
            yield path, j

def extract_text(j):
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

def build_embeddings(texts, model_name="sentence-transformers/allenai-specter"):
    from sentence_transformers import SentenceTransformer
    model_path = "./domain_adapted_model"
    model = SentenceTransformer(model_path)
    print(f"Model successfully loaded from: {model_path}")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("example item: ", X[0][0:20])
    return X.astype(np.float64) * 1e1

def compute_ranking_metrics(results_a, results_b):
    """Compare two ranking lists using Spearman and Kendall-tau correlations"""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    
    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return None, None
    
    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]
    
    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau, _ = kendalltau(rank_a, rank_b)
    
    return spearman_rho, kendall_tau

def compute_ndcg(results_pred, results_ref, k=10):
    """
    Compute NDCG@k treating reference ranking as ground truth.
    
    Args:
        results_pred: Predicted ranking [(idx, score), ...]
        results_ref: Reference ranking (ground truth) [(idx, score), ...]
        k: Cutoff position
    
    Returns:
        NDCG@k score
    """
    # Create relevance scores based on reference ranking
    # Items ranked higher in reference get higher relevance
    ref_indices = [idx for idx, _ in results_ref[:k]]
    
    # Build relevance vector for predicted ranking
    # Relevance = k - position_in_reference (higher rank = higher relevance)
    relevance_map = {idx: k - i for i, idx in enumerate(ref_indices)}
    
    # Get relevance scores for predicted ranking
    pred_indices = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]
    
    # Compute NDCG (sklearn expects shape (1, k) for single query)
    if sum(true_relevance) == 0:
        return 0.0
    
    try:
        # Normalize scores to [0, 1] for sklearn
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

def plot_comparison(queries, all_results, ids, titles, output_file="cve_comparison.png"):
    """Plot comparative analysis of different tau values"""
    fig, axes = plt.subplots(len(queries), 3, figsize=(18, 6*len(queries)))
    if len(queries) == 1:
        axes = axes.reshape(1, -1)
    
    tau_labels = ["Cosine (τ=1.0)", "Hybrid (τ=0.8)", "Taumode (τ=0.62)"]
    
    for qi, query in enumerate(queries):
        results_cosine, results_hybrid, results_taumode = all_results[qi]
        
        k = min(10, len(results_cosine))
        
        for ti, (results, label) in enumerate([
            (results_cosine, tau_labels[0]),
            (results_hybrid, tau_labels[1]),
            (results_taumode, tau_labels[2])
        ]):
            ax = axes[qi, ti]
            
            scores = [score for _, score in results[:k]]
            indices_plot = list(range(1, k+1))
            
            ax.bar(indices_plot, scores, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'][ti])
            ax.set_xlabel("Rank", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"Query {qi+1}: {label}\n{query[:50]}...", 
                        fontsize=9, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            for i, (idx, score) in enumerate(results[:k]):
                ax.text(i+1, score + 0.01*max(scores), 
                       ids[idx].split('-')[-1],
                       ha='center', va='bottom', fontsize=6, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.close()

def main(dataset_root=None, k=10):
    if dataset_root is None: 
        raise ValueError("Dataset root not set")
    
    # 1) Load CVEs
    ids, titles, docs = [], [], []
    print("Start JSON iteration")
    for _, j in tqdm(iter_cve_json(dataset_root)):
        cve_id, title, text = extract_text(j)
        ids.append(cve_id)
        titles.append(title)
        docs.append(text)
    if not docs:
        raise SystemExit("No CVE JSON files found.")

    # 2) Embed all documents
    print(f"Start Embeddings for {len(ids)} files")
    emb = build_embeddings(docs)

    # 3) Build ArrowSpace index
    graph_params = {
        "eps": 2.5, 
        "k": 10,
        "topk": 10,
        "p": 2.0, 
        "sigma": 1.0
    }

    TAU_COSINE = 1.0
    TAU_HYBRID = 0.8
    TAU_TAUMODE = 0.62

    import time
    print(f"Build space")
    start_time = time.perf_counter()
    try:
        aspace, gl = ArrowSpaceBuilder.build(graph_params, emb)
    finally:
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.6f} seconds")

    # 4) Queries
    queries = [
        "authenticated arbitrary file read path traversal",
        "remote code execution in ERP web component",
        "SQL injection in login endpoint",
    ]

    # 5) Search and compare
    qemb = build_embeddings(queries)
    
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
        
        all_results.append((results_cosine, results_hybrid, results_taumode))
        
        # Compute ranking correlations
        spear_c_h, kendall_c_h = compute_ranking_metrics(results_cosine, results_hybrid)
        spear_c_t, kendall_c_t = compute_ranking_metrics(results_cosine, results_taumode)
        spear_h_t, kendall_h_t = compute_ranking_metrics(results_hybrid, results_taumode)
        
        # Compute NDCG@k (using cosine as reference/"ground truth")
        ndcg_hybrid_vs_cosine = compute_ndcg(results_hybrid, results_cosine, k=10)
        ndcg_taumode_vs_cosine = compute_ndcg(results_taumode, results_cosine, k=10)
        ndcg_taumode_vs_hybrid = compute_ndcg(results_taumode, results_hybrid, k=10)
        
        comparison_metrics.append({
            'query': q,
            'cosine_hybrid_spearman': spear_c_h,
            'cosine_hybrid_kendall': kendall_c_h,
            'cosine_taumode_spearman': spear_c_t,
            'cosine_taumode_kendall': kendall_c_t,
            'hybrid_taumode_spearman': spear_h_t,
            'hybrid_taumode_kendall': kendall_h_t,
            'ndcg_hybrid_vs_cosine': ndcg_hybrid_vs_cosine,
            'ndcg_taumode_vs_cosine': ndcg_taumode_vs_cosine,
            'ndcg_taumode_vs_hybrid': ndcg_taumode_vs_hybrid,
        })
        
        # Display results
        print(f"\n{'-'*70}")
        print(f"COSINE ONLY (τ={TAU_COSINE})")
        print('-'*70)
        for rank, (idx, score) in enumerate(results_cosine[:10], 1):
            print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<45} [score={score:.4f}]")
        
        print(f"\n{'-'*70}")
        print(f"HYBRID (τ={TAU_HYBRID})")
        print('-'*70)
        for rank, (idx, score) in enumerate(results_hybrid[:10], 1):
            print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<45} [score={score:.4f}]")
        
        print(f"\n{'-'*70}")
        print(f"TAUMODE (τ={TAU_TAUMODE})")
        print('-'*70)
        for rank, (idx, score) in enumerate(results_taumode[:10], 1):
            print(f"{rank:2d}. {ids[idx]:<18} {titles[idx]:<45} [score={score:.4f}]")
        
        # Print metrics
        print(f"\n{'-'*70}")
        print("RANKING CORRELATION METRICS")
        print('-'*70)
        print(f"Cosine vs Hybrid:   Spearman ρ={spear_c_h:.3f}, Kendall τ={kendall_c_h:.3f}")
        print(f"Cosine vs Taumode:  Spearman ρ={spear_c_t:.3f}, Kendall τ={kendall_c_t:.3f}")
        print(f"Hybrid vs Taumode:  Spearman ρ={spear_h_t:.3f}, Kendall τ={kendall_h_t:.3f}")
        
        print(f"\n{'-'*70}")
        print("NDCG@10 METRICS (higher = better agreement)")
        print('-'*70)
        print(f"Hybrid vs Cosine (ref):   NDCG@10 = {ndcg_hybrid_vs_cosine:.4f}")
        print(f"Taumode vs Cosine (ref):  NDCG@10 = {ndcg_taumode_vs_cosine:.4f}")
        print(f"Taumode vs Hybrid (ref):  NDCG@10 = {ndcg_taumode_vs_hybrid:.4f}")
    
    # 6) Generate visualization
    plot_comparison(queries, all_results, ids, titles)
    
    # 7) Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY: Average Metrics Across All Queries")
    print('='*70)
    
    print("\nRanking Correlations:")
    print(f"  Cosine vs Hybrid:   Spearman ρ={np.mean([m['cosine_hybrid_spearman'] for m in comparison_metrics]):.3f}, "
          f"Kendall τ={np.mean([m['cosine_hybrid_kendall'] for m in comparison_metrics]):.3f}")
    print(f"  Cosine vs Taumode:  Spearman ρ={np.mean([m['cosine_taumode_spearman'] for m in comparison_metrics]):.3f}, "
          f"Kendall τ={np.mean([m['cosine_taumode_kendall'] for m in comparison_metrics]):.3f}")
    print(f"  Hybrid vs Taumode:  Spearman ρ={np.mean([m['hybrid_taumode_spearman'] for m in comparison_metrics]):.3f}, "
          f"Kendall τ={np.mean([m['hybrid_taumode_kendall'] for m in comparison_metrics]):.3f}")
    
    print("\nNDCG@10 Agreement Scores:")
    print(f"  Hybrid vs Cosine (ref):   {np.mean([m['ndcg_hybrid_vs_cosine'] for m in comparison_metrics]):.4f}")
    print(f"  Taumode vs Cosine (ref):  {np.mean([m['ndcg_taumode_vs_cosine'] for m in comparison_metrics]):.4f}")
    print(f"  Taumode vs Hybrid (ref):  {np.mean([m['ndcg_taumode_vs_hybrid'] for m in comparison_metrics]):.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Correlation: 1.0=identical, 0.0=uncorrelated, -1.0=inverted")
    print(f"  NDCG@10: 1.0=perfect agreement, 0.0=no agreement")
    print(f"  Lower NDCG vs cosine = spectral features significantly rerank results")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ArrowSpace CVEs database demo")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    args = parser.parse_args()
    main(dataset_root=args.dataset)
