"""
Topology-aware IR evaluation on MS MARCO using ArrowSpace with TAU SWEEP.

Computes six graph-aware metrics (G-RBP, TD-nDCG, IT-ERR, SQI@k, MRR-Topo, RBO-S)
for THREE tau values: 0.62, 0.55, 0.42

Dependencies:
  pip install datasets sentence-transformers numpy scikit-learn networkx scipy
"""

import numpy as np
import networkx as nx
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import json
import time
import gc

from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TAU_VALUES = [0.62, 0.8, 0.42, 0.0]  # Tau sweep
SCALING = 1e2
EPS = 10.0
CORPUS_SIZE = 90000
DATASET_SIZE = 10000
N_EVAL_QUERIES = 10
K_RESULTS = 50
K_EVAL = 50
KNN_K = 10

CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE = 32

GRAPH_PARAMS = {
    "eps": EPS,
    "k": 25,
    "topk": 25,
    "p": 2.0,
    "sigma": None
}

LAMBDA_WEIGHTS = {"ppr": 0.4, "cond": 0.3, "mod": 0.3}
MU_WEIGHTS = {"cond": 0.4, "mod": 0.3, "ppr": 0.3}
RBP_P = 0.9

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load MS MARCO dataset
# ──────────────────────────────────────────────────────────────────────────────
print("Loading MS MARCO dataset...")
try:
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco", split="validation")
    print(f"Loaded: {len(corpus_ds)} passages, {len(queries_ds)} queries")
except Exception as e:
    print(f"Error: {e}. Using fallback dataset...")
    ds = load_dataset("ms_marco", "v1.1", split="validation")
    print(f"Loaded MS MARCO validation: {len(ds)} samples")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build corpus and relevance mappings
# ──────────────────────────────────────────────────────────────────────────────
corpus = []
corpus_id_to_idx = {}
queries = []
query_id_to_idx = {}
relevance_scores = defaultdict(lambda: defaultdict(int))

if 'corpus_ds' in locals():
    print("Processing corpus...")
    for i, row in enumerate(corpus_ds):
        if i >= CORPUS_SIZE:
            break
        if i % 10000 == 0:
            print(f"  Processed {i}/{CORPUS_SIZE} passages")
        passage_id = row["_id"]
        text = row["title"] + " " + row["text"] if row["title"] else row["text"]
        corpus.append(text)
        corpus_id_to_idx[passage_id] = len(corpus) - 1
    
    print("Processing queries...")
    for row in queries_ds:
        query_id = row["_id"]
        query_text = row["text"]
        queries.append(query_text)
        query_id_to_idx[query_id] = len(queries) - 1
    
    print("Processing relevance judgments...")
    for row in qrels_ds:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = int(row["score"])
        
        if query_id in query_id_to_idx and corpus_id in corpus_id_to_idx and score > 0:
            q_idx = query_id_to_idx[query_id]
            c_idx = corpus_id_to_idx[corpus_id]
            relevance_scores[q_idx][c_idx] = score
else:
    print("Processing fallback dataset...")
    for i, row in enumerate(ds):
        if i >= CORPUS_SIZE:
            break
        if i % 10000 == 0:
            print(f"  Processed {i}/{CORPUS_SIZE} samples")
        query = row["query"]
        passages = row["passages"]
        
        if query not in query_id_to_idx:
            queries.append(query)
            query_id_to_idx[query] = len(queries) - 1
        q_idx = query_id_to_idx[query]
        
        for j, passage in enumerate(passages["passage_text"]):
            is_selected = passages.get("is_selected", [0] * len(passages["passage_text"]))[j]
            if passage not in corpus_id_to_idx:
                corpus.append(passage)
                corpus_id_to_idx[passage] = len(corpus) - 1
            c_idx = corpus_id_to_idx[passage]
            if is_selected == 1:
                relevance_scores[q_idx][c_idx] = 1

corpus = np.array(corpus)
queries = np.array(queries)
print(f"Corpus: {len(corpus)} | Queries: {len(queries)} | Relevant pairs: {sum(len(v) for v in relevance_scores.values())}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Embed corpus and queries IN BATCHES
# ──────────────────────────────────────────────────────────────────────────────
model_name = "sentence-transformers/all-mpnet-base-v2"
print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name)

print(f"Embedding corpus in batches of {CORPUS_BATCH_SIZE}...")
corpus_emb_batches = []
n_corpus_batches = (len(corpus) + CORPUS_BATCH_SIZE - 1) // CORPUS_BATCH_SIZE

for batch_idx in range(n_corpus_batches):
    start_idx = batch_idx * CORPUS_BATCH_SIZE
    end_idx = min(start_idx + CORPUS_BATCH_SIZE, len(corpus))
    print(f"  Batch {batch_idx + 1}/{n_corpus_batches}: [{start_idx}:{end_idx}]")
    
    batch_texts = corpus[start_idx:end_idx]
    batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    corpus_emb_batches.append(batch_emb)
    
    if batch_idx % 5 == 0:
        gc.collect()

print("Concatenating corpus embeddings...")
corpus_emb = np.vstack(corpus_emb_batches)
del corpus_emb_batches
gc.collect()
print(f"Corpus embeddings: {corpus_emb.shape}")

print(f"Embedding queries in batches of {QUERY_BATCH_SIZE}...")
query_emb_batches = []
n_query_batches = (len(queries) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE

for batch_idx in range(n_query_batches):
    start_idx = batch_idx * QUERY_BATCH_SIZE
    end_idx = min(start_idx + QUERY_BATCH_SIZE, len(queries))
    print(f"  Batch {batch_idx + 1}/{n_query_batches}: [{start_idx}:{end_idx}]")
    
    batch_texts = queries[start_idx:end_idx]
    batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    query_emb_batches.append(batch_emb)

print("Concatenating query embeddings...")
query_emb = np.vstack(query_emb_batches)
del query_emb_batches
gc.collect()
print(f"Query embeddings: {query_emb.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Sample dataset subset and select evaluation queries
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
dataset_size = min(DATASET_SIZE, len(corpus))
dataset_idx = rng.choice(len(corpus), size=dataset_size, replace=False)
dataset_idx_set = set(dataset_idx.tolist())

print("Finding valid evaluation queries...")
valid_queries = [q_idx for q_idx, rel_dict in relevance_scores.items()
                 if any(c_idx in dataset_idx_set for c_idx in rel_dict.keys())]

if len(valid_queries) < N_EVAL_QUERIES:
    print(f"Warning: only {len(valid_queries)} queries have relevant docs in subset")
    eval_queries = valid_queries
else:
    eval_queries = rng.choice(valid_queries, size=N_EVAL_QUERIES, replace=False).tolist()

print(f"Dataset size: {dataset_size} | Eval queries: {len(eval_queries)}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build ArrowSpace (once, reused for all tau values)
# ──────────────────────────────────────────────────────────────────────────────
print("Extracting corpus subset...")
corpus_subset = corpus_emb[dataset_idx]
corpus_subset_scaled = corpus_subset * SCALING

print("Building ArrowSpace...")
start = time.perf_counter()
aspace, gl = ArrowSpaceBuilder.build(GRAPH_PARAMS, corpus_subset_scaled.astype(np.float64))
elapsed = time.perf_counter() - start
print(f"ArrowSpace built in {elapsed:.2f}s")
gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Build NetworkX graph for topology metrics
# ──────────────────────────────────────────────────────────────────────────────
print("Building kNN graph for topology metrics...")
nbrs = NearestNeighbors(n_neighbors=KNN_K + 1, metric='cosine', n_jobs=-1)
nbrs.fit(corpus_subset)
distances, indices = nbrs.kneighbors(corpus_subset)

G = nx.Graph()
G.add_nodes_from(range(len(corpus_subset)))
for i in range(len(corpus_subset)):
    for j_idx, j in enumerate(indices[i][1:]):
        weight = 1.0 - distances[i][j_idx + 1]
        if weight > 0:
            G.add_edge(i, j, weight=weight)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("Detecting communities...")
from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(G, weight='weight'))
node_to_community = {}
for comm_idx, comm in enumerate(communities):
    for node in comm:
        node_to_community[node] = comm_idx
print(f"Detected {len(communities)} communities")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Topology metric functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_ppr(G, query_node_idx, nodes_set, alpha=0.85, max_iter=100):
    if query_node_idx not in G:
        return {n: 1.0 / len(nodes_set) for n in nodes_set}
    personalization = {n: 0.0 for n in G.nodes()}
    personalization[query_node_idx] = 1.0
    try:
        ppr = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter, weight='weight')
        return {n: ppr.get(n, 0.0) for n in nodes_set}
    except:
        return {n: 1.0 / len(nodes_set) for n in nodes_set}

def compute_conductance(G, nodes_set):
    if len(nodes_set) == 0:
        return 1.0
    try:
        complement = set(G.nodes()) - nodes_set
        if len(complement) == 0:
            return 0.0
        return nx.conductance(G, nodes_set, complement, weight='weight')
    except:
        return 1.0

def compute_modularity_delta(G, nodes_set, node_to_community):
    if len(nodes_set) == 0:
        return 0.0
    comm_counts = defaultdict(int)
    for node in nodes_set:
        comm_counts[node_to_community.get(node, -1)] += 1
    max_comm_size = max(comm_counts.values()) if comm_counts else 0
    return max_comm_size / len(nodes_set)

def compute_topology_factor(G, query_node_idx, result_indices, node_to_community, lambdas):
    T = []
    for i, node in enumerate(result_indices):
        partial_set = set(result_indices[:i+1])
        ppr_vals = compute_ppr(G, query_node_idx, partial_set)
        ppr_score = ppr_vals.get(node, 0.0)
        cond = compute_conductance(G, partial_set)
        cond_score = 1.0 - cond
        mod_score = compute_modularity_delta(G, partial_set, node_to_community)
        T_i = (lambdas["ppr"] * ppr_score + lambdas["cond"] * cond_score + lambdas["mod"] * mod_score)
        T.append(T_i)
    return T

# ──────────────────────────────────────────────────────────────────────────────
# 8. Metric implementations
# ──────────────────────────────────────────────────────────────────────────────
def g_rbp(relevances, topology_factors, p=0.9, k=10):
    score = 0.0
    for i in range(min(k, len(relevances))):
        score += (1 - p) * relevances[i] * (p ** i) * topology_factors[i]
    return score

def td_ndcg(relevances, topology_factors, k=10):
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        gain = (2 ** relevances[i]) - 1
        discount = np.log2(i + 2)
        dcg += (gain * topology_factors[i]) / discount
    ideal_rel = sorted(relevances[:k], reverse=True)
    max_T = max(topology_factors[:k]) if topology_factors else 1.0
    idcg = sum((2 ** r - 1) * max_T / np.log2(i + 2) for i, r in enumerate(ideal_rel))
    return dcg / idcg if idcg > 0 else 0.0

def it_err(relevances, topology_factors, k=10, max_grade=3):
    err = 0.0
    prob_stop = 0.0
    for i in range(min(k, len(relevances))):
        R_i = relevances[i] / max_grade
        R_T = R_i * topology_factors[i]
        err += ((1 - prob_stop) * R_T) / (i + 1)
        prob_stop += (1 - prob_stop) * R_T
    return err

def mrr_topo(relevances, topology_factors):
    for i, rel in enumerate(relevances):
        if rel > 0:
            return topology_factors[i] / (i + 1)
    return 0.0

def sqi(G, result_indices, query_node_idx, node_to_community, mus, k=10):
    result_set = set(result_indices[:k])
    cond = compute_conductance(G, result_set)
    mod = compute_modularity_delta(G, result_set, node_to_community)
    ppr_vals = compute_ppr(G, query_node_idx, result_set)
    ppr_sum = sum(ppr_vals.values())
    return mus["cond"] * (1 - cond) + mus["mod"] * mod + mus["ppr"] * ppr_sum

def rbo_stability(rank_list1, rank_list2, p=0.9):
    def rbo_score(S, T, p, depth):
        score = 0.0
        for d in range(1, depth + 1):
            agreement = len(set(S[:d]) & set(T[:d])) / d
            score += (p ** (d - 1)) * agreement
        return (1 - p) * score
    return rbo_score(rank_list1, rank_list2, p, min(len(rank_list1), len(rank_list2)))

# ──────────────────────────────────────────────────────────────────────────────
# 9. Retrieval functions
# ──────────────────────────────────────────────────────────────────────────────
def cosine_search(query_idx, k=K_RESULTS):
    query_vec = query_emb[query_idx]
    similarities = query_vec @ corpus_subset.T
    top_indices = np.argsort(-similarities)[:k]
    return [int(dataset_idx[i]) for i in top_indices], top_indices.tolist()

def lambda_search(query_idx, tau, k=K_RESULTS):
    query_vec = query_emb[query_idx]
    results = aspace.search(query_vec.astype(np.float64), gl, tau=tau)
    local_indices = [i for i, _ in results][:k]
    return [int(dataset_idx[i]) for i in local_indices], local_indices

# ──────────────────────────────────────────────────────────────────────────────
# 10. TAU SWEEP EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
all_results = {}

for tau_val in TAU_VALUES:
    print(f"\n{'='*95}")
    print(f"EVALUATING TAU = {tau_val}")
    print(f"{'='*95}\n")
    
    results = {
        "cosine": {"g_rbp": [], "td_ndcg": [], "it_err": [], "mrr_topo": [], "sqi": [], "rbo_s": []},
        "lambda": {"g_rbp": [], "td_ndcg": [], "it_err": [], "mrr_topo": [], "sqi": [], "rbo_s": []}
    }
    
    for i, q_idx in enumerate(eval_queries):
        print(f"Query {i+1}/{len(eval_queries)}: q_idx={q_idx}")
        
        rel_dict = relevance_scores[q_idx]
        query_vec = query_emb[q_idx]
        dist_to_graph = np.linalg.norm(corpus_subset - query_vec, axis=1)
        query_anchor = int(np.argmin(dist_to_graph))
        
        # Cosine retrieval
        cosine_global, cosine_local = cosine_search(q_idx, K_RESULTS)
        cosine_rel = [rel_dict.get(doc_id, 0) for doc_id in cosine_global]
        cosine_T = compute_topology_factor(G, query_anchor, cosine_local, node_to_community, LAMBDA_WEIGHTS)
        
        results["cosine"]["g_rbp"].append(g_rbp(cosine_rel, cosine_T, RBP_P, K_EVAL))
        results["cosine"]["td_ndcg"].append(td_ndcg(cosine_rel, cosine_T, K_EVAL))
        results["cosine"]["it_err"].append(it_err(cosine_rel, cosine_T, K_EVAL))
        results["cosine"]["mrr_topo"].append(mrr_topo(cosine_rel, cosine_T))
        results["cosine"]["sqi"].append(sqi(G, cosine_local, query_anchor, node_to_community, MU_WEIGHTS, K_EVAL))
        
        # Lambda retrieval with current tau
        lambda_global, lambda_local = lambda_search(q_idx, tau_val, K_RESULTS)
        lambda_rel = [rel_dict.get(doc_id, 0) for doc_id in lambda_global]
        lambda_T = compute_topology_factor(G, query_anchor, lambda_local, node_to_community, LAMBDA_WEIGHTS)
        
        results["lambda"]["g_rbp"].append(g_rbp(lambda_rel, lambda_T, RBP_P, K_EVAL))
        results["lambda"]["td_ndcg"].append(td_ndcg(lambda_rel, lambda_T, K_EVAL))
        results["lambda"]["it_err"].append(it_err(lambda_rel, lambda_T, K_EVAL))
        results["lambda"]["mrr_topo"].append(mrr_topo(lambda_rel, lambda_T))
        results["lambda"]["sqi"].append(sqi(G, lambda_local, query_anchor, node_to_community, MU_WEIGHTS, K_EVAL))
        
        # RBO-Stability
        rbo_val = rbo_stability(cosine_local, lambda_local, RBP_P)
        results["cosine"]["rbo_s"].append(rbo_val)
        results["lambda"]["rbo_s"].append(rbo_val)
        
        print(f"  Cosine  → G-RBP={results['cosine']['g_rbp'][-1]:.4f} | MRR-Topo={results['cosine']['mrr_topo'][-1]:.4f}")
        print(f"  Lambda  → G-RBP={results['lambda']['g_rbp'][-1]:.4f} | MRR-Topo={results['lambda']['mrr_topo'][-1]:.4f}")
        print(f"  RBO-S = {rbo_val:.4f}\n")
    
    all_results[tau_val] = results

# ──────────────────────────────────────────────────────────────────────────────
# 11. SUMMARY TABLE FOR ALL TAU VALUES
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print(f"COMPARATIVE SUMMARY: TAU SWEEP RESULTS")
print(f"{'='*110}\n")

print(f"{'Tau / Method':<30} {'G-RBP':<12} {'TD-nDCG':<12} {'IT-ERR':<12} {'MRR-Topo':<12} {'SQI@k':<12} {'RBO-S':<12}")
print(f"{'-'*110}")

for tau_val in TAU_VALUES:
    results = all_results[tau_val]
    cosine_means = {m: np.mean(s) for m, s in results["cosine"].items()}
    lambda_means = {m: np.mean(s) for m, s in results["lambda"].items()}
    
    print(f"\nτ = {tau_val}")
    print(f"{'  Cosine':<30} {cosine_means['g_rbp']:<12.4f} {cosine_means['td_ndcg']:<12.4f} {cosine_means['it_err']:<12.4f} {cosine_means['mrr_topo']:<12.4f} {cosine_means['sqi']:<12.4f} {cosine_means['rbo_s']:<12.4f}")
    print(f"{'  Lambda':<30} {lambda_means['g_rbp']:<12.4f} {lambda_means['td_ndcg']:<12.4f} {lambda_means['it_err']:<12.4f} {lambda_means['mrr_topo']:<12.4f} {lambda_means['sqi']:<12.4f} {lambda_means['rbo_s']:<12.4f}")
    
    # Improvements
    improv = {m: ((lambda_means[m] - cosine_means[m]) / cosine_means[m] * 100) if cosine_means[m] > 0 else 0.0 
              for m in ["g_rbp", "td_ndcg", "it_err", "mrr_topo", "sqi"]}
    print(f"{'  Improvement (%)':<30} {improv['g_rbp']:>+11.2f}% {improv['td_ndcg']:>+11.2f}% {improv['it_err']:>+11.2f}% {improv['mrr_topo']:>+11.2f}% {improv['sqi']:>+11.2f}%")

print(f"\n{'='*110}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Save results
# ──────────────────────────────────────────────────────────────────────────────
output = {
    "dataset": "MS MARCO (BeIR)",
    "dataset_size": int(dataset_size),
    "num_queries": len(eval_queries),
    "k_eval": int(K_EVAL),
    "tau_values": [float(t) for t in TAU_VALUES],
    "rbp_p": float(RBP_P),
    "lambda_weights": {k: float(v) for k, v in LAMBDA_WEIGHTS.items()},
    "mu_weights": {k: float(v) for k, v in MU_WEIGHTS.items()},
    "results_by_tau": {}
}

for tau_val in TAU_VALUES:
    results = all_results[tau_val]
    cosine_means = {m: float(np.mean(s)) for m, s in results["cosine"].items()}
    lambda_means = {m: float(np.mean(s)) for m, s in results["lambda"].items()}
    improvements = {m: float((lambda_means[m] - cosine_means[m]) / cosine_means[m] * 100) if cosine_means[m] > 0 else 0.0
                   for m in ["g_rbp", "td_ndcg", "it_err", "mrr_topo", "sqi"]}
    
    output["results_by_tau"][str(tau_val)] = {
        "cosine": cosine_means,
        "lambda": lambda_means,
        "improvements": improvements,
        "per_query": {
            "cosine": {m: [float(v) for v in s] for m, s in results["cosine"].items()},
            "lambda": {m: [float(v) for v in s] for m, s in results["lambda"].items()}
        }
    }

with open("topology_aware_tau_sweep.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ Results saved to: topology_aware_tau_sweep.json")

"""
Visualization script for topology-aware IR evaluation results.
Generates comparison plots across metrics, queries, and tau values.

Usage:
    python visualize_results.py

Dependencies:
    pip install matplotlib seaborn pandas numpy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Output directory
OUTPUT_DIR = Path("./visualization_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load results
print("Loading results from topology_aware_tau_sweep.json...")
with open("topology_aware_tau_sweep.json", "r") as f:
    data = json.load(f)

TAU_VALUES = data["tau_values"]
NUM_QUERIES = data["num_queries"]
METRICS = ["g_rbp", "td_ndcg", "it_err", "mrr_topo", "sqi"]

print(f"Tau values: {TAU_VALUES}")
print(f"Number of queries: {NUM_QUERIES}")
print(f"Metrics: {METRICS}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PER-QUERY PERFORMANCE COMPARISON (for each tau)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_query_comparison():
    """Plot per-query scores for each metric and tau value."""
    print("\nGenerating per-query comparison plots...")
    
    for tau in TAU_VALUES:
        tau_str = str(tau)
        per_query_data = data["results_by_tau"][tau_str]["per_query"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Per-Query Performance (τ = {tau})', fontsize=16, fontweight='bold')
        axes = axes.ravel()
        
        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            
            cosine_scores = per_query_data["cosine"][metric]
            lambda_scores = per_query_data["lambda"][metric]
            queries = list(range(1, len(cosine_scores) + 1))
            
            # Plot lines
            ax.plot(queries, cosine_scores, 'o-', label='Cosine', linewidth=2, markersize=6, color='#1f77b4')
            ax.plot(queries, lambda_scores, 's-', label='Lambda', linewidth=2, markersize=6, color='#ff7f0e')
            
            ax.set_xlabel('Query Index', fontsize=11)
            ax.set_ylabel(metric.upper().replace('_', '-'), fontsize=11)
            ax.set_title(metric.upper().replace('_', '-'), fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(queries)
        
        # Hide extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'per_query_comparison_tau_{tau}.png')
        plt.close()
        print(f"  ✓ Saved: per_query_comparison_tau_{tau}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. METRIC COMPARISON ACROSS TAU VALUES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_metric_by_tau():
    """Compare each metric across different tau values."""
    print("\nGenerating metric-by-tau comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Metric Performance Across Tau Values', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        tau_list = []
        cosine_means = []
        lambda_means = []
        
        for tau in TAU_VALUES:
            tau_str = str(tau)
            tau_list.append(tau)
            cosine_means.append(data["results_by_tau"][tau_str]["cosine"][metric])
            lambda_means.append(data["results_by_tau"][tau_str]["lambda"][metric])
        
        x = np.arange(len(tau_list))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cosine_means, width, label='Cosine', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, lambda_means, width, label='Lambda', color='#ff7f0e', alpha=0.8)
        
        ax.set_xlabel('Tau Value', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.upper().replace('_', '-'), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t:.2f}' for t in tau_list])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_by_tau.png')
    plt.close()
    print("  ✓ Saved: metrics_by_tau.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPROVEMENT PERCENTAGE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
def plot_improvement_heatmap():
    """Heatmap showing improvement percentages for each metric and tau."""
    print("\nGenerating improvement heatmap...")
    
    improvements = []
    for tau in TAU_VALUES:
        tau_str = str(tau)
        tau_improvements = data["results_by_tau"][tau_str]["improvements"]
        improvements.append([tau_improvements[m] for m in METRICS])
    
    improvements_array = np.array(improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(improvements_array, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_yticks(np.arange(len(TAU_VALUES)))
    ax.set_xticklabels([m.upper().replace('_', '-') for m in METRICS])
    ax.set_yticklabels([f'τ = {t:.2f}' for t in TAU_VALUES])
    
    # Add text annotations
    for i in range(len(TAU_VALUES)):
        for j in range(len(METRICS)):
            text = ax.text(j, i, f'{improvements_array[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Improvement: Lambda vs Cosine (%)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'improvement_heatmap.png')
    plt.close()
    print("  ✓ Saved: improvement_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RADAR CHART COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar_charts():
    """Radar charts comparing cosine vs lambda for each tau."""
    print("\nGenerating radar charts...")
    
    from math import pi
    
    for tau in TAU_VALUES:
        tau_str = str(tau)
        cosine_vals = [data["results_by_tau"][tau_str]["cosine"][m] for m in METRICS]
        lambda_vals = [data["results_by_tau"][tau_str]["lambda"][m] for m in METRICS]
        
        # Normalize to 0-1 scale for better visualization
        all_vals = cosine_vals + lambda_vals
        max_val = max(all_vals)
        min_val = min(all_vals)
        
        cosine_norm = [(v - min_val) / (max_val - min_val) for v in cosine_vals]
        lambda_norm = [(v - min_val) / (max_val - min_val) for v in lambda_vals]
        
        # Number of variables
        categories = [m.upper().replace('_', '-') for m in METRICS]
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        cosine_norm += cosine_norm[:1]
        lambda_norm += lambda_norm[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, cosine_norm, 'o-', linewidth=2, label='Cosine', color='#1f77b4')
        ax.fill(angles, cosine_norm, alpha=0.25, color='#1f77b4')
        
        ax.plot(angles, lambda_norm, 'o-', linewidth=2, label='Lambda', color='#ff7f0e')
        ax.fill(angles, lambda_norm, alpha=0.25, color='#ff7f0e')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Comparison (τ = {tau})', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'radar_chart_tau_{tau}.png')
        plt.close()
        print(f"  ✓ Saved: radar_chart_tau_{tau}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-QUERY IMPROVEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_query_improvements():
    """Show which queries benefit most from lambda approach."""
    print("\nGenerating per-query improvement analysis...")
    
    for tau in TAU_VALUES:
        tau_str = str(tau)
        per_query_data = data["results_by_tau"][tau_str]["per_query"]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        queries = list(range(1, NUM_QUERIES + 1))
        x = np.arange(len(queries))
        width = 0.15
        
        for idx, metric in enumerate(METRICS):
            cosine_scores = np.array(per_query_data["cosine"][metric])
            lambda_scores = np.array(per_query_data["lambda"][metric])
            
            improvements = ((lambda_scores - cosine_scores) / (cosine_scores + 1e-10)) * 100
            
            offset = (idx - len(METRICS)/2) * width
            ax.bar(x + offset, improvements, width, label=metric.upper().replace('_', '-'), alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Query Index', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title(f'Per-Query Improvement: Lambda vs Cosine (τ = {tau})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(queries)
        ax.legend(loc='best', ncol=3)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'per_query_improvements_tau_{tau}.png')
        plt.close()
        print(f"  ✓ Saved: per_query_improvements_tau_{tau}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BOX PLOT: SCORE DISTRIBUTION ACROSS QUERIES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_score_distributions():
    """Box plots showing score distributions for each metric."""
    print("\nGenerating score distribution box plots...")
    
    for tau in TAU_VALUES:
        tau_str = str(tau)
        per_query_data = data["results_by_tau"][tau_str]["per_query"]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Score Distributions (τ = {tau})', fontsize=16, fontweight='bold')
        axes = axes.ravel()
        
        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            
            cosine_scores = per_query_data["cosine"][metric]
            lambda_scores = per_query_data["lambda"][metric]
            
            box_data = [cosine_scores, lambda_scores]
            bp = ax.boxplot(box_data, labels=['Cosine', 'Lambda'], patch_artist=True)
            
            # Color boxes
            colors = ['#1f77b4', '#ff7f0e']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(metric.upper().replace('_', '-'), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'score_distributions_tau_{tau}.png')
        plt.close()
        print(f"  ✓ Saved: score_distributions_tau_{tau}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def generate_summary_report():
    """Generate a text summary report."""
    print("\nGenerating summary report...")
    
    with open(OUTPUT_DIR / "summary_report.txt", "w") as f:
        f.write(f"K_EVAL {K_EVAL}\n")
        f.write(f"K_RESULTS {K_RESULTS}\n")
        f.write(f"KNN_K {KNN_K}\n")
        f.write(f"GRAPH PARAMS {GRAPH_PARAMS}\n")
        f.write("="*80 + "\n")
        f.write("TOPOLOGY-AWARE IR EVALUATION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {data['dataset']}\n")
        f.write(f"Dataset Size: {data['dataset_size']}\n")
        f.write(f"Number of Queries: {data['num_queries']}\n")
        f.write(f"Tau Values Tested: {TAU_VALUES}\n")
        f.write(f"Metrics: {METRICS}\n\n")
        
        for tau in TAU_VALUES:
            tau_str = str(tau)
            f.write("="*80 + "\n")
            f.write(f"TAU = {tau}\n")
            f.write("="*80 + "\n\n")
            
            cosine = data["results_by_tau"][tau_str]["cosine"]
            lambda_ = data["results_by_tau"][tau_str]["lambda"]
            improv = data["results_by_tau"][tau_str]["improvements"]
            
            f.write(f"{'Metric':<15} {'Cosine':<12} {'Lambda':<12} {'Improvement':<15}\n")
            f.write("-"*80 + "\n")
            
            for metric in METRICS:
                f.write(f"{metric.upper():<15} {cosine[metric]:<12.4f} {lambda_[metric]:<12.4f} {improv[metric]:>+14.2f}%\n")
            
            f.write("\n")
        
        # Best tau per metric
        f.write("="*80 + "\n")
        f.write("BEST TAU VALUE PER METRIC\n")
        f.write("="*80 + "\n\n")
        
        for metric in METRICS:
            best_tau = None
            best_improvement = -float('inf')
            
            for tau in TAU_VALUES:
                tau_str = str(tau)
                improvement = data["results_by_tau"][tau_str]["improvements"][metric]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_tau = tau
            
            f.write(f"{metric.upper():<15} τ = {best_tau:.2f}  ({best_improvement:+.2f}%)\n")
    
    print("  ✓ Saved: summary_report.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TOPOLOGY-AWARE IR EVALUATION - VISUALIZATION SUITE")
print("="*80)

plot_per_query_comparison()
plot_metric_by_tau()
plot_improvement_heatmap()
plot_radar_charts()
plot_per_query_improvements()
plot_score_distributions()
generate_summary_report()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*")):
    print(f"  ✓ {file.name}")
print("\n" + "="*80 + "\n")


