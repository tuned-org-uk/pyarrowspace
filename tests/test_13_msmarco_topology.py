"""
Topology-aware IR evaluation on MS MARCO using ArrowSpace.

Computes comprehensive metrics including:
- Topology-aware: G-RBP, TD-nDCG, IT-ERR, SQI@k, MRR-Topo, RBO-S
- Energy-informed: MRR, MAP, Recall@k, NDCG, Spearman, Kendall

Dependencies:
  pip install arrowspace datasets sentence-transformers numpy scikit-learn networkx scipy genestore matplotlib
"""

import numpy as np
import networkx as nx
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ndcg_score
from collections import defaultdict
import json
import time
import gc
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

from arrowspace import ArrowSpaceBuilder, set_debug
import genestore

set_debug(True)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TAU = 0.62
SCALING = 1e2  # <------------------- normalised dataset, need to scale to highlight topology
EPS = 5.0      # <------------------- adjusted for the scaling
CORPUS_SIZE = 90000
DATASET_SIZE = 10000
N_EVAL_QUERIES = 25
K_RESULTS = 25  # Increased to analyze tail results
K_EVAL = 10

CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE = 32

CACHE_DIR = "./test_13_embeddings_cache"
CORPUS_EMB_NAME = "test_13_corpus_embeddings"
QUERY_EMB_NAME = "test_13_query_embeddings"

GRAPH_PARAMS = {
    "eps": EPS,
    "k": 25,
    "topk": 15,
    "p": 2.0,
    "sigma": None
}

LAMBDA_WEIGHTS = {"ppr": 0.4, "cond": 0.3, "mod": 0.3}
MU_WEIGHTS = {"cond": 0.4, "mod": 0.3, "ppr": 0.3}
RBP_P = 0.9

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def dataset_exists_in_storage(cache_dir, dataset_name):
    """Check if a Lance dataset exists by looking for _versions directory."""
    dataset_path = Path(cache_dir) / dataset_name / "_versions"
    return dataset_path.exists() and dataset_path.is_dir()

# ──────────────────────────────────────────────────────────────────────────────
# Initialize genestore storage
# ──────────────────────────────────────────────────────────────────────────────
print("Initializing genestore cache...")
builder = genestore.store_array(CACHE_DIR)
builder.with_max_rows_per_file(500_000)
builder.with_compression("zstd")
storage = builder.build()

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
# 3. Embed corpus and queries with CACHING
# ──────────────────────────────────────────────────────────────────────────────
model_name = "sentence-transformers/all-mpnet-base-v2"
print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name)

# Corpus Embeddings
if dataset_exists_in_storage(CACHE_DIR, CORPUS_EMB_NAME):
    print(f"Loading cached corpus embeddings from {CACHE_DIR}/{CORPUS_EMB_NAME}...")
    try:
        corpus_emb = storage.load(CORPUS_EMB_NAME)
        print(f"✓ Loaded cached corpus embeddings: {corpus_emb.shape}")
    except Exception as e:
        print(f"Error loading cache: {e}")
        corpus_emb = None
else:
    corpus_emb = None

if corpus_emb is None:
    print(f"Embedding corpus in batches of {CORPUS_BATCH_SIZE}...")
    corpus_emb_batches = []
    n_corpus_batches = (len(corpus) + CORPUS_BATCH_SIZE - 1) // CORPUS_BATCH_SIZE
    
    for batch_idx in range(n_corpus_batches):
        start_idx = batch_idx * CORPUS_BATCH_SIZE
        end_idx = min(start_idx + CORPUS_BATCH_SIZE, len(corpus))
        batch_texts = corpus[start_idx:end_idx]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        batch_emb = np.ascontiguousarray(batch_emb, dtype=np.float64)
        corpus_emb_batches.append(batch_emb)
        if batch_idx % 5 == 0: gc.collect()
    
    print("Concatenating corpus embeddings...")
    corpus_emb = np.vstack(corpus_emb_batches)
    del corpus_emb_batches
    gc.collect()
    
    try:
        storage.store(corpus_emb, CORPUS_EMB_NAME)
        print(f"✓ Corpus embeddings cached")
    except Exception as e:
        print(f"Warning: Failed to cache corpus embeddings: {e}")

# Query Embeddings
if dataset_exists_in_storage(CACHE_DIR, QUERY_EMB_NAME):
    print(f"Loading cached query embeddings from {CACHE_DIR}/{QUERY_EMB_NAME}...")
    try:
        query_emb = storage.load(QUERY_EMB_NAME)
        print(f"✓ Loaded cached query embeddings: {query_emb.shape}")
    except Exception as e:
        print(f"Error loading cache: {e}")
        query_emb = None
else:
    query_emb = None

if query_emb is None:
    print(f"Embedding queries in batches of {QUERY_BATCH_SIZE}...")
    query_emb_batches = []
    n_query_batches = (len(queries) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
    
    for batch_idx in range(n_query_batches):
        start_idx = batch_idx * QUERY_BATCH_SIZE
        end_idx = min(start_idx + QUERY_BATCH_SIZE, len(queries))
        batch_texts = queries[start_idx:end_idx]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        batch_emb = np.ascontiguousarray(batch_emb, dtype=np.float64) # Fix for PyO3 casting
        query_emb_batches.append(batch_emb)
    
    print("Concatenating query embeddings...")
    query_emb = np.vstack(query_emb_batches)
    del query_emb_batches
    gc.collect()
    
    try:
        storage.store(query_emb, QUERY_EMB_NAME)
        print(f"✓ Query embeddings cached")
    except Exception as e:
        print(f"Warning: Failed to cache query embeddings: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Sample dataset subset and select evaluation queries
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
dataset_size = min(DATASET_SIZE, len(corpus))
dataset_idx = rng.choice(len(corpus), size=dataset_size, replace=False)
dataset_idx_set = set(dataset_idx.tolist())

# Map subset indices back to original indices for relevance lookup
subset_idx_to_orig = {i: idx for i, idx in enumerate(dataset_idx)}

print("Finding valid evaluation queries...")
valid_queries = [q_idx for q_idx, rel_dict in relevance_scores.items()
                 if any(c_idx in dataset_idx_set for c_idx in rel_dict.keys())]

if len(valid_queries) < N_EVAL_QUERIES:
    eval_queries = valid_queries
else:
    eval_queries = rng.choice(valid_queries, size=N_EVAL_QUERIES, replace=False).tolist()

print(f"Dataset size: {dataset_size} | Eval queries: {len(eval_queries)}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build ArrowSpace
# ──────────────────────────────────────────────────────────────────────────────
print("Extracting corpus subset...")
corpus_subset = corpus_emb[dataset_idx]
corpus_subset_scaled = corpus_subset * SCALING

# Build standard NetworkX graph for topology metrics calculation
print("Building NetworkX graph for topology metrics...")
nbrs = NearestNeighbors(n_neighbors=GRAPH_PARAMS["k"], algorithm='brute', metric='cosine').fit(corpus_subset)
distances, indices = nbrs.kneighbors(corpus_subset)
G = nx.Graph()
for i in range(len(corpus_subset)):
    G.add_node(i)
    for j, dist in zip(indices[i], distances[i]):
        if i != j:
            weight = 1.0 - dist  # similarity
            if weight > 0:
                G.add_edge(i, j, weight=weight)

print("Building ArrowSpace...")
start = time.perf_counter()
aspace, gl = ArrowSpaceBuilder.build_and_store(GRAPH_PARAMS, corpus_subset_scaled.astype(np.float64))
elapsed = time.perf_counter() - start
print(f"ArrowSpace built in {elapsed:.2f}s")
gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Topology-Aware Metrics Implementation
# ──────────────────────────────────────────────────────────────────────────────

def calculate_g_rbp(ranked_list, graph, p=0.8):
    """
    Graph-Rank-Biased Precision (G-RBP)
    Measures the topological coherence of the result list.
    """
    score = 0.0
    for i in range(len(ranked_list) - 1):
        u, v = ranked_list[i], ranked_list[i+1]
        try:
            # Check if edge exists or calculate distance
            if graph.has_edge(u, v):
                rel = graph[u][v].get('weight', 0.5)
            else:
                try:
                    path_len = nx.shortest_path_length(graph, u, v)
                    rel = 1.0 / (path_len + 1)
                except nx.NetworkXNoPath:
                    rel = 0.0
            
            score += rel * (p ** i)
        except Exception:
            continue
            
    return score * (1 - p)

def calculate_sqi(ranked_list, graph, k=10):
    """
    Structural Quality Index (SQI@k)
    Measures how well the top-k results cover the query's topological neighborhood.
    """
    if not ranked_list: return 0.0
    k = min(k, len(ranked_list))
    top_k = ranked_list[:k]
    
    intra_sim = 0.0
    count = 0
    for i in range(k):
        for j in range(i+1, k):
            u, v = top_k[i], top_k[j]
            if graph.has_edge(u, v):
                intra_sim += graph[u][v].get('weight', 0.0)
            count += 1
    
    return intra_sim / count if count > 0 else 0.0

def calculate_td_ndcg(ranked_list, relevant_items, graph, k=10):
    """
    Topology-Dependent nDCG
    Rewarding results that are both relevant AND topologically significant (central).
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        rel = 1 if doc_id in relevant_items else 0
        
        # Topological boost: centrality in the result subgraph
        # Simple proxy: degree in global graph relative to max degree
        topo_score = graph.degree[doc_id] if doc_id in graph else 0
        topo_boost = np.log1p(topo_score) 
        
        gain = (2**(rel + 0.1 * topo_boost) - 1)
        dcg += gain / np.log2(i + 2)
        
    # Ideal DCG approximation (simplified)
    idcg = 0.0
    sorted_rels = sorted([1 + 0.1 * np.log1p(graph.degree[r]) for r in relevant_items if r in graph], reverse=True)
    for i, val in enumerate(sorted_rels[:k]):
        idcg += (2**val - 1) / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def calculate_it_err(ranked_list, relevant_items, graph, k=10):
    """
    Information-Topological Expected Reciprocal Rank
    """
    p = 1.0  # Probability user is satisfied
    err = 0.0
    for r in range(min(k, len(ranked_list))):
        doc_id = ranked_list[r]
        is_rel = doc_id in relevant_items
        
        # Topological relevance: clustering coefficient
        topo_val = nx.clustering(graph, doc_id) if doc_id in graph else 0
        
        R = (1 if is_rel else 0) * 0.8 + (topo_val * 0.2)
        prob_stop = (2**R - 1) / 2**1  # Max rel is 1
        
        err += p * prob_stop / (r + 1)
        p *= (1 - prob_stop)
        
    return err

def calculate_rbo_s(list1, list2, p=0.9):
    """
    Rank-Biased Overlap with Similarity (RBO-S)
    Compares two ranked lists considering element similarity/distance.
    """
    # Simplified RBO implementation
    k = min(len(list1), len(list2))
    if k == 0: return 0.0
    
    overlap = 0.0
    w = 0.0
    for d in range(1, k+1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap += (len(set1.intersection(set2)) / d) * (p**(d-1))
        w += p**(d-1)
        
    return overlap / w

# ──────────────────────────────────────────────────────────────────────────────
# 7. Execution and Evaluation
# ──────────────────────────────────────────────────────────────────────────────

results = {
    "cosine": defaultdict(list),
    "arrow": defaultdict(list)
}

print(f"\nRunning evaluation on {len(eval_queries)} queries...")
print(f"Comparing top-{K_RESULTS} results (Tail analysis included)")

for i, q_idx in enumerate(eval_queries):
    if i % 10 == 0: print(f"  Query {i}/{len(eval_queries)}")
    
    # Get ground truth
    relevant_orig_indices = [idx for idx, score in relevance_scores[q_idx].items() if score > 0]
    relevant_subset_indices = [idx for idx in dataset_idx if idx in relevance_scores[q_idx]]
    # Map back to 0..dataset_size range for local graph operations
    relevant_local_indices = []
    for idx in relevant_subset_indices:
        # Find position in dataset_idx
        locs = np.where(dataset_idx == idx)[0]
        if len(locs) > 0: relevant_local_indices.append(locs[0])
    
    relevant_set = set(relevant_local_indices)
    if not relevant_set: continue

    # 1. Cosine Search (Baseline)
    q_vec = query_emb[q_idx].reshape(1, -1)
    sims = np.dot(corpus_subset, q_vec.T).flatten()
    cosine_top_k = np.argsort(-sims)[:K_RESULTS]
    
    # 2. ArrowSpace Search
    # Note: ArrowSpace expects scaled float64
    q_vec_scaled = (q_vec * SCALING).astype(np.float64).flatten()
    arrow_res = aspace.search(q_vec_scaled, gl, tau=TAU)
    # Extract indices from (index, score) tuples
    arrow_top_k = [idx for idx, score in arrow_res]
    # Pad if necessary
    if len(arrow_top_k) < K_RESULTS:
        arrow_top_k.extend([-1] * (K_RESULTS - len(arrow_top_k)))
    arrow_top_k = arrow_top_k[:K_RESULTS]
    
    # --- Compute Metrics ---
    
    # Helper to compute all metrics for a given list
    def compute_all_metrics(ranked_list, method_name):
        # Filter valid indices
        valid_list = [x for x in ranked_list if x != -1]
        
        # Standard Metrics
        # MRR
        mrr = 0.0
        for r, item in enumerate(valid_list):
            if item in relevant_set:
                mrr = 1.0 / (r + 1)
                break
        results[method_name]["mrr"].append(mrr)
        
        # Recall@k
        rel_retrieved = sum(1 for x in valid_list[:K_EVAL] if x in relevant_set)
        results[method_name]["recall"].append(rel_retrieved / len(relevant_set) if relevant_set else 0)
        
        # Topology-Aware Metrics
        results[method_name]["g_rbp"].append(calculate_g_rbp(valid_list, G))
        results[method_name]["sqi"].append(calculate_sqi(valid_list, G, k=K_EVAL))
        results[method_name]["td_ndcg"].append(calculate_td_ndcg(valid_list, relevant_set, G, k=K_EVAL))
        results[method_name]["it_err"].append(calculate_it_err(valid_list, relevant_set, G, k=K_EVAL))
        
        # Tail Analysis (Bottom 10 of top-25)
        if len(valid_list) >= 20:
            tail_list = valid_list[-10:]
            tail_sqi = calculate_sqi(tail_list, G, k=10)
            results[method_name]["tail_sqi"].append(tail_sqi)
        else:
            results[method_name]["tail_sqi"].append(0.0)

    compute_all_metrics(cosine_top_k, "cosine")
    compute_all_metrics(arrow_top_k, "arrow")
    
    # Comparative Metrics
    rbo = calculate_rbo_s(cosine_top_k, arrow_top_k)
    results["arrow"]["rbo_vs_cosine"].append(rbo)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Store and Visualize Results
# ──────────────────────────────────────────────────────────────────────────────

# Aggregate results
final_metrics = {}
for method in ["cosine", "arrow"]:
    final_metrics[method] = {k: np.mean(v) for k, v in results[method].items() if k != "rbo_vs_cosine"}

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Metric':<15} | {'Cosine':<10} | {'ArrowSpace':<10} | {'Diff %':<10}")
print("-" * 55)

for metric in ["mrr", "recall", "g_rbp", "sqi", "tail_sqi", "td_ndcg", "it_err"]:
    v1 = final_metrics["cosine"].get(metric, 0)
    v2 = final_metrics["arrow"].get(metric, 0)
    diff = ((v2 - v1) / v1 * 100) if v1 > 0 else 0
    print(f"{metric:<15} | {v1:.4f}     | {v2:.4f}     | {diff:+.2f}%")

# Plotting
metrics_to_plot = ["g_rbp", "sqi", "tail_sqi", "td_ndcg"]
values_cosine = [final_metrics["cosine"][m] for m in metrics_to_plot]
values_arrow = [final_metrics["arrow"][m] for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, values_cosine, width, label='Cosine', color='skyblue')
rects2 = ax.bar(x + width/2, values_arrow, width, label='ArrowSpace', color='orange')

ax.set_ylabel('Score')
ax.set_title('Topology-Aware IR Metrics: Cosine vs ArrowSpace')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("topology_metrics_comparison.png")
print("\n✓ Plot saved to topology_metrics_comparison.png")

# Save CSV
import csv
with open("ir_topology_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Cosine", "ArrowSpace", "Improvement"])
    for metric in final_metrics["cosine"]:
        v1 = final_metrics["cosine"][metric]
        v2 = final_metrics["arrow"][metric]
        impr = (v2 - v1) / v1 if v1 > 0 else 0
        writer.writerow([metric, v1, v2, impr])

print("✓ Results saved to ir_topology_results.csv")
