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
SCALING = 1e2
EPS = 10.0
CORPUS_SIZE = 90000
DATASET_SIZE = 30000
N_EVAL_QUERIES = 100
K_RESULTS = 20
K_EVAL = 10

CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE = 32

CACHE_DIR = "./test_12_embeddings_cache"
CORPUS_EMB_NAME = "test_12_corpus_embeddings"
QUERY_EMB_NAME = "test_12_query_embeddings"

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

# Corpus Embeddings with Caching
if dataset_exists_in_storage(CACHE_DIR, CORPUS_EMB_NAME):
    print(f"Loading cached corpus embeddings from {CACHE_DIR}/{CORPUS_EMB_NAME}...")
    try:
        corpus_emb = storage.load(CORPUS_EMB_NAME)
        print(f"✓ Loaded cached corpus embeddings: {corpus_emb.shape}")
    except Exception as e:
        print(f"Error loading cache: {e}")
        print("Proceeding to compute embeddings...")
        corpus_emb = None
else:
    print(f"No cached corpus embeddings found in {CACHE_DIR}/{CORPUS_EMB_NAME}")
    corpus_emb = None

STORED = False
if corpus_emb is None:
    print(f"Embedding corpus in batches of {CORPUS_BATCH_SIZE}...")
    corpus_emb_batches = []
    n_corpus_batches = (len(corpus) + CORPUS_BATCH_SIZE - 1) // CORPUS_BATCH_SIZE
    
    for batch_idx in range(n_corpus_batches):
        start_idx = batch_idx * CORPUS_BATCH_SIZE
        end_idx = min(start_idx + CORPUS_BATCH_SIZE, len(corpus))
        print(f"  Batch {batch_idx + 1}/{n_corpus_batches}: [{start_idx}:{end_idx}]")
        
        batch_texts = corpus[start_idx:end_idx]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        batch_emb = np.ascontiguousarray(batch_emb, dtype=np.float64)
        corpus_emb_batches.append(batch_emb)
        
        if batch_idx % 5 == 0:
            gc.collect()
    
    print("Concatenating corpus embeddings...")
    corpus_emb = np.vstack(corpus_emb_batches)
    del corpus_emb_batches
    gc.collect()
    
    print(f"Saving corpus embeddings to {CACHE_DIR}/{CORPUS_EMB_NAME}...")
    try:
        storage.store(corpus_emb, CORPUS_EMB_NAME)
        print(f"✓ Corpus embeddings cached")
        STORED = True
    except Exception as e:
        print(f"Warning: Failed to cache corpus embeddings: {e}")

print(f"Corpus embeddings: {corpus_emb.shape}")

# Query Embeddings with Caching
if dataset_exists_in_storage(CACHE_DIR, QUERY_EMB_NAME):
    print(f"Loading cached query embeddings from {CACHE_DIR}/{QUERY_EMB_NAME}...")
    try:
        query_emb = storage.load(QUERY_EMB_NAME)
        print(f"✓ Loaded cached query embeddings: {query_emb.shape}")
    except Exception as e:
        print(f"Error loading cache: {e}")
        print("Proceeding to compute embeddings...")
        query_emb = None
else:
    print(f"No cached query embeddings found in {CACHE_DIR}/{QUERY_EMB_NAME}")
    query_emb = None

if query_emb is None:
    print(f"Embedding queries in batches of {QUERY_BATCH_SIZE}...")
    query_emb_batches = []
    n_query_batches = (len(queries) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
    
    for batch_idx in range(n_query_batches):
        start_idx = batch_idx * QUERY_BATCH_SIZE
        end_idx = min(start_idx + QUERY_BATCH_SIZE, len(queries))
        print(f"  Batch {batch_idx + 1}/{n_query_batches}: [{start_idx}:{end_idx}]")
        
        batch_texts = queries[start_idx:end_idx]
        batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        # Convert to float64 and ensure contiguous before appending
        batch_emb = np.ascontiguousarray(batch_emb, dtype=np.float64)
        query_emb_batches.append(batch_emb)
    
    print("Concatenating query embeddings...")
    query_emb = np.vstack(query_emb_batches)
    del query_emb_batches
    gc.collect()
    
    print(f"Saving query embeddings to {CACHE_DIR}/{QUERY_EMB_NAME}...")
    try:
        storage.store(query_emb, QUERY_EMB_NAME)
        print(f"✓ Query embeddings cached")
        STORED = True
    except Exception as e:
        print(f"Warning: Failed to cache query embeddings: {e}")

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
# 5. Build ArrowSpace
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
knn_k = 15
nbrs = NearestNeighbors(n_neighbors=knn_k + 1, metric='cosine', n_jobs=-1)
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
    """Compute personalized PageRank from query node for result nodes."""
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
    """Compute conductance of a node set."""
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
    """Compute modularity contribution of a node set."""
    if len(nodes_set) == 0:
        return 0.0
    
    comm_counts = defaultdict(int)
    for node in nodes_set:
        comm_counts[node_to_community.get(node, -1)] += 1
    
    max_comm_size = max(comm_counts.values()) if comm_counts else 0
    return max_comm_size / len(nodes_set)

def compute_topology_factor(G, query_node_idx, result_indices, node_to_community, lambdas):
    """Compute topology factor T_i for each result rank i."""
    T = []
    for i, node in enumerate(result_indices):
        partial_set = set(result_indices[:i+1])
        
        ppr_vals = compute_ppr(G, query_node_idx, partial_set)
        ppr_score = ppr_vals.get(node, 0.0)
        
        cond = compute_conductance(G, partial_set)
        cond_score = 1.0 - cond
        
        mod_score = compute_modularity_delta(G, partial_set, node_to_community)
        
        T_i = (lambdas["ppr"] * ppr_score +
               lambdas["cond"] * cond_score +
               lambdas["mod"] * mod_score)
        T.append(T_i)
    
    return T

# ──────────────────────────────────────────────────────────────────────────────
# 8. Energy-informed metrics (from test_9_CVE_db_fixed_eta.py)
# ──────────────────────────────────────────────────────────────────────────────
def compute_mrr(results, relevant_set):
    """Compute Mean Reciprocal Rank for a single query."""
    for rank, idx in enumerate(results, 1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0

def compute_map(results, relevant_set, k=20):
    """Compute Mean Average Precision@k."""
    results_k = results[:k]
    relevant_count = 0
    precision_sum = 0.0
    
    for rank, idx in enumerate(results_k, 1):
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
    found = sum(1 for idx in results_k if idx in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0

def compute_ranking_metrics(results_a, results_b):
    """Compute Spearman and Kendall-tau correlations between two rankings."""
    shared = set(results_a) & set(results_b)
    if len(shared) < 2:
        return 0.0, 0.0
    
    rank_a = [results_a.index(idx) for idx in shared]
    rank_b = [results_b.index(idx) for idx in shared]
    
    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau_val, _ = kendalltau(rank_a, rank_b)
    
    return spearman_rho, kendall_tau_val

def compute_ndcg_energy(results_pred, results_ref, k=10):
    """Compute NDCG@k treating reference ranking as ground truth."""
    ref_indices = results_ref[:k]
    relevance_map = {idx: k - i for i, idx in enumerate(ref_indices)}
    
    pred_indices = results_pred[:k]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]
    
    if sum(true_relevance) == 0:
        return 0.0
    
    try:
        # Create dummy scores for predicted results
        pred_scores = np.arange(len(pred_indices), 0, -1)
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

# ──────────────────────────────────────────────────────────────────────────────
# 9. Topology-aware metric implementations
# ──────────────────────────────────────────────────────────────────────────────
def g_rbp(relevances, topology_factors, p=0.9, k=10):
    """Graph-aware Rank-Biased Precision."""
    score = 0.0
    for i in range(min(k, len(relevances))):
        r_i = relevances[i]
        T_i = topology_factors[i]
        score += (1 - p) * r_i * (p ** i) * T_i
    return score

def td_ndcg(relevances, topology_factors, k=10):
    """Topology-Discounted nDCG."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        gain = (2 ** relevances[i]) - 1
        discount = np.log2(i + 2)
        T_i = topology_factors[i]
        dcg += (gain * T_i) / discount
    
    ideal_rel = sorted(relevances[:k], reverse=True)
    max_T = max(topology_factors[:k]) if topology_factors else 1.0
    idcg = sum((2 ** r - 1) * max_T / np.log2(i + 2) for i, r in enumerate(ideal_rel))
    
    return dcg / idcg if idcg > 0 else 0.0

def it_err(relevances, topology_factors, k=10, max_grade=3):
    """Intent- and Topology-aware Expected Reciprocal Rank."""
    err = 0.0
    prob_stop = 0.0
    for i in range(min(k, len(relevances))):
        R_i = relevances[i] / max_grade
        T_i = topology_factors[i]
        R_T = R_i * T_i
        err += ((1 - prob_stop) * R_T) / (i + 1)
        prob_stop += (1 - prob_stop) * R_T
    return err

def mrr_topo(relevances, topology_factors):
    """Topology-aware Mean Reciprocal Rank (MRR-Topo)."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            rank = i + 1
            T_i = topology_factors[i]
            return T_i / rank
    return 0.0

def sqi(G, result_indices, query_node_idx, node_to_community, mus, k=10):
    """Subgraph Quality Index."""
    result_set = set(result_indices[:k])
    
    cond = compute_conductance(G, result_set)
    mod = compute_modularity_delta(G, result_set, node_to_community)
    ppr_vals = compute_ppr(G, query_node_idx, result_set)
    ppr_sum = sum(ppr_vals.values())
    
    score = mus["cond"] * (1 - cond) + mus["mod"] * mod + mus["ppr"] * ppr_sum
    return score

def rbo_stability(rank_list1, rank_list2, p=0.9):
    """Rank-Biased Overlap between two rankings."""
    def rbo_score(S, T, p, depth):
        score = 0.0
        for d in range(1, depth + 1):
            set_S = set(S[:d])
            set_T = set(T[:d])
            overlap_size = len(set_S & set_T)
            agreement = overlap_size / d
            score += (p ** (d - 1)) * agreement
        return (1 - p) * score
    
    depth = min(len(rank_list1), len(rank_list2))
    return rbo_score(rank_list1, rank_list2, p, depth)

# ──────────────────────────────────────────────────────────────────────────────
# 10. Retrieval functions
# ──────────────────────────────────────────────────────────────────────────────
def cosine_search(query_idx, k=K_RESULTS):
    """Cosine similarity search in dataset subset."""
    query_vec = query_emb[query_idx]
    similarities = query_vec @ corpus_subset.T
    top_indices = np.argsort(-similarities)[:k]
    return [int(dataset_idx[i]) for i in top_indices], top_indices.tolist()

def lambda_search(query_idx, k=K_RESULTS):
    """Lambda-aware search via ArrowSpace."""
    query_vec = query_emb[query_idx]
    results = aspace.search(query_vec.astype(np.float64), gl, tau=TAU)
    local_indices = [i for i, _ in results][:k]
    return [int(dataset_idx[i]) for i in local_indices], local_indices

# ──────────────────────────────────────────────────────────────────────────────
# 11. Evaluation loop (CORRECTED)
# ──────────────────────────────────────────────────────────────────────────────
results = {
    "cosine": {
        # Topology-aware metrics
        "g_rbp": [], "td_ndcg": [], "it_err": [], "mrr_topo": [], "sqi": [], "rbo_s": [],
        # Energy-informed metrics
        "mrr": [], "map": [], "recall@10": [], "recall@20": [], "ndcg": [],
        "spearman": [], "kendall": []
    },
    "lambda": {
        # Topology-aware metrics
        "g_rbp": [], "td_ndcg": [], "it_err": [], "mrr_topo": [], "sqi": [], "rbo_s": [],
        # Energy-informed metrics
        "mrr": [], "map": [], "recall@10": [], "recall@20": [], "ndcg": [],
        "spearman": [], "kendall": []
    }
}

print(f"\n{'='*85}")
print(f"RUNNING EVALUATION ON {len(eval_queries)} QUERIES")
print(f"{'='*85}\n")

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
    
    # Topology-aware metrics
    results["cosine"]["g_rbp"].append(g_rbp(cosine_rel, cosine_T, RBP_P, K_EVAL))
    results["cosine"]["td_ndcg"].append(td_ndcg(cosine_rel, cosine_T, K_EVAL))
    results["cosine"]["it_err"].append(it_err(cosine_rel, cosine_T, K_EVAL))
    results["cosine"]["mrr_topo"].append(mrr_topo(cosine_rel, cosine_T))
    results["cosine"]["sqi"].append(sqi(G, cosine_local, query_anchor, node_to_community, MU_WEIGHTS, K_EVAL))
    
    # Lambda retrieval
    lambda_global, lambda_local = lambda_search(q_idx, K_RESULTS)
    lambda_rel = [rel_dict.get(doc_id, 0) for doc_id in lambda_global]
    lambda_T = compute_topology_factor(G, query_anchor, lambda_local, node_to_community, LAMBDA_WEIGHTS)
    
    # Topology-aware metrics
    results["lambda"]["g_rbp"].append(g_rbp(lambda_rel, lambda_T, RBP_P, K_EVAL))
    results["lambda"]["td_ndcg"].append(td_ndcg(lambda_rel, lambda_T, K_EVAL))
    results["lambda"]["it_err"].append(it_err(lambda_rel, lambda_T, K_EVAL))
    results["lambda"]["mrr_topo"].append(mrr_topo(lambda_rel, lambda_T))
    results["lambda"]["sqi"].append(sqi(G, lambda_local, query_anchor, node_to_community, MU_WEIGHTS, K_EVAL))
    
    # RBO-Stability (shared between methods)
    rbo_val = rbo_stability(cosine_local, lambda_local, RBP_P)
    results["cosine"]["rbo_s"].append(rbo_val)
    results["lambda"]["rbo_s"].append(rbo_val)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORRECTED: Energy-informed metrics
    # ═══════════════════════════════════════════════════════════════════════════
    # Build relevant set from ground truth relevance scores
    relevant_set_global = set(idx for idx, score in rel_dict.items() if score > 0)
    
    # Map to local indices in dataset_idx
    global_to_local = {int(dataset_idx[i]): i for i in range(len(dataset_idx))}
    relevant_set_local = set(global_to_local[g] for g in relevant_set_global if g in global_to_local)
    
    # For cosine - compute metrics against relevant set
    if len(relevant_set_local) > 0:
        results["cosine"]["mrr"].append(compute_mrr(cosine_local, relevant_set_local))
        results["cosine"]["map"].append(compute_map(cosine_local, relevant_set_local, K_RESULTS))
        results["cosine"]["recall@10"].append(compute_recall_at_k(cosine_local, relevant_set_local, 10))
        results["cosine"]["recall@20"].append(compute_recall_at_k(cosine_local, relevant_set_local, 20))
    else:
        # No relevant documents for this query
        results["cosine"]["mrr"].append(0.0)
        results["cosine"]["map"].append(0.0)
        results["cosine"]["recall@10"].append(0.0)
        results["cosine"]["recall@20"].append(0.0)
    
    # For lambda - compute metrics against relevant set
    if len(relevant_set_local) > 0:
        results["lambda"]["mrr"].append(compute_mrr(lambda_local, relevant_set_local))
        results["lambda"]["map"].append(compute_map(lambda_local, relevant_set_local, K_RESULTS))
        results["lambda"]["recall@10"].append(compute_recall_at_k(lambda_local, relevant_set_local, 10))
        results["lambda"]["recall@20"].append(compute_recall_at_k(lambda_local, relevant_set_local, 20))
    else:
        # No relevant documents for this query
        results["lambda"]["mrr"].append(0.0)
        results["lambda"]["map"].append(0.0)
        results["lambda"]["recall@10"].append(0.0)
        results["lambda"]["recall@20"].append(0.0)
    
    # Ranking correlation (lambda vs cosine)
    spear, kendall = compute_ranking_metrics(lambda_local, cosine_local)
    results["cosine"]["spearman"].append(spear)
    results["cosine"]["kendall"].append(kendall)
    results["lambda"]["spearman"].append(spear)
    results["lambda"]["kendall"].append(kendall)
    
    # NDCG (lambda vs cosine as reference)
    ndcg_val = compute_ndcg_energy(lambda_local, cosine_local, k=K_EVAL)
    results["cosine"]["ndcg"].append(ndcg_val)
    results["lambda"]["ndcg"].append(ndcg_val)
    
    print(f"  Cosine  → G-RBP={results['cosine']['g_rbp'][-1]:.4f} | MRR={results['cosine']['mrr'][-1]:.4f} | MAP={results['cosine']['map'][-1]:.4f}")
    print(f"  Lambda  → G-RBP={results['lambda']['g_rbp'][-1]:.4f} | MRR={results['lambda']['mrr'][-1]:.4f} | MAP={results['lambda']['map'][-1]:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Summary table
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*120}")
print(f"SUMMARY: COMPREHENSIVE METRICS (Mean over {len(eval_queries)} queries)")
print(f"{'='*120}\n")

cosine_means = {metric: np.mean(scores) for metric, scores in results["cosine"].items()}
lambda_means = {metric: np.mean(scores) for metric, scores in results["lambda"].items()}

# Topology-aware metrics
print("TOPOLOGY-AWARE METRICS:")
print(f"{'Method':<25} {'G-RBP':<12} {'TD-nDCG':<12} {'IT-ERR':<12} {'MRR-Topo':<12} {'SQI@k':<12} {'RBO-S':<12}")
print(f"{'-'*120}")
print(f"{'Cosine Similarity':<25} "
      f"{cosine_means['g_rbp']:<12.4f} "
      f"{cosine_means['td_ndcg']:<12.4f} "
      f"{cosine_means['it_err']:<12.4f} "
      f"{cosine_means['mrr_topo']:<12.4f} "
      f"{cosine_means['sqi']:<12.4f} "
      f"{cosine_means['rbo_s']:<12.4f}")
print(f"{'Lambda-aware':<25} "
      f"{lambda_means['g_rbp']:<12.4f} "
      f"{lambda_means['td_ndcg']:<12.4f} "
      f"{lambda_means['it_err']:<12.4f} "
      f"{lambda_means['mrr_topo']:<12.4f} "
      f"{lambda_means['sqi']:<12.4f} "
      f"{lambda_means['rbo_s']:<12.4f}")

# Energy-informed metrics
print("\n\nENERGY-INFORMED METRICS:")
print(f"{'Method':<25} {'MRR':<12} {'MAP':<12} {'Recall@10':<12} {'Recall@20':<12} {'NDCG@10':<12} {'Spearman':<12} {'Kendall':<12}")
print(f"{'-'*120}")
print(f"{'Cosine Similarity':<25} "
      f"{cosine_means['mrr']:<12.4f} "
      f"{cosine_means['map']:<12.4f} "
      f"{cosine_means['recall@10']:<12.4f} "
      f"{cosine_means['recall@20']:<12.4f} "
      f"{cosine_means['ndcg']:<12.4f} "
      f"{cosine_means['spearman']:<12.4f} "
      f"{cosine_means['kendall']:<12.4f}")
print(f"{'Lambda-aware':<25} "
      f"{lambda_means['mrr']:<12.4f} "
      f"{lambda_means['map']:<12.4f} "
      f"{lambda_means['recall@10']:<12.4f} "
      f"{lambda_means['recall@20']:<12.4f} "
      f"{lambda_means['ndcg']:<12.4f} "
      f"{lambda_means['spearman']:<12.4f} "
      f"{lambda_means['kendall']:<12.4f}")
print(f"{'='*120}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 13. Visualization
# ──────────────────────────────────────────────────────────────────────────────
print("Generating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ArrowSpace: Topology-Aware Metrics', fontsize=16, fontweight='bold')

# Topology-aware metrics
topo_metrics = ['g_rbp', 'td_ndcg', 'it_err', 'mrr_topo', 'sqi', 'rbo_s']
topo_labels = ['G-RBP', 'TD-nDCG', 'IT-ERR', 'MRR-Topo', 'SQI@k', 'RBO-S']

for idx, (metric, label) in enumerate(zip(topo_metrics, topo_labels)):
    ax = axes[idx // 3, idx % 3]
    x_pos = np.arange(2)
    values = [cosine_means[metric], lambda_means[metric]]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Cosine', 'Lambda'], fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('test_12_topology_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Topology metrics visualization saved to: test_12_topology_metrics.png")
plt.close()

# Energy-informed metrics
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('ArrowSpace: Energy-Informed Search Metrics Comparison', fontsize=16, fontweight='bold')

energy_metrics = ['mrr', 'map', 'recall@10', 'recall@20', 'ndcg', 'spearman', 'kendall']
energy_labels = ['MRR', 'MAP', 'Recall@10', 'Recall@20', 'NDCG@10', 'Spearman ρ', 'Kendall τ']

for idx, (metric, label) in enumerate(zip(energy_metrics, energy_labels)):
    ax = axes[idx // 4, idx % 4]
    x_pos = np.arange(2)
    values = [cosine_means[metric], lambda_means[metric]]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Cosine', 'Lambda'], fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Hide unused subplot
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('test_12_energy_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Energy metrics visualization saved to: test_12_energy_metrics.png")
plt.close()

# Combined comparison chart
fig, ax = plt.subplots(figsize=(16, 8))
all_metrics = ['G-RBP', 'TD-nDCG', 'MRR-Topo', 'SQI', 'MRR', 'MAP', 'Recall@10', 'NDCG']
metric_keys = ['g_rbp', 'td_ndcg', 'mrr_topo', 'sqi', 'mrr', 'map', 'recall@10', 'ndcg']

x_pos = np.arange(len(all_metrics))
width = 0.35

cosine_vals = [cosine_means[k] for k in metric_keys]
lambda_vals = [lambda_means[k] for k in metric_keys]

bars1 = ax.bar(x_pos - width/2, cosine_vals, width, label='Cosine Similarity',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, lambda_vals, width, label='Lambda-aware (ArrowSpace)',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Comprehensive Metrics Comparison: Cosine vs Lambda-aware Search', fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(all_metrics, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('test_12_combined_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Combined metrics visualization saved to: test_12_combined_metrics.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 14. Save results
# ──────────────────────────────────────────────────────────────────────────────
cosine_means_serializable = {k: float(v) for k, v in cosine_means.items()}
lambda_means_serializable = {k: float(v) for k, v in lambda_means.items()}

improvements = {}
for metric in metric_keys:
    if cosine_means[metric] > 0:
        improvements[metric] = ((lambda_means[metric] - cosine_means[metric]) / cosine_means[metric]) * 100
    else:
        improvements[metric] = 0.0

improvements_serializable = {k: float(v) for k, v in improvements.items()}

results_serializable = {
    method: {
        metric: [float(score) for score in scores]
        for metric, scores in metrics.items()
    }
    for method, metrics in results.items()
}

output = {
    "dataset": "MS MARCO (BeIR)",
    "dataset_size": int(dataset_size),
    "num_queries": len(eval_queries),
    "k_eval": int(K_EVAL),
    "tau": float(TAU),
    "rbp_p": float(RBP_P),
    "lambda_weights": {k: float(v) for k, v in LAMBDA_WEIGHTS.items()},
    "mu_weights": {k: float(v) for k, v in MU_WEIGHTS.items()},
    "batch_sizes": {
        "corpus": int(CORPUS_BATCH_SIZE),
        "query": int(QUERY_BATCH_SIZE)
    },
    "cache_dir": CACHE_DIR,
    "metrics": {
        "cosine": cosine_means_serializable,
        "lambda_aware": lambda_means_serializable,
        "improvements": improvements_serializable
    },
    "per_query": results_serializable
}

with open("test_12_topology_aware_evaluation.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "="*120)
print("EVALUATION COMPLETE")
print("="*120)
print("✓ Results saved to: test_12_topology_aware_evaluation.json")
if STORED:
    print(f"✓ Embeddings cached in: {CACHE_DIR}")
print("✓ Visualizations generated:")
print("  - test_12_topology_metrics.png")
print("  - test_12_energy_metrics.png")
print("  - test_12_combined_metrics.png")
print("="*120)
