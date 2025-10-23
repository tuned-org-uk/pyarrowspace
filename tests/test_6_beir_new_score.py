"""
Topology-aware IR evaluation on MS MARCO using ArrowSpace.

Computes six graph-aware metrics (G-RBP, TD-nDCG, IT-ERR, SQI@k, MRR-Topo, RBO-S)
on 10 queries and compares cosine baseline vs lambda-aware retrieval.

Dependencies:
  pip install datasets sentence-transformers numpy scikit-learn networkx scipy
  (also requires arrowspace to be built and importable)
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

set_debug(True)  # Optional: Rust debug prints

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TAU = 0.62
SCALING = 1e2
EPS = 10.0
CORPUS_SIZE = 90000
DATASET_SIZE = 10000     # Subset for ArrowSpace
N_EVAL_QUERIES = 10      # Evaluate on 10 queries
K_RESULTS = 20           # Retrieve top-k
K_EVAL = 10              # Evaluate metrics at k=10

# Batching configuration for embeddings
CORPUS_BATCH_SIZE = 256  # Batch size for corpus embedding
QUERY_BATCH_SIZE = 32    # Batch size for query embedding

# Graph parameters
GRAPH_PARAMS = {
    "eps": EPS,
    "k": 25,
    "topk": 15,
    "p": 2.0,
    "sigma": None
}

# Topology metric weights
LAMBDA_WEIGHTS = {"ppr": 0.4, "cond": 0.3, "mod": 0.3}
MU_WEIGHTS = {"cond": 0.4, "mod": 0.3, "ppr": 0.3}

# RBP persistence parameter
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
    # Process corpus
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
    
    # Process queries
    print("Processing queries...")
    for row in queries_ds:
        query_id = row["_id"]
        query_text = row["text"]
        queries.append(query_text)
        query_id_to_idx[query_id] = len(queries) - 1
    
    # Process qrels (graded relevance)
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
    # Fallback format
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
    batch_emb = model.encode(
        batch_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64
    )
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
    batch_emb = model.encode(
        batch_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32
    )
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
# 8. Metric implementations
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
    """
    Topology-aware Mean Reciprocal Rank (MRR-Topo).
    
    MRR-Topo = T_r * (1 / rank_r)
    where r is the rank of the first relevant document,
    and T_r is the topology factor at that rank.
    """
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
# 9. Retrieval functions
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
# 10. Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
results = {
    "cosine": {
        "g_rbp": [],
        "td_ndcg": [],
        "it_err": [],
        "mrr_topo": [],
        "sqi": [],
        "rbo_s": []
    },
    "lambda": {
        "g_rbp": [],
        "td_ndcg": [],
        "it_err": [],
        "mrr_topo": [],
        "sqi": [],
        "rbo_s": []
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
    
    # ── Cosine retrieval ──
    cosine_global, cosine_local = cosine_search(q_idx, K_RESULTS)
    cosine_rel = [rel_dict.get(doc_id, 0) for doc_id in cosine_global]
    cosine_T = compute_topology_factor(G, query_anchor, cosine_local, node_to_community, LAMBDA_WEIGHTS)
    
    results["cosine"]["g_rbp"].append(g_rbp(cosine_rel, cosine_T, RBP_P, K_EVAL))
    results["cosine"]["td_ndcg"].append(td_ndcg(cosine_rel, cosine_T, K_EVAL))
    results["cosine"]["it_err"].append(it_err(cosine_rel, cosine_T, K_EVAL))
    results["cosine"]["mrr_topo"].append(mrr_topo(cosine_rel, cosine_T))
    results["cosine"]["sqi"].append(sqi(G, cosine_local, query_anchor, node_to_community, MU_WEIGHTS, K_EVAL))
    
    # ── Lambda retrieval ──
    lambda_global, lambda_local = lambda_search(q_idx, K_RESULTS)
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
    
    print(f"  Cosine  → G-RBP={results['cosine']['g_rbp'][-1]:.4f} | TD-nDCG={results['cosine']['td_ndcg'][-1]:.4f} | IT-ERR={results['cosine']['it_err'][-1]:.4f} | MRR-Topo={results['cosine']['mrr_topo'][-1]:.4f} | SQI={results['cosine']['sqi'][-1]:.4f}")
    print(f"  Lambda  → G-RBP={results['lambda']['g_rbp'][-1]:.4f} | TD-nDCG={results['lambda']['td_ndcg'][-1]:.4f} | IT-ERR={results['lambda']['it_err'][-1]:.4f} | MRR-Topo={results['lambda']['mrr_topo'][-1]:.4f} | SQI={results['lambda']['sqi'][-1]:.4f}")
    print(f"  RBO-S = {rbo_val:.4f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Summary table
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*95}")
print(f"SUMMARY: TOPOLOGY-AWARE METRICS (Mean over {len(eval_queries)} queries)")
print(f"{'='*95}\n")

print(f"{'Method':<25} {'G-RBP':<12} {'TD-nDCG':<12} {'IT-ERR':<12} {'MRR-Topo':<12} {'SQI@k':<12} {'RBO-S':<12}")
print(f"{'-'*95}")

cosine_means = {metric: np.mean(scores) for metric, scores in results["cosine"].items()}
lambda_means = {metric: np.mean(scores) for metric, scores in results["lambda"].items()}

print(f"{'Cosine Similarity':<25} "
      f"{cosine_means['g_rbp']:<12.4f} "
      f"{cosine_means['td_ndcg']:<12.4f} "
      f"{cosine_means['it_err']:<12.4f} "
      f"{cosine_means['mrr_topo']:<12.4f} "
      f"{cosine_means['sqi']:<12.4f} "
      f"{cosine_means['rbo_s']:<12.4f}")

print(f"{'Lambda-aware (ArrowSpace)':<25} "
      f"{lambda_means['g_rbp']:<12.4f} "
      f"{lambda_means['td_ndcg']:<12.4f} "
      f"{lambda_means['it_err']:<12.4f} "
      f"{lambda_means['mrr_topo']:<12.4f} "
      f"{lambda_means['sqi']:<12.4f} "
      f"{lambda_means['rbo_s']:<12.4f}")

print(f"{'-'*95}")

improvements = {}
for metric in ["g_rbp", "td_ndcg", "it_err", "mrr_topo", "sqi"]:
    if cosine_means[metric] > 0:
        improvements[metric] = ((lambda_means[metric] - cosine_means[metric]) / cosine_means[metric]) * 100
    else:
        improvements[metric] = 0.0

print(f"{'Improvement (%)':<25} "
      f"{improvements['g_rbp']:>+11.2f}% "
      f"{improvements['td_ndcg']:>+11.2f}% "
      f"{improvements['it_err']:>+11.2f}% "
      f"{improvements['mrr_topo']:>+11.2f}% "
      f"{improvements['sqi']:>+11.2f}%")

print(f"{'='*95}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 12. Save results
# ──────────────────────────────────────────────────────────────────────────────
cosine_means_serializable = {k: float(v) for k, v in cosine_means.items()}
lambda_means_serializable = {k: float(v) for k, v in lambda_means.items()}
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
    "metrics": {
        "cosine": cosine_means_serializable,
        "lambda_aware": lambda_means_serializable,
        "improvements": improvements_serializable
    },
    "per_query": results_serializable
}

with open("topology_aware_evaluation.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ Results saved to: topology_aware_evaluation.json")
