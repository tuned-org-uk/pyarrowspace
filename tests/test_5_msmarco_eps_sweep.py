# pip install datasets sentence-transformers numpy scikit-learn beir scipy
# ensure arrowspace is installed/built and importable

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json
import time
from scipy import stats

from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)

# Configuration
TAU = 0.62
SCALING = 1e2
BASE_EPS = 10.0  # Starting point for EPS sweep
EPS_STEPS = 5   # Number of values to test around BASE_EPS

# EPS sweep range: will test [BASE_EPS * 0.5, BASE_EPS, BASE_EPS * 1.5]
EPS_VALUES = [BASE_EPS * factor for factor in [0.5, 1.0, 1.5]]

print(f"EPS sweep values: {EPS_VALUES}")

# 1) Load MS MARCO dataset
try:
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco", split="validation")
    print(f"Loaded MS MARCO: {len(corpus_ds)} passages, {len(queries_ds)} queries")
except Exception as e:
    print(f"Error loading BeIR dataset: {e}")
    ds = load_dataset("ms_marco", "v1.1", split="validation")
    print(f"Loaded MS MARCO validation: {len(ds)} samples")

# 2) Build corpus and relevance mapping
corpus = []
corpus_id_to_idx = {}
queries = []
query_id_to_idx = {}
positives = defaultdict(set)
relevance_scores = defaultdict(lambda: defaultdict(int))

CORPUS_SIZE = 200000
SAMPLE = 200000

if 'corpus_ds' in locals():
    for i, row in enumerate(corpus_ds):
        if i >= CORPUS_SIZE:
            break
        passage_id = row["_id"]
        text = row["title"] + " " + row["text"] if row["title"] else row["text"]
        corpus.append(text)
        corpus_id_to_idx[passage_id] = len(corpus) - 1
    
    for row in queries_ds:
        query_id = row["_id"]
        query_text = row["text"]
        queries.append(query_text)
        query_id_to_idx[query_id] = len(queries) - 1
    
    for row in qrels_ds:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = int(row["score"])
        
        if query_id in query_id_to_idx and corpus_id in corpus_id_to_idx and score > 0:
            q_idx = query_id_to_idx[query_id]
            c_idx = corpus_id_to_idx[corpus_id]
            positives[q_idx].add(c_idx)
            relevance_scores[q_idx][c_idx] = score
else:
    for i, row in enumerate(ds):
        if i >= CORPUS_SIZE:
            break
        query = row["query"]
        passages = row["passages"]
        
        if query not in query_id_to_idx:
            queries.append(query)
            query_id_to_idx[query] = len(queries) - 1
        
        q_idx = query_id_to_idx[query]
        
        for i, passage in enumerate(passages["passage_text"]):
            passage_text = passage
            is_selected = passages.get("is_selected", 0)[i]
            
            if passage_text not in corpus_id_to_idx:
                corpus.append(passage_text)
                corpus_id_to_idx[passage_text] = len(corpus) - 1
            
            c_idx = corpus_id_to_idx[passage_text]
            
            if is_selected == 1:
                positives[q_idx].add(c_idx)
                relevance_scores[q_idx][c_idx] = 1

corpus = np.array(corpus)
queries = np.array(queries)
N_corpus = len(corpus)
N_queries = len(queries)

print(f"Total passages: {N_corpus}")
print(f"Total queries: {N_queries}")
print(f"Queries with positives: {len(positives)}")

# 3) Embed data
model_name = "sentence-transformers/all-mpnet-base-v2"
print(f"Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

print("Embedding corpus...")
batch_size = 512
corpus_emb = []
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
    corpus_emb.append(batch_emb)

corpus_emb = np.vstack(corpus_emb)
print(f"Corpus embeddings shape: {corpus_emb.shape}")

print("Embedding queries...")
query_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)
print(f"Query embeddings shape: {query_emb.shape}")

corpus_emb_norm = corpus_emb
query_emb_norm = query_emb

# 4) Create dataset split
rng = np.random.default_rng(42)

dataset_size = min(SAMPLE, N_corpus)
dataset_idx = rng.choice(N_corpus, size=dataset_size, replace=False)
dataset_idx_set = set(dataset_idx.tolist())

valid_queries = []
for q_idx, relevant_passages in positives.items():
    relevant_in_dataset = [p for p in relevant_passages if p in dataset_idx_set]
    if len(relevant_in_dataset) > 0:
        valid_queries.append(q_idx)

eval_size = min(50, len(valid_queries))
if len(valid_queries) == 0:
    raise RuntimeError("No queries have relevant passages in the dataset subset")

eval_queries = rng.choice(np.array(valid_queries), size=eval_size, replace=False)
print(f"Dataset size: {len(dataset_idx)} | Evaluation queries: {len(eval_queries)}")

# Prepare corpus subset (same for all EPS values)
corpus_subset = corpus_emb[dataset_idx]
corpus_subset_norm = corpus_subset * SCALING

# 5) Build ArrowSpace for each EPS value
print("\n" + "="*80)
print("BUILDING ARROWSPACE WITH EPS SWEEP")
print("="*80)

arrowspaces = {}
build_times = {}
build_status = {}

for eps_value in EPS_VALUES:
    print(f"\n{'='*80}")
    print(f"Testing EPS = {eps_value}")
    print(f"{'='*80}")
    
    graph_params = {
        "eps": eps_value,
        "k": 25,
        "topk": 15,
        "p": 2.0,
        "sigma": None
    }
    
    start_time = time.perf_counter()
    try:
        aspace, gl = ArrowSpaceBuilder.build(graph_params, corpus_subset_norm.astype(np.float64, copy=False))
        end_time = time.perf_counter()
        build_time = end_time - start_time
        
        arrowspaces[eps_value] = (aspace, gl)
        build_times[eps_value] = build_time
        build_status[eps_value] = "SUCCESS"
        
        print(f"✓ Build successful in {build_time:.2f}s")
        
    except Exception as e:
        end_time = time.perf_counter()
        build_time = end_time - start_time
        
        build_times[eps_value] = build_time
        build_status[eps_value] = "FAILED"
        
        print(f"✗ Build failed after {build_time:.2f}s")
        print(f"  Error: {str(e)[:200]}")

print("\n" + "="*80)
print("BUILD SUMMARY")
print("="*80)
print(f"{'EPS':<10} {'Status':<15} {'Build Time (s)':<15}")
print("-"*80)
for eps in EPS_VALUES:
    print(f"{eps:<10.3f} {build_status[eps]:<15} {build_times[eps]:<15.2f}")
print("="*80)

# Filter to only successful builds
successful_eps = [eps for eps in EPS_VALUES if build_status[eps] == "SUCCESS"]

if len(successful_eps) == 0:
    raise RuntimeError("No EPS values resulted in successful ArrowSpace build")

print(f"\n✓ {len(successful_eps)}/{len(EPS_VALUES)} EPS values built successfully")
print(f"  Successful EPS: {successful_eps}")

# 6) Evaluation metrics
def dcg_at_k(relevances, k=10):
    """Calculate Discounted Cumulative Gain at k"""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)

def ndcg_at_k(retrieved_ids, relevance_dict, k=10):
    """Calculate Normalized Discounted Cumulative Gain at k"""
    retrieved_relevances = [relevance_dict.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    dcg = dcg_at_k(retrieved_relevances, k)
    ideal_relevances = sorted(relevance_dict.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """Calculate recall@k"""
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if len(relevant_set) == 0:
        return 0.0
    return len(retrieved_k & relevant_set) / len(relevant_set)

def mrr_score(retrieved_ids, relevant_ids):
    """Calculate Mean Reciprocal Rank"""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def lambda_search(query_idx, aspace, gl, tau):
    """Lambda-aware search"""
    query_vec = query_emb[query_idx]
    results = aspace.search(query_vec.astype(np.float64), gl, tau=tau)
    local_indices = [i for i, _ in results]
    return [int(dataset_idx[i]) for i in local_indices]

# 7) Run evaluation for all successful EPS values
all_results = {}

print("\n" + "="*80)
print("RUNNING EVALUATION FOR ALL SUCCESSFUL EPS VALUES")
print("="*80)

for eps_value in successful_eps:
    print(f"\n{'='*80}")
    print(f"Evaluating EPS = {eps_value} (tau={TAU})")
    print(f"{'='*80}")
    
    aspace, gl = arrowspaces[eps_value]
    
    recalls = []
    mrrs = []
    ndcgs = []
    
    print(f"Running {len(eval_queries)} queries...")
    for i, q_idx in enumerate(eval_queries):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(eval_queries)} queries")
        
        all_relevant = positives[q_idx]
        relevant_in_dataset = [p for p in all_relevant if p in dataset_idx_set]
        
        if len(relevant_in_dataset) == 0:
            continue
        
        rel_dict = relevance_scores[q_idx]
        
        results = lambda_search(q_idx, aspace, gl, tau=TAU)
        recalls.append(recall_at_k(results, relevant_in_dataset, k=10))
        mrrs.append(mrr_score(results, relevant_in_dataset))
        ndcgs.append(ndcg_at_k(results, rel_dict, k=10))
    
    all_results[eps_value] = {
        "eps": eps_value,
        "build_time": build_times[eps_value],
        "recall@10": np.mean(recalls),
        "mrr": np.mean(mrrs),
        "ndcg@10": np.mean(ndcgs),
        "raw_scores": {
            "recalls": recalls,
            "mrrs": mrrs,
            "ndcgs": ndcgs
        }
    }
    
    print(f"\nResults for EPS = {eps_value}:")
    print(f"  Build time: {build_times[eps_value]:.2f}s")
    print(f"  Recall@10:  {np.mean(recalls):.4f}")
    print(f"  MRR:        {np.mean(mrrs):.4f}")
    print(f"  nDCG@10:    {np.mean(ndcgs):.4f}")

# 8) Comparative analysis
print("\n" + "="*80)
print("COMPARATIVE RESULTS - EPS SWEEP")
print("="*80)
print(f"Dataset: MS MARCO (BeIR)")
print(f"Dataset size: {dataset_size}")
print(f"Evaluated queries: {len(eval_queries)}")
print(f"Tau: {TAU}")

print(f"\n{'EPS':<10} {'Build (s)':<12} {'Recall@10':<12} {'MRR':<12} {'nDCG@10':<12}")
print("-"*80)

for eps_value in sorted(successful_eps):
    res = all_results[eps_value]
    print(f"{eps_value:<10.3f} {res['build_time']:<12.2f} {res['recall@10']:<12.4f} {res['mrr']:<12.4f} {res['ndcg@10']:<12.4f}")

print("="*80)

# 9) Best EPS analysis
print(f"\n{'='*80}")
print("BEST EPS VALUE PER METRIC")
print("="*80)

best_recall_eps = max(all_results.items(), key=lambda x: x[1]['recall@10'])
best_mrr_eps = max(all_results.items(), key=lambda x: x[1]['mrr'])
best_ndcg_eps = max(all_results.items(), key=lambda x: x[1]['ndcg@10'])
fastest_build_eps = min(all_results.items(), key=lambda x: x[1]['build_time'])

print(f"Recall@10:   EPS = {best_recall_eps[0]:.3f} → {best_recall_eps[1]['recall@10']:.4f}")
print(f"MRR:         EPS = {best_mrr_eps[0]:.3f} → {best_mrr_eps[1]['mrr']:.4f}")
print(f"nDCG@10:     EPS = {best_ndcg_eps[0]:.3f} → {best_ndcg_eps[1]['ndcg@10']:.4f}")
print(f"Fastest:     EPS = {fastest_build_eps[0]:.3f} → {fastest_build_eps[1]['build_time']:.2f}s")

print("="*80)

# 10) EPS sensitivity analysis
if len(successful_eps) > 1:
    print(f"\n{'='*80}")
    print("EPS SENSITIVITY ANALYSIS")
    print("="*80)
    
    eps_array = np.array(sorted(successful_eps))
    recall_array = np.array([all_results[eps]['recall@10'] for eps in sorted(successful_eps)])
    mrr_array = np.array([all_results[eps]['mrr'] for eps in sorted(successful_eps)])
    ndcg_array = np.array([all_results[eps]['ndcg@10'] for eps in sorted(successful_eps)])
    
    # Calculate variance
    recall_std = np.std(recall_array)
    mrr_std = np.std(mrr_array)
    ndcg_std = np.std(ndcg_array)
    
    print(f"Metric variance across EPS values:")
    print(f"  Recall@10 std: {recall_std:.4f} ({recall_std/np.mean(recall_array)*100:.1f}%)")
    print(f"  MRR std:       {mrr_std:.4f} ({mrr_std/np.mean(mrr_array)*100:.1f}%)")
    print(f"  nDCG@10 std:   {ndcg_std:.4f} ({ndcg_std/np.mean(ndcg_array)*100:.1f}%)")
    
    print(f"\nRecommendation:")
    if recall_std / np.mean(recall_array) < 0.05:
        print("  Low sensitivity: Results are stable across EPS values")
        print(f"  → Use fastest build: EPS = {fastest_build_eps[0]:.3f}")
    else:
        print("  High sensitivity: EPS value significantly affects performance")
        print(f"  → Use best nDCG: EPS = {best_ndcg_eps[0]:.3f}")

print("="*80)

# 11) Save results to JSON
output_results = {
    "dataset": "MS MARCO (BeIR)",
    "dataset_size": dataset_size,
    "num_queries": len(eval_queries),
    "tau": TAU,
    "scaling": SCALING,
    "eps_sweep": {
        "base_eps": BASE_EPS,
        "tested_values": EPS_VALUES,
        "successful_values": successful_eps,
        "failed_values": [eps for eps in EPS_VALUES if build_status[eps] == "FAILED"]
    },
    "results_by_eps": {}
}

for eps_value in EPS_VALUES:
    output_results["results_by_eps"][str(eps_value)] = {
        "eps": eps_value,
        "build_status": build_status[eps_value],
        "build_time": build_times[eps_value]
    }
    
    if eps_value in all_results:
        output_results["results_by_eps"][str(eps_value)]["metrics"] = {
            "recall@10": float(all_results[eps_value]["recall@10"]),
            "mrr": float(all_results[eps_value]["mrr"]),
            "ndcg@10": float(all_results[eps_value]["ndcg@10"])
        }

# Add best values
output_results["best_configs"] = {
    "best_recall": {"eps": best_recall_eps[0], "value": float(best_recall_eps[1]['recall@10'])},
    "best_mrr": {"eps": best_mrr_eps[0], "value": float(best_mrr_eps[1]['mrr'])},
    "best_ndcg": {"eps": best_ndcg_eps[0], "value": float(best_ndcg_eps[1]['ndcg@10'])},
    "fastest_build": {"eps": fastest_build_eps[0], "time": float(fastest_build_eps[1]['build_time'])}
}

with open("beir_eps_sweep_results.json", "w") as f:
    json.dump(output_results, f, indent=2)

print("\n✓ Results saved to: beir_eps_sweep_results.json")
