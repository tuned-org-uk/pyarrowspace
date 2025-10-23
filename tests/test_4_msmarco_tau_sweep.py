# pip install datasets sentence-transformers numpy scikit-learn beir scipy
# ensure arrowspace is installed/built and importable

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import json
import time
from scipy import stats

from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)  # optional: Rust-side debug prints to stderr

# Tau sweep configuration
TAU_MODES = {
    "cosine": 1.0,
    "taumode": 0.62,
    "extra-taumode": 0.51
}

SCALING = 1e2
EPS = 10.0

# 1) Load MS MARCO dataset from BEIR
try:
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco", split="validation")
    print(f"Loaded MS MARCO: {len(corpus_ds)} passages, {len(queries_ds)} queries")
except Exception as e:
    print(f"Error loading BeIR dataset: {e}")
    print("Falling back to huggingface MS MARCO dataset...")
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

# Process corpus
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

# 5) Build ArrowSpace
corpus_subset = corpus_emb[dataset_idx]
corpus_subset_norm = corpus_subset * SCALING

graph_params = {
    "eps": EPS,
    "k": 25,
    "topk": 15,
    "p": 2.0,
    "sigma": None
}

print("Building ArrowSpace on corpus subset...")
print(f"Sample embedding shape: {corpus_subset_norm[0].shape}")

start_time = time.perf_counter()
try:
    aspace, gl = ArrowSpaceBuilder.build(graph_params, corpus_subset_norm.astype(np.float64, copy=False))
finally:
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"ArrowSpace build time: {elapsed_time:.6f} seconds")

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

def lambda_search(query_idx, gl, tau):
    """Lambda-aware search with specified tau"""
    query_vec = query_emb[query_idx]
    results = aspace.search(query_vec.astype(np.float64), gl, tau=tau)
    local_indices = [i for i, _ in results]
    return [int(dataset_idx[i]) for i in local_indices]

# 7) Run evaluation for all tau modes
all_results = {}

print("\n" + "="*80)
print("RUNNING TAU SWEEP EVALUATION")
print("="*80)

for mode_name, tau_value in TAU_MODES.items():
    print(f"\n{'='*80}")
    print(f"Evaluating mode: {mode_name.upper()} (tau={tau_value})")
    print(f"{'='*80}")
    
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
        
        # Search with current tau
        results = lambda_search(q_idx, gl, tau=tau_value)
        recalls.append(recall_at_k(results, relevant_in_dataset, k=10))
        mrrs.append(mrr_score(results, relevant_in_dataset))
        ndcgs.append(ndcg_at_k(results, rel_dict, k=10))
    
    # Store results
    all_results[mode_name] = {
        "tau": tau_value,
        "recall@10": np.mean(recalls),
        "mrr": np.mean(mrrs),
        "ndcg@10": np.mean(ndcgs),
        "raw_scores": {
            "recalls": recalls,
            "mrrs": mrrs,
            "ndcgs": ndcgs
        }
    }
    
    print(f"\nResults for {mode_name} (tau={tau_value}):")
    print(f"  Recall@10: {np.mean(recalls):.4f}")
    print(f"  MRR:       {np.mean(mrrs):.4f}")
    print(f"  nDCG@10:   {np.mean(ndcgs):.4f}")

# 8) Comparative analysis
print("\n" + "="*80)
print("COMPARATIVE RESULTS - TAU SWEEP")
print("="*80)
print(f"Dataset: MS MARCO (BeIR)")
print(f"Dataset size: {dataset_size}")
print(f"Evaluated queries: {len(eval_queries)}")

print(f"\n{'Mode':<20} {'Tau':<8} {'Recall@10':<12} {'MRR':<12} {'nDCG@10':<12}")
print("-"*80)

for mode_name in ["cosine", "taumode", "extra-taumode"]:
    res = all_results[mode_name]
    print(f"{mode_name:<20} {res['tau']:<8.2f} {res['recall@10']:<12.4f} {res['mrr']:<12.4f} {res['ndcg@10']:<12.4f}")

print("="*80)

# 9) Relative improvements (compared to cosine baseline)
print(f"\n{'='*80}")
print("RELATIVE IMPROVEMENTS vs COSINE BASELINE (tau=1.0)")
print("="*80)

baseline = all_results["cosine"]
print(f"\n{'Mode':<20} {'Tau':<8} {'Recall@10':<15} {'MRR':<15} {'nDCG@10':<15}")
print("-"*80)

for mode_name in ["taumode", "extra-taumode"]:
    res = all_results[mode_name]
    
    recall_imp = (res['recall@10'] - baseline['recall@10']) / baseline['recall@10'] * 100
    mrr_imp = (res['mrr'] - baseline['mrr']) / baseline['mrr'] * 100
    ndcg_imp = (res['ndcg@10'] - baseline['ndcg@10']) / baseline['ndcg@10'] * 100
    
    print(f"{mode_name:<20} {res['tau']:<8.2f} {recall_imp:>+14.2f}% {mrr_imp:>+14.2f}% {ndcg_imp:>+14.2f}%")

print("="*80)

# 10) Statistical significance tests
print(f"\n{'='*80}")
print("STATISTICAL SIGNIFICANCE (paired t-test vs cosine)")
print("="*80)

baseline_recalls = all_results["cosine"]["raw_scores"]["recalls"]
baseline_mrrs = all_results["cosine"]["raw_scores"]["mrrs"]
baseline_ndcgs = all_results["cosine"]["raw_scores"]["ndcgs"]

for mode_name in ["taumode", "extra-taumode"]:
    print(f"\n{mode_name.upper()} (tau={TAU_MODES[mode_name]}) vs COSINE:")
    print("-"*60)
    
    mode_recalls = all_results[mode_name]["raw_scores"]["recalls"]
    mode_mrrs = all_results[mode_name]["raw_scores"]["mrrs"]
    mode_ndcgs = all_results[mode_name]["raw_scores"]["ndcgs"]
    
    # Recall@10
    t_stat, p_value = stats.ttest_rel(mode_recalls, baseline_recalls)
    sig = "✓ Yes" if p_value < 0.05 and t_stat > 0 else "✗ No"
    print(f"Recall@10:  t={t_stat:>7.4f}, p={p_value:.6f}, Significant: {sig}")
    
    # MRR
    t_stat, p_value = stats.ttest_rel(mode_mrrs, baseline_mrrs)
    sig = "✓ Yes" if p_value < 0.05 and t_stat > 0 else "✗ No"
    print(f"MRR:        t={t_stat:>7.4f}, p={p_value:.6f}, Significant: {sig}")
    
    # nDCG@10
    t_stat, p_value = stats.ttest_rel(mode_ndcgs, baseline_ndcgs)
    sig = "✓ Yes" if p_value < 0.05 and t_stat > 0 else "✗ No"
    print(f"nDCG@10:    t={t_stat:>7.4f}, p={p_value:.6f}, Significant: {sig}")

print("\n" + "="*80)

# 11) Best tau mode analysis
print(f"\n{'='*80}")
print("BEST TAU MODE PER METRIC")
print("="*80)

best_recall_mode = max(all_results.items(), key=lambda x: x[1]['recall@10'])
best_mrr_mode = max(all_results.items(), key=lambda x: x[1]['mrr'])
best_ndcg_mode = max(all_results.items(), key=lambda x: x[1]['ndcg@10'])

print(f"Recall@10: {best_recall_mode[0]} (tau={best_recall_mode[1]['tau']}) = {best_recall_mode[1]['recall@10']:.4f}")
print(f"MRR:       {best_mrr_mode[0]} (tau={best_mrr_mode[1]['tau']}) = {best_mrr_mode[1]['mrr']:.4f}")
print(f"nDCG@10:   {best_ndcg_mode[0]} (tau={best_ndcg_mode[1]['tau']}) = {best_ndcg_mode[1]['ndcg@10']:.4f}")

print("="*80)

# 12) Save results to JSON
output_results = {
    "dataset": "MS MARCO (BeIR)",
    "dataset_size": dataset_size,
    "num_queries": len(eval_queries),
    "graph_params": graph_params,
    "tau_modes": TAU_MODES,
    "results_by_mode": {}
}

for mode_name, res in all_results.items():
    output_results["results_by_mode"][mode_name] = {
        "tau": res["tau"],
        "metrics": {
            "recall@10": float(res["recall@10"]),
            "mrr": float(res["mrr"]),
            "ndcg@10": float(res["ndcg@10"])
        }
    }
    
    # Add improvements vs baseline
    if mode_name != "cosine":
        baseline = all_results["cosine"]
        output_results["results_by_mode"][mode_name]["improvements_vs_cosine"] = {
            "recall@10_pct": float((res['recall@10'] - baseline['recall@10']) / baseline['recall@10'] * 100),
            "mrr_pct": float((res['mrr'] - baseline['mrr']) / baseline['mrr'] * 100),
            "ndcg@10_pct": float((res['ndcg@10'] - baseline['ndcg@10']) / baseline['ndcg@10'] * 100)
        }

with open("beir_tau_sweep_results.json", "w") as f:
    json.dump(output_results, f, indent=2)

print("\n✓ Results saved to: beir_tau_sweep_results.json")
