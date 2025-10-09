# pip install datasets sentence-transformers numpy scikit-learn beir
# ensure arrowspace is installed/built and importable

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import json

from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)  # optional: Rust-side debug prints to stderr

ALPHA = 0.6

# 1) Load MS MARCO dataset from BEIR
try:
    # Load MS MARCO passage ranking dataset
    corpus_ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco", split="validation")  # validation split for qrels
    
    print(f"Loaded MS MARCO: {len(corpus_ds)} passages, {len(queries_ds)} queries")
    
except Exception as e:
    print(f"Error loading BeIR dataset: {e}")
    print("Falling back to huggingface MS MARCO dataset...")
    # Alternative: use the HuggingFace MS MARCO dataset
    ds = load_dataset("ms_marco", "v1.1", split="validation")
    print(f"Loaded MS MARCO validation: {len(ds)} samples")

# 2) Build corpus and relevance mapping for MS MARCO
corpus = []
corpus_id_to_idx = {}
queries = []
query_id_to_idx = {}
positives = defaultdict(set)  # query_idx -> set of relevant passage indices

CORPUS_SIZE = 1024
SAMPLE = 1024

# Process corpus (passages)
if 'corpus_ds' in locals():
    # Using BeIR format
    for i, row in enumerate(corpus_ds):
        if i >= CORPUS_SIZE:  # Limit corpus size for memory efficiency
            break
        passage_id = row["_id"]
        text = row["title"] + " " + row["text"] if row["title"] else row["text"]
        corpus.append(text)
        corpus_id_to_idx[passage_id] = len(corpus) - 1
    
    # Process queries
    for row in queries_ds:
        query_id = row["_id"]
        query_text = row["text"]
        queries.append(query_text)
        query_id_to_idx[query_id] = len(queries) - 1
    
    # Process relevance judgments (qrels)
    for row in qrels_ds:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = int(row["score"])
        
        if query_id in query_id_to_idx and corpus_id in corpus_id_to_idx and score > 0:
            q_idx = query_id_to_idx[query_id]
            c_idx = corpus_id_to_idx[corpus_id]
            positives[q_idx].add(c_idx)

else:
    # Using alternative MS MARCO format
    for i, row in enumerate(ds):
        if i >= CORPUS_SIZE:  # Limit for memory
            break
        
        query = row["query"]
        passages = row["passages"]
        
        # Add query
        if query not in query_id_to_idx:
            queries.append(query)
            query_id_to_idx[query] = len(queries) - 1
        
        q_idx = query_id_to_idx[query]
        
        # Add passages and mark relevant ones
        for i, passage in enumerate(passages["passage_text"]):
            passage_text = passage
            is_selected = passages.get("is_selected", 0)[i]
            
            if passage_text not in corpus_id_to_idx:
                corpus.append(passage_text)
                corpus_id_to_idx[passage_text] = len(corpus) - 1
            
            c_idx = corpus_id_to_idx[passage_text]
            
            if is_selected == 1:  # Relevant passage
                positives[q_idx].add(c_idx)

corpus = np.array(corpus)
queries = np.array(queries)
N_corpus = len(corpus)
N_queries = len(queries)

print(f"Total passages: {N_corpus}")
print(f"Total queries: {N_queries}")
print(f"Queries with positives: {len(positives)}")

# 3) Use a more suitable embedding model for MS MARCO
# BGE models are state-of-the-art for retrieval tasks
model_name = "sentence-transformers/all-mpnet-base-v2" # "BAAI/bge-base-en-v1.5"
print(f"Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

# Embed corpus (in batches to manage memory)
print("Embedding corpus...")
batch_size = 512
corpus_emb = []
for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
    corpus_emb.append(batch_emb)

corpus_emb = np.vstack(corpus_emb)
print(f"Corpus embeddings shape: {corpus_emb.shape}")

# Embed queries
print("Embedding queries...")
query_emb = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)
print(f"Query embeddings shape: {query_emb.shape}")

# Optional: L2-normalize for cosine similarity
corpus_emb_norm = corpus_emb # normalize(corpus_emb, norm="l2", axis=1)
query_emb_norm = query_emb # normalize(query_emb, norm="l2", axis=1)

# 4) Create dataset split: use subset of corpus as index, queries for evaluation
rng = np.random.default_rng(42)

# Use a manageable dataset size
dataset_size = min(SAMPLE, N_corpus)
dataset_idx = rng.choice(N_corpus, size=dataset_size, replace=False)
dataset_idx_set = set(dataset_idx.tolist())

# Select queries that have relevant passages in the dataset subset
valid_queries = []
for q_idx, relevant_passages in positives.items():
    # Check if any relevant passages are in the dataset subset
    relevant_in_dataset = [p for p in relevant_passages if p in dataset_idx_set]
    if len(relevant_in_dataset) > 0:
        valid_queries.append(q_idx)

# Sample evaluation queries
eval_size = min(50, len(valid_queries))
if len(valid_queries) == 0:
    raise RuntimeError("No queries have relevant passages in the dataset subset")

eval_queries = rng.choice(np.array(valid_queries), size=eval_size, replace=False)
print(f"Dataset size: {len(dataset_idx)} | Evaluation queries: {len(eval_queries)}")

# 5) Build ArrowSpace on the dataset subset
corpus_subset = corpus_emb[dataset_idx]
corpus_subset_norm = corpus_subset * 1e2 # normalize(corpus_subset, norm="l2", axis=1)

# Graph parameters optimized for high-dimensional embeddings
graph_params = {
    "eps": 0.1,  # Higher threshold for dense embeddings
    "k": 8,      # More neighbors for better connectivity
    "topk": 3,
    "p": 2.0,
    "sigma": 0.1 * 0.5,
}

print("Building ArrowSpace on corpus subset...")
print(f"Sample embedding shape: {corpus_subset_norm[0].shape}")
print(f"Sample normalized embedding: {corpus_subset_norm[0][:5]}")

import time
print(f"Build space")
start_time = time.perf_counter()
try:
    aspace, gl = ArrowSpaceBuilder.build(graph_params, corpus_subset_norm.astype(np.float64, copy=False))
except:
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    raise
finally:
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

# 6) Evaluation: Recall@10 and MRR for cosine vs lambda-aware retrieval
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

def cosine_search(query_idx, k=10):
    """Cosine similarity search in dataset subset"""
    query_vec = query_emb_norm[query_idx]
    similarities = query_vec @ corpus_subset_norm.T
    top_indices = np.argsort(-similarities)[:k]
    # Map back to global corpus indices
    return [int(dataset_idx[i]) for i in top_indices]

def lambda_search(query_idx, gl):
    """Lambda-aware search in dataset subset"""
    query_vec = query_emb[query_idx]
    results = aspace.search(query_vec.astype(np.float64), gl, tau=ALPHA)
    local_indices = [i for i, _ in results]
    # Map back to global corpus indices
    return [int(dataset_idx[i]) for i in local_indices]

# Run evaluation
cosine_recalls = []
lambda_recalls = []
cosine_mrrs = []
lambda_mrrs = []

print("Running evaluation...")
for i, q_idx in enumerate(eval_queries):
    if i % 100 == 0:
        print(f"Processed {i}/{len(eval_queries)} queries")
    
    # Get relevant passages in dataset subset
    all_relevant = positives[q_idx]
    relevant_in_dataset = [p for p in all_relevant if p in dataset_idx_set]
    
    if len(relevant_in_dataset) == 0:
        continue
    
    # Cosine similarity retrieval
    cosine_results = cosine_search(q_idx, k=20)
    cosine_recalls.append(recall_at_k(cosine_results, relevant_in_dataset, k=10))
    cosine_mrrs.append(mrr_score(cosine_results, relevant_in_dataset))
    
    # Lambda-aware retrieval
    lambda_results = lambda_search(q_idx, gl)
    lambda_recalls.append(recall_at_k(lambda_results, relevant_in_dataset, k=10))
    lambda_mrrs.append(mrr_score(lambda_results, relevant_in_dataset))

# Print results
print(f"\n=== EVALUATION RESULTS ===")
print(f"Evaluated queries: {len(cosine_recalls)}")
print(f"Cosine Similarity:")
print(f"  Recall@10: {np.mean(cosine_recalls):.4f}")
print(f"  MRR:       {np.mean(cosine_mrrs):.4f}")
print(f"Lambda-aware Search:")
print(f"  Recall@10: {np.mean(lambda_recalls):.4f}")
print(f"  MRR:       {np.mean(lambda_mrrs):.4f}")

# Statistical significance test (optional)
from scipy import stats
if len(cosine_recalls) > 30:  # Sufficient samples for t-test
    t_stat, p_value = stats.ttest_rel(lambda_recalls, cosine_recalls)
    print(f"\nStatistical significance (paired t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant improvement: {'Yes' if p_value < 0.05 and t_stat > 0 else 'No'}")
