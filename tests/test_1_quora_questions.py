# pip install datasets sentence-transformers numpy scikit-learn
# ensure arrowspace is installed/built and importable

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict

from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)  # optional: Rust-side debug prints to stderr

ALPHA = 0.62
SAMPLE = 30000

# 1) Load Quora Duplicate Questions (pair-class: sentence1, sentence2, label)
ds = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")

# 2) Build a corpus of unique questions and a ground-truth mapping of duplicates
corpus = []
qid = {}
positives = defaultdict(set)  # global_id -> set of duplicate global_ids

def get_id(q):
    if q not in qid:
        qid[q] = len(corpus)
        corpus.append(q)
    return qid[q]

for row in ds:
    s1, s2, lab = row["sentence1"], row["sentence2"], int(row["label"])
    id1, id2 = get_id(s1), get_id(s2)
    if lab == 1:
        positives[id1].add(id2)
        positives[id2].add(id1)

corpus = np.array(corpus)
N = len(corpus)
print(f"Total unique questions: {N}")

# 3) Embed all questions once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

# # Optional: L2-normalize for cosine baseline (we will compare query vs dataset subset only)
emb_norm = emb # emb_norm = normalize(emb, norm="l2", axis=1)

# 4) Random split: 10k dataset (index) and 1k queries (disjoint), ensuring queries have in-dataset positives
rng = np.random.default_rng(42)

dataset_size = min(SAMPLE, N)
dataset_idx = rng.choice(N, size=dataset_size, replace=False)
dataset_idx_set = set(dataset_idx.tolist())

# Candidate queries: items not in dataset with at least one duplicate inside dataset
query_pool = []
for g_id, pos in positives.items():
    if g_id in dataset_idx_set:
        continue
    # filter positives to those in dataset
    gold_in_dataset = [p for p in pos if p in dataset_idx_set]
    if len(gold_in_dataset) > 0:
        query_pool.append(g_id)

if len(query_pool) == 0:
    raise RuntimeError("No queries have duplicates inside the 10k dataset subset; try a different random seed or larger dataset subset.")

query_size = min(1000, len(query_pool))
queries = rng.choice(np.array(query_pool), size=query_size, replace=False)
print(f"Dataset size: {len(dataset_idx)} | Query count: {len(queries)} (eligible with in-dataset positives)")

# 5) Build ArrowSpace on the dataset subset only
emb_ds = emb[dataset_idx]
emb_ds_norm = emb_ds * 100 # remove normalisation: normalize(emb_ds, norm="l2", axis=1)

graph_params = {
    "eps": 0.5,
    "k": 4,
    "topk": 2,
    "p": 2.0,
    "sigma": 0.25,
}

print("item example normalised", emb_ds_norm[0])
import time
print(f"Building space")
start_time = time.perf_counter()
aspace, gl = ArrowSpaceBuilder.build(graph_params, emb_ds_norm.astype(np.float64, copy=False))
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")
print("ArrowSpace built successfully!")

# 6) Evaluation: Recall@10 for cosine vs lambda-aware (restricted to dataset subset)
def recall_at_k(hits_global_ids, gold_set_global, gl):
    return 1.0 if any(h in gold_set_global for h in hits_global_ids[:gl.graph_params["topk"]]) else 0.0

def cosine_topk_global_from_dataset(query_global_id, gl):
    # cosine between query and indexed dataset subset only
    sims = emb_norm[query_global_id] @ emb_ds_norm.T
    idx_local = np.argsort(-sims)[:gl.graph_params["topk"]]
    # map dataset-local indices back to global ids
    return [int(dataset_idx[i]) for i in idx_local]

def lambda_aware_topk_global_from_dataset(query_vec, gl):
    # lambda-aware search against dataset subset index; returns local indices
    results = aspace.search(query_vec.astype(np.float64), gl=gl, tau=ALPHA)
    idx_local = [i for i, _ in results]
    return [int(dataset_idx[i]) for i in idx_local]

r10_cos, r10_lam = [], []

for q in queries:
    # restrict gold to duplicates present in dataset subset
    gold = positives[q]
    gold_in_dataset = set([g for g in gold if g in dataset_idx_set])
    if len(gold_in_dataset) == 0:
        # skip queries that (after split) have no in-dataset duplicates
        continue

    # Cosine baseline
    topk_cos_global = cosine_topk_global_from_dataset(q, gl)
    # exclude the query itself if it accidentally appears (shouldn't due to disjoint split)
    topk_cos_global = [g for g in topk_cos_global if g != q][:10]
    r10_cos.append(recall_at_k(topk_cos_global, gold_in_dataset, gl))

    # Lambda-aware
    topk_lam_global = lambda_aware_topk_global_from_dataset(emb[q], gl)
    topk_lam_global = [g for g in topk_lam_global if g != q][:10]
    r10_lam.append(recall_at_k(topk_lam_global, gold_in_dataset, gl))

print(f"Queries evaluated (with in-dataset positives): {len(r10_cos)}")
print(f"Recall@10 (cosine on dataset):        {np.mean(r10_cos):.4f}")
print(f"Recall@10 (lambda-aware on dataset):  {np.mean(r10_lam):.4f}")
