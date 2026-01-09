"""
Topology-Aware Open Retrieval Evaluation on arXiv(-like) Scientific Papers
Memory-optimized (< ~10GB RAM) + novelty/variety emphasis for RAG.

This experiment modifies a standard baseline-vs-method comparison (cosine vs ArrowSpace)
to explicitly reward *open retrieval*: returning useful additional documents beyond a
cosine nearest-neighbor baseline.

Motivation & literature links (diversity/novelty evaluation):
- Diversified search evaluation aims to reward novelty/coverage and penalize redundancy,
  e.g. alpha-nDCG (Novelty-Biased DCG) and related cascade/intent-aware measures. [Clarke et al.]
  NIST report: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=907308  (see alpha-nDCG)
- Intent-aware diversification metrics such as ERR-IA are defined as expectations across intents
  and are used to evaluate diversification quality (coverage vs redundancy tradeoffs). [Chapelle et al.]
  PDF (overview and ERR-IA discussion): https://sji.soc.uconn.edu/papers/err_ia.pdf
- Diversity metrics are studied axiomatically; alpha-nDCG is a canonical example in that literature.
  Paper: https://arxiv.org/pdf/1805.02334.pdf

Ranking difference / overlap:
- Rank-Biased Overlap (RBO) is a similarity measure for (possibly indefinite/incomplete) rankings.
  Original paper PDF: http://blog.mobile.codalism.com/research/papers/wmz10_tois.pdf

How the "open retrieval" score works:
- Let C be the cosine top-k result set and A be the ArrowSpace top-k result set.
- Extras = `A / C` (ordered as in A).
- We score:
  (1) novelty: |Extras|/k
  (2) extra_quality: precision of Extras under a (weak) relevance heuristic
  (3) extra_diversity: average pairwise cosine distance among Extras' embeddings
  (4) extra_category_coverage: unique categories covered by Extras normalized by |Extras|
- final_open_retrieval_score = weighted combination of the above.
This is intended to highlight how ArrowSpace can provide more varied candidates for RAG,
helping avoid "reasoning/retrieval deadlocks" where a retriever repeatedly returns the same
semantic neighborhood.

Dependencies:
  pip install datasets sentence-transformers numpy scikit-learn networkx scipy matplotlib genestore arrowspace

Usage:
  python arxiv_open_retrieval.py

Notes:
- This is an *offline* evaluation on a heuristic relevance signal (keyword overlap) because most
  arXiv datasets do not provide qrels; replace with proper judgments if available.
"""

from __future__ import annotations

import os
import gc
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from arrowspace import ArrowSpaceBuilder, set_debug
import genestore


# ──────────────────────────────────────────────────────────────────────────────
# Debug
# ──────────────────────────────────────────────────────────────────────────────
set_debug(True)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration - memory optimized
# ──────────────────────────────────────────────────────────────────────────────
TAU = 0.62
SCALING = 1e2
EPS = 5.0

DATASET_SIZE = 10_000
DATASET_SUBSET = 10_000
N_EVAL_QUERIES = 15

K_RESULTS = 20
TAIL_START = 10  # tail emphasis starts at rank 11 (0-indexed: 10)

CORPUS_BATCH_SIZE = 256
QUERY_BATCH_SIZE = 32

CACHE_DIR = "./arxiv_embeddings_cache"
CORPUS_EMB_NAME = "arxiv_corpus_embeddings"
QUERY_EMB_NAME = "arxiv_query_embeddings"

GRAPH_PARAMS = {
    "eps": EPS,
    "k": 25,
    "topk": 15,
    "p": 2.0,
    "sigma": None,
}

# Open retrieval weights (tune depending on what you want to emphasize)
W_NOVELTY = 0.35
W_EXTRA_PRECISION = 0.35
W_EXTRA_DIVERSITY = 0.20
W_EXTRA_COVERAGE = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# arXiv Category Hierarchy (used as a crude "intent/aspect" proxy)
# ──────────────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES = {
    "cs": ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.RO"],
    "math": ["math.CO", "math.NA", "math.OC", "math.PR", "math.ST"],
    "physics": ["physics.comp-ph", "physics.data-an", "physics.bio-ph"],
    "stat": ["stat.ML", "stat.ME", "stat.TH"],
    "q-bio": ["q-bio.QM", "q-bio.NC", "q-bio.BM"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: storage / dataset / text parsing
# ──────────────────────────────────────────────────────────────────────────────
def dataset_exists_in_storage(cache_dir: str, dataset_name: str) -> bool:
    """Check if a Lance dataset exists by looking for the _versions directory."""
    dataset_path = Path(cache_dir) / dataset_name / "_versions"
    return dataset_path.exists() and dataset_path.is_dir()


def extract_arxiv_categories(text: str) -> List[str]:
    """
    Extract arXiv categories from text by substring matching.

    This is intentionally simple: the evaluation goal is not classification accuracy,
    but having a coarse-grained "aspect/intent" proxy to measure variety/coverage.
    """
    cats = []
    tl = text.lower()
    for _, subcats in ARXIV_CATEGORIES.items():
        for c in subcats:
            if c.lower() in tl:
                cats.append(c)
    return cats if cats else ["unknown"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: embedding + caching
# ──────────────────────────────────────────────────────────────────────────────
def init_genestore(cache_dir: str):
    """Initialize genestore storage at cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    builder = genestore.store_array(cache_dir)
    builder.with_max_rows_per_file(500_000)
    builder.with_compression("zstd")
    return builder.build()


def embed_corpus_with_cache(
    storage,
    model: SentenceTransformer,
    corpus: List[str],
    dataset_name: str,
    batch_size: int = CORPUS_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed corpus with genestore caching.

    Uses np.ascontiguousarray(dtype=float64) to avoid ndarray casting issues when calling PyO3.
    """
    if dataset_exists_in_storage(CACHE_DIR, dataset_name):
        try:
            X = storage.load(dataset_name)
            if X.shape[0] == len(corpus):
                print(f"✓ Loaded cached corpus embeddings: {X.shape}")
                return X
            print("⚠ Cache size mismatch, recomputing corpus embeddings...")
        except Exception as e:
            print(f"⚠ Cache load failed, recomputing: {e}")

    print(f"Embedding corpus in batches (batch_size={batch_size})...")
    batches = []
    n_batches = (len(corpus) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, len(corpus))
        if bi % 5 == 0:
            print(f"  Batch {bi+1}/{n_batches}: [{s}:{e}]")
        emb = model.encode(corpus[s:e], convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        emb = np.ascontiguousarray(emb, dtype=np.float64)
        batches.append(emb)
        if bi % 3 == 0:
            gc.collect()

    X = np.vstack(batches)
    del batches
    gc.collect()

    print(f"Caching corpus embeddings as '{dataset_name}'...")
    try:
        storage.store(X, dataset_name)
        print("✓ Cached corpus embeddings")
    except Exception as e:
        print(f"Warning: Failed to cache corpus embeddings: {e}")

    return X


def embed_queries_with_cache(
    storage,
    model: SentenceTransformer,
    queries: List[str],
    dataset_name: str,
) -> np.ndarray:
    """Embed queries with optional caching."""
    if dataset_exists_in_storage(CACHE_DIR, dataset_name):
        try:
            Q = storage.load(dataset_name)
            if Q.shape[0] == len(queries):
                print(f"✓ Loaded cached query embeddings: {Q.shape}")
                return Q
            print("⚠ Cache size mismatch, recomputing query embeddings...")
        except Exception as e:
            print(f"⚠ Cache load failed, recomputing: {e}")

    Q = model.encode(queries, convert_to_numpy=True, show_progress_bar=False, batch_size=QUERY_BATCH_SIZE)
    Q = np.ascontiguousarray(Q, dtype=np.float64)

    print(f"Caching query embeddings as '{dataset_name}'...")
    try:
        storage.store(Q, dataset_name)
        print("✓ Cached query embeddings")
    except Exception as e:
        print(f"Warning: Failed to cache query embeddings: {e}")

    return Q


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: relevance heuristic + novelty/diversity metrics
# ──────────────────────────────────────────────────────────────────────────────
def keyword_overlap_count(query: str, doc: str) -> int:
    """Simple relevance proxy: number of shared whitespace tokens."""
    q = set(query.lower().split())
    d = set(doc.lower().split())
    return len(q & d)


def build_relevance_sets(
    queries: List[str],
    corpus: List[str],
    dataset_idx: np.ndarray,
    min_overlap: int = 3,
) -> List[Set[int]]:
    """
    Build per-query relevant doc sets within the *subset* using keyword overlap.

    This is a weak signal, but it is cheap and allows comparing methods without qrels.
    """
    rel_sets: List[Set[int]] = []
    for qi, q in enumerate(queries):
        qkw = set(q.lower().split())
        rel_local: Set[int] = set()
        for local_i in range(len(dataset_idx)):
            global_i = int(dataset_idx[local_i])
            okw = set(corpus[global_i].lower().split())
            if len(qkw & okw) >= min_overlap:
                rel_local.add(local_i)
        rel_sets.append(rel_local)
        print(f"  Query {qi+1}: {len(rel_local)} relevant (heuristic)")
    return rel_sets


def novelty_extras(order_a: List[int], order_b: List[int]) -> List[int]:
    """Return items in A that are not in B, preserving A order."""
    sb = set(order_b)
    return [x for x in order_a if x not in sb]


def avg_pairwise_cosine_distance(vectors: np.ndarray) -> float:
    """Average pairwise cosine distance among rows. Returns 0 if <2 vectors."""
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    X = vectors / norms
    sim = X @ X.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def category_coverage(local_indices: List[int], local_to_categories: Dict[int, List[str]]) -> int:
    """Number of unique categories covered by a list of local indices."""
    cats = set()
    for li in local_indices:
        for c in local_to_categories.get(li, ["unknown"]):
            cats.add(c)
    return len(cats)


def compute_mrr(results: List[int], relevant_set: Set[int]) -> float:
    """MRR with binary relevance."""
    for rank, idx in enumerate(results, 1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_map(results: List[int], relevant_set: Set[int], k: int = 20) -> float:
    """MAP@k with binary relevance."""
    results_k = results[:k]
    rel_cnt = 0
    p_sum = 0.0
    for rank, idx in enumerate(results_k, 1):
        if idx in relevant_set:
            rel_cnt += 1
            p_sum += rel_cnt / rank
    return p_sum / min(len(relevant_set), k) if rel_cnt > 0 else 0.0


def compute_recall_at_k(results: List[int], relevant_set: Set[int], k: int = 10) -> float:
    """Recall@k with binary relevance."""
    found = sum(1 for idx in results[:k] if idx in relevant_set)
    return found / len(relevant_set) if relevant_set else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval functions
# ──────────────────────────────────────────────────────────────────────────────
def cosine_search(query_vec: np.ndarray, corpus_mat: np.ndarray, k: int) -> List[int]:
    """Cosine top-k using dot product (assumes embeddings are comparable)."""
    sims = query_vec @ corpus_mat.T
    top = np.argsort(-sims)[:k]
    return top.tolist()


def arrowspace_search(query_vec: np.ndarray, aspace, gl, tau: float, k: int) -> List[int]:
    """ArrowSpace search; returns local indices in ranked order."""
    res = aspace.search(query_vec.astype(np.float64), gl, tau=tau)
    return [i for i, _ in res][:k]


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Initializing genestore cache... \n")
    storage = init_genestore(CACHE_DIR)

    # 1) Load dataset
    print(f"Loading arXiv papers (limit={DATASET_SIZE})...")
    dataset = None
    dataset_source = None

    # Option 1
    try:
        print("  Attempting: ccdv/arxiv-classification...")
        dataset = load_dataset(
            "ccdv/arxiv-classification",
            split=f"train[:{DATASET_SIZE}]",
            trust_remote_code=False,
        )
        dataset_source = "ccdv/arxiv-classification"
        print(f"  ✓ Loaded from {dataset_source}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Option 2
    if dataset is None:
        try:
            print("  Attempting: CShorten/ML-ArXiv-Papers...")
            dataset = load_dataset("CShorten/ML-ArXiv-Papers", split=f"train[:{DATASET_SIZE}]")
            dataset_source = "CShorten/ML-ArXiv-Papers"
            print(f"  ✓ Loaded from {dataset_source}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Option 3
    if dataset is None:
        print("  Using Wikipedia fallback...")
        dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{DATASET_SIZE}]")
        dataset_source = "wikipedia"
        print(f"  ✓ Loaded from {dataset_source}")

    print(f"✓ Dataset: {dataset_source}, samples: {len(dataset)}")

    # 2) Extract text
    print("\nExtracting text (memory-efficient)...")
    corpus: List[str] = []
    paper_categories: List[List[str]] = []

    for i, item in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(dataset)} items")

        if dataset_source == "ccdv/arxiv-classification":
            text = item.get("text", "") or item.get("abstract", "")
            cats = [item.get("label", "unknown")]
        elif dataset_source == "CShorten/ML-ArXiv-Papers":
            text = item.get("abstract", "") or item.get("summaries", "")
            cats = extract_arxiv_categories(text)
        else:
            text = (item.get("text", "") or "")[:1000]
            cats = ["general"]

        if len(text) > 2000:
            text = text[:2000]
        if len(text) < 50:
            continue

        corpus.append(text)
        paper_categories.append(cats)

    print(f"\n✓ Corpus: {len(corpus)} documents (kept as Python list)")

    del dataset
    gc.collect()

    # 3) Category summary
    print("\nBuilding category mappings...")
    category_to_papers = defaultdict(list)
    for i, cats in enumerate(paper_categories):
        for c in cats:
            category_to_papers[c].append(i)

    print(f"Found {len(category_to_papers)} categories")
    top_cats = sorted(category_to_papers.keys(), key=lambda x: len(category_to_papers[x]), reverse=True)[:5]
    for c in top_cats:
        print(f"  {c}: {len(category_to_papers[c])} docs")

    # 4) Queries
    CROSS_DOMAIN_QUERIES = [
        "quantum machine learning algorithms for optimization",
        "neural networks for particle physics simulations",
        "graph neural networks with topological data analysis",
        "deep learning for protein structure prediction",
        "variational inference for Bayesian neural networks",
        "stochastic differential equations in quantum mechanics",
        "computational algebraic geometry for machine learning",
        "reinforcement learning with statistical guarantees",
        "tensor networks for quantum simulation",
        "spectral methods for graph learning",
        "topological signal processing on manifolds",
        "causal inference in machine learning",
        "information geometry in statistical physics",
        "geometric mechanics and symplectic integrators",
        "machine learning for genomic sequence analysis",
    ]
    queries = CROSS_DOMAIN_QUERIES[:N_EVAL_QUERIES]
    print(f"\nQueries: {len(queries)}")

    # 5) Embed with caching
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    corpus_emb = embed_corpus_with_cache(storage, model, corpus, CORPUS_EMB_NAME, batch_size=CORPUS_BATCH_SIZE)
    query_emb = embed_queries_with_cache(storage, model, queries, QUERY_EMB_NAME)

    print(f"\n✓ Corpus embeddings: {corpus_emb.shape} (~{corpus_emb.nbytes / 1024**2:.1f} MB)")
    print(f"✓ Query embeddings: {query_emb.shape}")

    # 6) Sample subset + build ArrowSpace
    rng = np.random.default_rng(42)
    dataset_size = min(DATASET_SUBSET, len(corpus))
    dataset_idx = rng.choice(len(corpus), size=dataset_size, replace=False)

    print(f"\nSampling {dataset_size} documents...")
    corpus_subset = corpus_emb[dataset_idx]
    corpus_subset_scaled = corpus_subset * SCALING

    print("Building ArrowSpace...")
    t0 = time.perf_counter()
    aspace, gl = ArrowSpaceBuilder.build_and_store(GRAPH_PARAMS, corpus_subset_scaled.astype(np.float64))
    print(f"✓ ArrowSpace built in {time.perf_counter() - t0:.2f}s")

    # local idx -> categories (for coverage scoring)
    local_to_categories: Dict[int, List[str]] = {}
    for local_i in range(len(dataset_idx)):
        global_i = int(dataset_idx[local_i])
        local_to_categories[local_i] = paper_categories[global_i]

    # 7) Relevance sets via keyword overlap heuristic
    print("\nBuilding relevance sets (heuristic keyword overlap)...")
    query_relevant_papers = build_relevance_sets(queries, corpus, dataset_idx, min_overlap=3)

    # 8) Evaluation
    results = {
        "cosine": {"mrr": [], "map": [], "recall@10": []},
        "arrow": {"mrr": [], "map": [], "recall@10": []},
        "open": {
            "extras_count": [],
            "extras_ratio": [],
            "extras_precision": [],
            "extras_diversity": [],
            "extras_category_coverage": [],
            "tail_extras_count": [],
            "tail_extras_ratio": [],
            "final_open_retrieval_score": [],
        },
    }

    print(f"\n{'='*90}")
    print(f"EVALUATING {len(queries)} QUERIES (k={K_RESULTS}, tail={TAIL_START+1}..{K_RESULTS})")
    print(f"{'='*90}\n")

    for qi, q in enumerate(queries):
        print(f"Query {qi+1}: {q[:70]}...")

        qv = np.ascontiguousarray(query_emb[qi], dtype=np.float64)

        cosine_rank = cosine_search(qv, corpus_subset, k=K_RESULTS)
        arrow_rank = arrowspace_search(qv, aspace, gl, tau=TAU, k=K_RESULTS)

        relevant_set = query_relevant_papers[qi]

        # baseline metrics
        results["cosine"]["mrr"].append(compute_mrr(cosine_rank, relevant_set))
        results["cosine"]["map"].append(compute_map(cosine_rank, relevant_set, k=K_RESULTS))
        results["cosine"]["recall@10"].append(compute_recall_at_k(cosine_rank, relevant_set, k=10))

        results["arrow"]["mrr"].append(compute_mrr(arrow_rank, relevant_set))
        results["arrow"]["map"].append(compute_map(arrow_rank, relevant_set, k=K_RESULTS))
        results["arrow"]["recall@10"].append(compute_recall_at_k(arrow_rank, relevant_set, k=10))

        # open retrieval extras: ArrowSpace-only docs
        extras = novelty_extras(arrow_rank, cosine_rank)
        extras_count = len(extras)
        extras_ratio = extras_count / float(K_RESULTS)

        extras_relevant = sum(1 for x in extras if x in relevant_set)
        extras_precision = (extras_relevant / extras_count) if extras_count else 0.0

        # diversity among extra docs (embedding-space variety proxy)
        if extras_count >= 2:
            extras_vecs = corpus_subset[np.array(extras, dtype=int)]
            extras_div = avg_pairwise_cosine_distance(extras_vecs)
        else:
            extras_div = 0.0

        extras_cov = category_coverage(extras, local_to_categories)
        extras_cov_norm = (extras_cov / extras_count) if extras_count else 0.0

        # tail extras focus (ranks 11..k)
        cosine_tail = cosine_rank[TAIL_START:]
        arrow_tail = arrow_rank[TAIL_START:]
        tail_extras = novelty_extras(arrow_tail, cosine_tail)
        tail_extras_count = len(tail_extras)
        tail_extras_ratio = tail_extras_count / float(max(1, (K_RESULTS - TAIL_START)))

        # final score: "open retrieval" emphasizing novelty + quality + variety
        final_open = (
            W_NOVELTY * extras_ratio +
            W_EXTRA_PRECISION * extras_precision +
            W_EXTRA_DIVERSITY * extras_div +
            W_EXTRA_COVERAGE * extras_cov_norm
        )

        results["open"]["extras_count"].append(extras_count)
        results["open"]["extras_ratio"].append(extras_ratio)
        results["open"]["extras_precision"].append(extras_precision)
        results["open"]["extras_diversity"].append(extras_div)
        results["open"]["extras_category_coverage"].append(extras_cov)
        results["open"]["tail_extras_count"].append(tail_extras_count)
        results["open"]["tail_extras_ratio"].append(tail_extras_ratio)
        results["open"]["final_open_retrieval_score"].append(final_open)

        print(f"  Cosine: MRR={results['cosine']['mrr'][-1]:.3f}, MAP={results['cosine']['map'][-1]:.3f}, R@10={results['cosine']['recall@10'][-1]:.3f}")
        print(f"  Arrow : MRR={results['arrow']['mrr'][-1]:.3f}, MAP={results['arrow']['map'][-1]:.3f}, R@10={results['arrow']['recall@10'][-1]:.3f}")
        print(f"  OPEN  : extras={extras_count}/{K_RESULTS} (tail extras={tail_extras_count}/{K_RESULTS-TAIL_START}), "
              f"extra_prec={extras_precision:.3f}, extra_div={extras_div:.3f}, extra_cov={extras_cov}, score={final_open:.3f}")

        # Optional qualitative print: show a few "extras"
        if extras_count:
            show_n = min(3, extras_count)
            print("  Example ArrowSpace-only docs:")
            for j in range(show_n):
                li = extras[j]
                gi = int(dataset_idx[li])
                cats = local_to_categories.get(li, ["unknown"])
                ov = keyword_overlap_count(q, corpus[gi])
                snippet = corpus[gi][:160].replace("\n", " ")
                print(f"    + extra#{j+1}: overlap={ov}, cats={cats} | {snippet}...")
        print()

    # 9) Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY ({dataset_source})")
    print(f"{'='*90}\n")

    for metric in ["mrr", "map", "recall@10"]:
        c = float(np.mean(results["cosine"][metric]))
        a = float(np.mean(results["arrow"][metric]))
        delta = ((a - c) / c * 100.0) if c > 1e-12 else 0.0
        print(f"{metric.upper():15s}: Cosine={c:.4f}, Arrow={a:.4f}, Δ={delta:+.1f}%")

    open_mean = float(np.mean(results["open"]["final_open_retrieval_score"]))
    extras_ratio_mean = float(np.mean(results["open"]["extras_ratio"]))
    tail_extras_ratio_mean = float(np.mean(results["open"]["tail_extras_ratio"]))
    print()
    print(f"{'OPEN_SCORE':15s}: mean={open_mean:.4f}")
    print(f"{'EXTRAS_RATIO':15s}: mean={extras_ratio_mean:.4f}")
    print(f"{'TAIL_EXTRAS':15s}: mean={tail_extras_ratio_mean:.4f}")

    # 10) Save CSV
    csv_path = "arxiv_open_retrieval_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query_idx", "query_text",
            "cosine_mrr", "arrow_mrr",
            "cosine_map", "arrow_map",
            "cosine_recall10", "arrow_recall10",
            "extras_count", "extras_ratio", "extras_precision",
            "extras_diversity", "extras_category_coverage",
            "tail_extras_count", "tail_extras_ratio",
            "final_open_retrieval_score",
        ])
        for i, q in enumerate(queries):
            w.writerow([
                i, q,
                results["cosine"]["mrr"][i], results["arrow"]["mrr"][i],
                results["cosine"]["map"][i], results["arrow"]["map"][i],
                results["cosine"]["recall@10"][i], results["arrow"]["recall@10"][i],
                results["open"]["extras_count"][i],
                results["open"]["extras_ratio"][i],
                results["open"]["extras_precision"][i],
                results["open"]["extras_diversity"][i],
                results["open"]["extras_category_coverage"][i],
                results["open"]["tail_extras_count"][i],
                results["open"]["tail_extras_ratio"][i],
                results["open"]["final_open_retrieval_score"][i],
            ])
    print(f"\n✓ Saved CSV: {csv_path}")

    # 11) Visualizations
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))

    # Baseline metrics
    for j, (mkey, title) in enumerate([("mrr", "MRR"), ("map", "MAP"), ("recall@10", "Recall@10")]):
        c = float(np.mean(results["cosine"][mkey]))
        a = float(np.mean(results["arrow"][mkey]))
        ax[j].bar([0, 1], [c, a], color=["#3498db", "#e74c3c"], alpha=0.85)
        ax[j].set_xticks([0, 1])
        ax[j].set_xticklabels(["Cosine", "Arrow"])
        ax[j].set_title(title)
        ax[j].grid(axis="y", alpha=0.3)

    # Open retrieval score distribution
    ax[3].hist(results["open"]["final_open_retrieval_score"], bins=25, color="#8e44ad", alpha=0.85)
    ax[3].set_title("OPEN score (distribution)")
    ax[3].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = "arxiv_open_retrieval_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved plot: {plot_path}")

    print(f"\n{'='*90}")
    print("✓ COMPLETE")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
