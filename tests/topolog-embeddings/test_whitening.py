"""
Tests embedding whitening on CVE dataset using domain_adapted_model.
Stores results using genestore for persistent caching.

Integrates with existing CVE helpers to:
1. Load real CVE data
2. Measure anisotropy reduction (crucial for topological search)
3. Visualize singular value spectrum
4. Cache embeddings using genestore

Usage:
  python test_whitening_cve.py --dataset_root /path/to/cvelistV5 --cache_dir ../../cve_cache
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import genestore  # Import genestore

# Reuse your existing helper functions directly to ensure consistency
def iter_cve_json(root_dir, start=2024, end=2025): # Default to recent for speed
    """Iterate over CVE JSON files in date range."""
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end+1)):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    yield path, json.load(f)
                except Exception:
                    continue

def extract_text(j):
    """Extract searchable text from CVE JSON (Same as your module)."""
    cve_id = j.get("cveMetadata", {}).get("cveId", "")
    cna = j.get("containers", {}).get("cna", {})
    title = cna.get("title", "") or ""
    
    descs = []
    for d in cna.get("descriptions", []) or []:
        if isinstance(d, dict):
            val = d.get("value") or ""
            if val: descs.append(val)
    description = " ".join(descs)
    
    products = []
    for a in cna.get("affected", []) or []:
        vendor = a.get("vendor") or ""
        product = a.get("product") or ""
        if vendor or product:
            products.append(f"{vendor} {product}".strip())
    prod_str = " ".join(products)
    
    text = " | ".join([s for s in [cve_id, title, description, prod_str] if s])
    return cve_id or "(unknown)", title or "(no title)", text

def build_embeddings(texts, model_path="./domain_adapted_model"):
    """Generate embeddings using fine-tuned model."""
    print(f"Loading model from: {model_path}")
    try:
        model = SentenceTransformer(model_path)
    except:
        print(f"Warning: {model_path} not found. Falling back to 'all-mpnet-base-v2'")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
    print(f"Generating embeddings for {len(texts)} texts...")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return X.astype(np.float64)

# ──────────────────────────────────────────────────────────────────────────────
# Whitening Logic
# ──────────────────────────────────────────────────────────────────────────────
class Whitener:
    def __init__(self):
        self.W = None
        self.mu = None

    def fit(self, embeddings):
        """Compute ZCA whitening matrix."""
        self.mu = np.mean(embeddings, axis=0)
        embeddings = embeddings - self.mu
        cov = np.dot(embeddings.T, embeddings) / (embeddings.shape[0] - 1)
        U, S, V = np.linalg.svd(cov)
        epsilon = 1e-7
        self.W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
        return self

    def transform(self, embeddings):
        return np.dot(embeddings - self.mu, self.W)

def calculate_anisotropy(embeddings):
    """Measure how 'cone-like' the embeddings are (Lower is better)."""
    embeddings = embeddings - np.mean(embeddings, axis=0)
    _, S, _ = np.linalg.svd(embeddings)
    return S[0] / np.sum(S)

# ──────────────────────────────────────────────────────────────────────────────
# Main Test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test whitening on CVE embeddings")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to CVE JSON root")
    parser.add_argument("--model_path", type=str, default="./domain_adapted_model", help="Path to model")
    parser.add_argument("--limit", type=int, default=5000, help="Max CVEs to load for test")
    parser.add_argument("--cache_dir", type=str, default="./cve_cache", help="Directory for genestore cache")
    args = parser.parse_args()

    # 1. Initialize Genestore
    print(f"Initializing genestore in {args.cache_dir}...")
    builder = genestore.store_array(args.cache_dir)
    builder.with_max_rows_per_file(500_000)
    builder.with_compression("zstd")
    storage = builder.build()

    # 2. Load Data
    print("Loading CVEs...")
    docs = []
    titles = []
    for _, j in tqdm(iter_cve_json(args.dataset_root)):
        _, t, text = extract_text(j)
        if text:
            docs.append(text)
            titles.append(t)
            if len(docs) >= args.limit:
                break
    
    if not docs:
        print("No CVEs found! Check path.")
        return

    # 3. Generate Embeddings (Original)
    dataset_name_orig = "cve_embeddings_original"
    
    # Try to load from cache first
    try:
        print(f"Attempting to load {dataset_name_orig} from cache...")
        emb_orig = storage.load(dataset_name_orig)
        print(f"✓ Loaded from cache. Shape: {emb_orig.shape}")
        
        # Verify length matches current limit (simple check)
        if len(emb_orig) != len(docs):
             print(f"⚠ Cache size ({len(emb_orig)}) != Limit ({len(docs)}). Recomputing...")
             raise ValueError("Size mismatch")
             
    except Exception as e:
        print(f"Cache miss or mismatch ({e}). Computing embeddings...")
        emb_orig = build_embeddings(docs, args.model_path)
        
        # Store to cache
        print(f"Caching {dataset_name_orig}...")
        try:
            path = storage.store(emb_orig, dataset_name_orig)
            print(f"✓ Stored at: {path}")
        except Exception as store_err:
             print(f"⚠ Failed to store cache: {store_err}")

    # 4. Apply Whitening
    print("Fitting whitening transformation...")
    whitener = Whitener()
    whitener.fit(emb_orig)
    emb_white = whitener.transform(emb_orig)

    # Store Whitened Embeddings
    dataset_name_white = "cve_embeddings_whitened"
    print(f"Caching {dataset_name_white}...")
    try:
        # Ensure float64 and contiguous before storing
        emb_white_store = np.ascontiguousarray(emb_white, dtype=np.float64)
        path = storage.store(emb_white_store, dataset_name_white)
        print(f"✓ Stored at: {path}")
    except Exception as store_err:
        print(f"⚠ Failed to store whitened cache: {store_err}")

    # 5. Metrics & Visualization
    aniso_orig = calculate_anisotropy(emb_orig)
    aniso_white = calculate_anisotropy(emb_white)

    print("\n" + "="*50)
    print("WHITENING RESULTS")
    print("="*50)
    print(f"Anisotropy (Original): {aniso_orig:.4f}")
    print(f"Anisotropy (Whitened): {aniso_white:.4f}")
    print(f"Improvement: {(aniso_orig - aniso_white)/aniso_orig*100:.1f}%")

    # Plot Spectrum
    def get_singular_values(embs):
        embs = embs - np.mean(embs, axis=0)
        _, S, _ = np.linalg.svd(embs)
        return S

    plt.figure(figsize=(10, 6))
    plt.semilogy(get_singular_values(emb_orig), label='Original', alpha=0.7)
    plt.semilogy(get_singular_values(emb_white), label='Whitened', alpha=0.7)
    plt.title(f"Singular Value Spectrum (N={len(docs)})")
    plt.xlabel("Dimension Index")
    plt.ylabel("Singular Value (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("cve_whitening_spectrum.png")
    print("\n✓ Saved spectrum plot to cve_whitening_spectrum.png")

    # 6. Simple Query Test (Consistency Check)
    test_queries = [
        "sql injection vulnerability",
        "remote code execution",
        "buffer overflow in kernel"
    ]
    print("\nRunning consistency check on queries...")
    
    # Embed queries using same model
    try:
        model = SentenceTransformer(args.model_path)
    except:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
    q_emb_orig = model.encode(test_queries)
    q_emb_white = whitener.transform(q_emb_orig)

    for i, q in enumerate(test_queries):
        # Original Top-3
        sims_orig = cosine_similarity([q_emb_orig[i]], emb_orig)[0]
        top_idx_orig = np.argsort(-sims_orig)[:3]
        
        # Whitened Top-3
        sims_white = cosine_similarity([q_emb_white[i]], emb_white)[0]
        top_idx_white = np.argsort(-sims_white)[:3]
        
        print(f"\nQuery: '{q}'")
        print("  Original Top 1:", titles[top_idx_orig[0]])
        print("  Whitened Top 1:", titles[top_idx_white[0]])
        
        # Overlap check
        overlap = len(set(top_idx_orig) & set(top_idx_white))
        print(f"  Top-3 Overlap: {overlap}/3")
        
    # Final Verification Loop (Load back whitened data)
    print("\nVerifying stored data integrity...")
    try:
        loaded_white = storage.load(dataset_name_white)
        if np.allclose(emb_white, loaded_white):
            print("✓ Verification PASSED: Loaded whitened data matches computed data.")
        else:
            print("❌ Verification FAILED: Data mismatch.")
    except Exception as e:
        print(f"❌ Verification FAILED: Could not load data. {e}")

if __name__ == "__main__":
    main()
