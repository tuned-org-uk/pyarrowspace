"""
Tests Parametric UMAP on CVE embeddings for Topological Search.

This method learns a Neural Network projection (Parametric UMAP) on top of 
Transformer embeddings to explicitly optimize for local neighborhood preservation.

Integrates with existing CVE helpers to:
1. Load real CVE data
2. Compute base embeddings (Transformer)
3. Train Parametric UMAP (Topological Projection)
4. Cache results with genestore

Usage:
  python test_parametric_umap.py --dataset_root /path/to/cvelistV5 --cache_dir ./cve_cache
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
import genestore
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP

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
# Parametric UMAP Logic
# ──────────────────────────────────────────────────────────────────────────────
def train_parametric_umap(embeddings, n_components=128, n_neighbors=15, min_dist=0.1):
    """
    Trains a Parametric UMAP neural network to project embeddings into a 
    topologically optimized space.
    
    Args:
        embeddings: Input vectors (e.g. 768-dim from Transformer)
        n_components: Output dimension (can be same as input or lower)
        n_neighbors: Controls local topology preservation (key for ArrowSpace)
        min_dist: Controls how tightly points pack together
        
    Returns:
        model: Trained ParametricUMAP model
        projected_embeddings: The transformed data
    """
    print(f"Training Parametric UMAP (Input: {embeddings.shape[1]}d -> Output: {n_components}d)...")
    
    # We use a custom encoder definition to keep dimensions high if desired, 
    # or let UMAP define a default default MLP.
    # For 'Topology Preserving' re-embedding, we often want to keep dimensions relatively high 
    # (e.g. 128 or even 768) rather than visualizing in 2D.
    
    # Define a custom encoder to ensure we don't compress too hard
    dims = embeddings.shape[1]
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(dims,)),
        tf.keras.layers.Dense(dims // 2, activation="relu"),
        tf.keras.layers.Dense(dims // 2, activation="relu"),
        tf.keras.layers.Dense(n_components, name="z")
    ])

    embedder = ParametricUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine', # ArrowSpace uses cosine/angular logic usually
        encoder=encoder,
        verbose=True,
        batch_size=64, # Larger batch size helps stability
        n_epochs=5,    # Quick training for demo
    )
    
    projected_data = embedder.fit_transform(embeddings)
    return embedder, projected_data

def calculate_trustworthiness(orig_X, new_X, k=15):
    """
    Measures how well the local neighborhood is preserved.
    Scikit-learn implementation of Trustworthiness.
    """
    from sklearn.manifold import trustworthiness
    # We sample if dataset is too large because this is O(N^2)
    n = min(len(orig_X), 1000)
    idx = np.random.choice(len(orig_X), n, replace=False)
    return trustworthiness(orig_X[idx], new_X[idx], n_neighbors=k)

# ──────────────────────────────────────────────────────────────────────────────
# Main Test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test Parametric UMAP on CVE embeddings")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to CVE JSON root")
    parser.add_argument("--model_path", type=str, default="./domain_adapted_model", help="Path to model")
    parser.add_argument("--limit", type=int, default=5000, help="Max CVEs to load for test")
    parser.add_argument("--cache_dir", type=str, default="./cve_cache", help="Directory for genestore cache")
    parser.add_argument("--output_dim", type=int, default=128, help="Dimension for topological embedding")
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
        
        if len(emb_orig) != len(docs):
             print(f"⚠ Cache size mismatch. Recomputing...")
             raise ValueError("Size mismatch")
             
    except Exception as e:
        print(f"Cache miss ({e}). Computing embeddings...")
        emb_orig = build_embeddings(docs, args.model_path)
        try:
            storage.store(emb_orig, dataset_name_orig)
        except Exception as store_err:
             print(f"⚠ Failed to store cache: {store_err}")

    # 4. Train Parametric UMAP (Topological Projection)
    # Note: We don't cache the *model* here (would require saving TF model), 
    # but we cache the *result vectors*.
    
    dataset_name_pumap = f"cve_embeddings_pumap_{args.output_dim}d"
    
    try:
        print(f"Attempting to load {dataset_name_pumap} from cache...")
        emb_pumap = storage.load(dataset_name_pumap)
        print(f"✓ Loaded P-UMAP embeddings from cache.")
        # We need the model object for new queries, but for this test script 
        # we will re-train to demonstrate the process if we were just doing static analysis.
        # In production: Save the 'embedder' object using pickle/tf.save.
        print("(Re-training model to transform test queries...)")
        embedder, _ = train_parametric_umap(emb_orig, n_components=args.output_dim)
        
    except Exception:
        print("Cache miss. Training Parametric UMAP...")
        embedder, emb_pumap = train_parametric_umap(emb_orig, n_components=args.output_dim)
        
        print(f"Caching {dataset_name_pumap}...")
        try:
            # Ensure float64 and contiguous
            emb_pumap_store = np.ascontiguousarray(emb_pumap, dtype=np.float64)
            path = storage.store(emb_pumap_store, dataset_name_pumap)
            print(f"✓ Stored at: {path}")
        except Exception as store_err:
            print(f"⚠ Failed to store P-UMAP cache: {store_err}")

    # 5. Metrics & Visualization
    print("\n" + "="*50)
    print("TOPOLOGICAL ANALYSIS")
    print("="*50)
    
    # Trustworthiness: How well are local neighborhoods preserved?
    # (Higher is better, 1.0 is perfect)
    trust = calculate_trustworthiness(emb_orig, emb_pumap, k=15)
    print(f"Trustworthiness (Neighborhood Preservation): {trust:.4f}")
    
    # Plot Scatter (Use first 2 dims just for a rough look, even if high-dim)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original (PCA proj)")
    from sklearn.decomposition import PCA
    pca_orig = PCA(n_components=2).fit_transform(emb_orig)
    plt.scatter(pca_orig[:, 0], pca_orig[:, 1], s=1, alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Parametric UMAP ({args.output_dim}D proj)")
    # If output is > 2D, PCA it down for viz
    if args.output_dim > 2:
        pca_pumap = PCA(n_components=2).fit_transform(emb_pumap)
        plt.scatter(pca_pumap[:, 0], pca_pumap[:, 1], s=1, alpha=0.5)
    else:
        plt.scatter(emb_pumap[:, 0], emb_pumap[:, 1], s=1, alpha=0.5)
        
    plt.savefig("cve_pumap_viz.png")
    print("\n✓ Saved visualization to cve_pumap_viz.png")

    # 6. Simple Query Test (Inference)
    test_queries = [
        "sql injection vulnerability",
        "remote code execution",
        "buffer overflow in kernel"
    ]
    print("\nRunning inference check on queries...")
    
    # Embed queries using BASE model
    try:
        model = SentenceTransformer(args.model_path)
    except:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    q_emb_orig = model.encode(test_queries)
    
    # Project queries using P-UMAP Network
    q_emb_pumap = embedder.transform(q_emb_orig)

    for i, q in enumerate(test_queries):
        # Original Top-3
        sims_orig = cosine_similarity([q_emb_orig[i]], emb_orig)[0]
        top_idx_orig = np.argsort(-sims_orig)[:3]
        
        # P-UMAP Top-3
        sims_pumap = cosine_similarity([q_emb_pumap[i]], emb_pumap)[0]
        top_idx_pumap = np.argsort(-sims_pumap)[:3]
        
        print(f"\nQuery: '{q}'")
        print("  Original Top 1:", titles[top_idx_orig[0]])
        print("  P-UMAP Top 1:  ", titles[top_idx_pumap[0]])
        
        overlap = len(set(top_idx_orig) & set(top_idx_pumap))
        print(f"  Top-3 Overlap: {overlap}/3")
        
    # Verify Data Integrity
    print("\nVerifying stored data...")
    try:
        loaded = storage.load(dataset_name_pumap)
        # Check first 100 rows to save time on large datasets
        if np.allclose(emb_pumap[:100], loaded[:100]):
             print("✓ Verification PASSED.")
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    main()
