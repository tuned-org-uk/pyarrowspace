# Example: CVE semantic search with pyarrowspace
# Requirements:
#   pip install sentence-transformers numpy
#   pyarrowspace built/installed and importable
# ##################################################
# Need to download the CVE database in a local directory
# Directory structure:
#   dataset/
#     2025/
#       0xxx/
#         CVE-2025-0001.json
####################################################
# python tests/test_2_CVE_db.py --dataset <dataset_dir>
#
import os, json, glob
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from arrowspace import ArrowSpaceBuilder, set_debug

set_debug(True)  # optional: Rust-side debug prints to stderr

START_YEAR = 2020
END_YEAR = 2020

def iter_cve_json(root_dir, start=START_YEAR, end=END_YEAR):
    for path in glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True):
        if any(str(y) in path for y in range(start, end+1)):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    j = json.load(f)
                except Exception:
                    continue
            yield path, j

def extract_text(j):
    # cve id
    cve_id = j.get("cveMetadata", {}).get("cveId", "")
    cna = j.get("containers", {}).get("cna", {})
    title = cna.get("title", "") or ""
    # descriptions (concatenate all lang entries)
    descs = []
    for d in cna.get("descriptions", []) or []:
        if isinstance(d, dict):
            val = d.get("value") or ""
            if val:
                descs.append(val)
    description = " ".join(descs)
    # problemTypes -> CWE IDs
    cwes = []
    for pt in cna.get("problemTypes", []) or []:
        for d in pt.get("descriptions", []) or []:
            cwe = d.get("cweId")
            if cwe:
                cwes.append(cwe)
    cwe_str = " ".join(cwes)
    # metrics -> CVSS vector string if present
    cvss_vec = ""
    for m in cna.get("metrics", []) or []:
        v31 = m.get("cvssV3_1")
        if isinstance(v31, dict):
            vs = v31.get("vectorString")
            if vs:
                cvss_vec = vs
                break
    # affected vendor/product summary
    affected = cna.get("affected", []) or []
    products = []
    for a in affected:
        vendor = a.get("vendor") or ""
        product = a.get("product") or ""
        if vendor or product:
            products.append(f"{vendor} {product}".strip())
    prod_str = " ".join(products)

    text = " | ".join(
        [s for s in [cve_id, title, description, cwe_str, cvss_vec, prod_str] if s]
    )
    return cve_id or "(unknown)", title or "(no title)", text

def build_embeddings(texts, model_name="sentence-transformers/allenai-specter"):
    # model = SentenceTransformer(model_name)
    # X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # print("example item: ", X[0][0:20])
    # return X.astype(np.float64) * 1e5

    from sentence_transformers import SentenceTransformer
    model_path = "./domain_adapted_model"
    model = SentenceTransformer(model_path)
    print(f"Model successfully loaded from: {model_path}")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("example item: ", X[0][0:20])
    return X.astype(np.float64) * 1e1

def main(dataset_root=None, k=10):
    if dataset_root is None: raise ValueError("Dataset root not set")
    # 1) Load CVEs and construct corpus
    ids, titles, docs = [], [], []
    print("Start JSON iteration")
    for _, j in tqdm(iter_cve_json(dataset_root)):
        cve_id, title, text = extract_text(j)
        ids.append(cve_id)
        titles.append(title)
        docs.append(text)
    if not docs:
        raise SystemExit("No CVE JSON files found.")

    # 2) Embed all documents
    print(f"Start Embeddings for {len(ids)} files")
    emb = build_embeddings(docs)  # (N, D)

    # 3) Build ArrowSpace index (lambda-graph)
    graph_params = {
        "eps": 2.5, 
        "k": 10,
        "topk": 10,
        "p": 2.0, 
        "sigma":  1.0
    }

    ALPHA = 0.62

    import time
    print(f"Build space")
    start_time = time.perf_counter()
    try:
        aspace, gl = ArrowSpaceBuilder.build(graph_params, emb)
    except:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")
        raise
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")

    # 4) Example queries
    queries = [
        "authenticated arbitrary file read path traversal",  # inspired by CVE-2025-0001
        "remote code execution in ERP web component",
        "SQL injection in login endpoint",
    ]

    # 5) Encode queries and search (lambda-aware)
    qemb = build_embeddings(queries)
    for qi, q in enumerate(queries):
        print(f"=={qi}==================================")
        results_lambda = aspace.search(qemb[qi], gl, tau=ALPHA)
        results_cosine = aspace.search(qemb[qi], gl, tau=1.0)
        print("\nQuery:", q)
        print("--lambda--------------------------------")
        for rank, (idx, score) in enumerate(results_lambda, 1):
            print(f"{rank:2d}. {ids[idx]}  {titles[idx]}  [score={score:.4f}]")
        print("--cosine--------------------------------")
        for rank, (idx, score) in enumerate(results_cosine, 1):
            print(f"{rank:2d}. {ids[idx]}  {titles[idx]}  [score={score:.4f}]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ArrowSpace CVEs database demo")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    args = parser.parse_args()
    main(dataset_root=args.dataset)