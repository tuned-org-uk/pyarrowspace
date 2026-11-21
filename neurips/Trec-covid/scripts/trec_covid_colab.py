#!/usr/bin/env python3
"""Build a provenance-bound MiniLM task bundle and evaluate TREC-COVID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


SOURCE_URL = "https://github.com/tuned-org-uk/pyarrowspace.git"
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
DATA_MD5 = "ce62140cb23feb9becf6270d0d1fe6d1"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_DOCUMENTS = 171_332
EXPECTED_QUERIES = 50
SEED = 3407
TAU_VALUES = (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0)
K = 10
SCALE = 100.0
GRAPH_PARAMS = {"eps": 3.9686, "k": 30, "topk": K, "p": 2.0, "sigma": None}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_task_inputs(
    corpus_ids: list[str], query_ids: list[str], qrels: dict[str, dict[str, int]]
) -> int:
    if not corpus_ids or not query_ids:
        raise ValueError("Corpus and query IDs must be nonempty")
    if len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Corpus and query IDs must be unique")
    if set(qrels) != set(query_ids):
        raise ValueError("Qrels must contain exactly the task query IDs")
    corpus_id_set = set(corpus_ids)
    pair_count = 0
    for query_id in query_ids:
        judgments = qrels[query_id]
        if not judgments:
            raise ValueError(f"Query has no judgments: {query_id}")
        if not any(grade > 0 for grade in judgments.values()):
            raise ValueError(f"Query has no positive judgments: {query_id}")
        for document_id, grade in judgments.items():
            if document_id not in corpus_id_set:
                raise ValueError(f"Qrel references unknown document: {document_id}")
            if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
                raise ValueError(f"Invalid qrel grade: {grade!r}")
            pair_count += 1
    return pair_count


def metrics_at_k(
    ranking: list[str], judgments: dict[str, int], k: int = K
) -> dict[str, float]:
    if k < 1 or len(ranking) < k or len(set(ranking[:k])) != k:
        raise ValueError(f"Ranking must contain at least {k} unique IDs")
    retrieved = ranking[:k]
    gains = [
        (2 ** int(judgments.get(document_id, 0)) - 1) / math.log2(position + 2)
        for position, document_id in enumerate(retrieved)
    ]
    ideal_grades = sorted((int(grade) for grade in judgments.values()), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(ideal_grades)
    )
    relevant_total = sum(grade > 0 for grade in judgments.values())
    relevant_retrieved = sum(judgments.get(document_id, 0) > 0 for document_id in retrieved)
    return {
        f"ndcg@{k}": sum(gains) / ideal_dcg if ideal_dcg else 0.0,
        f"recall@{k}": relevant_retrieved / relevant_total if relevant_total else 0.0,
        f"precision@{k}": relevant_retrieved / k,
    }


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved_destination and resolved_destination not in target.parents:
                raise ValueError(f"Unsafe path in BEIR archive: {member.filename}")
        archive.extractall(destination)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return list(_iter_jsonl(path))


def _read_document_ids(path: Path) -> list[str]:
    return [str(record["_id"]) for record in _iter_jsonl(path)]


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected TREC-COVID qrels header: {reader.fieldnames}")
        for row in reader:
            query_id = row["query-id"]
            document_id = row["corpus-id"]
            if document_id in qrels.setdefault(query_id, {}):
                raise ValueError(f"Duplicate qrel: {query_id}/{document_id}")
            qrels[query_id][document_id] = int(row["score"])
    return qrels


def run_benchmark(output_root: Path, expected_source: str) -> dict[str, object]:
    import numpy as np
    import torch
    from arrowspace import ArrowSpaceBuilder
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    output_root.mkdir(parents=True, exist_ok=False)
    repository_root = Path(__file__).resolve().parents[3]
    actual_source = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_source != expected_source:
        raise ValueError(f"Runner source is {actual_source}; notebook pinned {expected_source}")

    input_root = output_root / "inputs"
    input_root.mkdir()
    archive_path = input_root / "trec-covid.zip"
    urllib.request.urlretrieve(DATA_URL, archive_path)
    archive_md5 = digest(archive_path, "md5")
    if archive_md5 != DATA_MD5:
        raise ValueError(f"BEIR archive MD5 mismatch: {archive_md5}")
    archive_sha256 = digest(archive_path)
    extracted_root = input_root / "extracted"
    _safe_extract(archive_path, extracted_root)

    task_root = extracted_root / "trec-covid"
    corpus_path = task_root / "corpus.jsonl"
    queries_path = task_root / "queries.jsonl"
    qrels_source_path = task_root / "qrels" / "test.tsv"
    if not all(path.is_file() for path in (corpus_path, queries_path, qrels_source_path)):
        raise FileNotFoundError("BEIR archive lacks corpus, queries, or test qrels")
    query_records = _read_jsonl(queries_path)
    document_ids = _read_document_ids(corpus_path)
    query_ids = [str(record["_id"]) for record in query_records]
    if len(document_ids) != EXPECTED_DOCUMENTS or len(query_ids) != EXPECTED_QUERIES:
        raise ValueError(f"Unexpected BEIR sizes: {len(document_ids)} docs, {len(query_ids)} queries")
    qrels = _read_qrels(qrels_source_path)
    qrel_pair_count = validate_task_inputs(document_ids, query_ids, qrels)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    if model_snapshot.name != MODEL_REVISION:
        raise ValueError(f"Model snapshot does not match pinned revision: {model_snapshot}")
    model = SentenceTransformer(str(model_snapshot), device=device)
    model.max_seq_length = 256
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 384:
        raise ValueError(f"Unexpected MiniLM output dimension: {dimension}")

    corpus_embedding_path = output_root / "corpus-embeddings.npy"
    corpus_embeddings = np.lib.format.open_memmap(
        corpus_embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(document_ids), dimension),
    )
    batch_texts: list[str] = []
    offset = 0
    for record in _iter_jsonl(corpus_path):
        batch_texts.append(f"{record.get('title') or ''}\n{record.get('text') or ''}".strip())
        if len(batch_texts) == 512:
            vectors = np.asarray(
                model.encode(
                    batch_texts,
                    batch_size=64,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if vectors.shape != (len(batch_texts), dimension) or not np.isfinite(vectors).all():
                raise ValueError("Invalid MiniLM document embedding batch")
            corpus_embeddings[offset : offset + len(batch_texts)] = vectors
            offset += len(batch_texts)
            batch_texts.clear()
    if batch_texts:
        vectors = np.asarray(
            model.encode(
                batch_texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (len(batch_texts), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Invalid MiniLM document embedding batch")
        corpus_embeddings[offset : offset + len(batch_texts)] = vectors
        offset += len(batch_texts)
    corpus_embeddings.flush()
    if offset != len(document_ids):
        raise ValueError("Not all corpus documents were embedded")

    query_embeddings = np.asarray(
        model.encode(
            [str(record["text"]) for record in query_records],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if query_embeddings.shape != (len(query_ids), dimension) or not np.isfinite(query_embeddings).all():
        raise ValueError("Invalid MiniLM query embeddings")
    query_embedding_path = output_root / "query-embeddings.npy"
    np.save(query_embedding_path, query_embeddings, allow_pickle=False)

    model_files = {
        str(path.relative_to(model_snapshot)): digest(path)
        for path in sorted(model_snapshot.rglob("*"))
        if path.is_file()
    }
    embedding_metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_files_sha256": model_files,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "dimension": dimension,
        "max_seq_length": model.max_seq_length,
        "document_text": "title + newline + text",
        "query_text": "query text",
        "normalization": "L2 unit norm",
        "dtype": "float32",
        "document_batch_size": 512,
        "torch_seed": SEED,
    }

    document_ids_path = output_root / "document-ids.json"
    query_ids_path = output_root / "query-ids.json"
    qrels_path = output_root / "qrels.json"
    document_ids_path.write_text(json.dumps(document_ids, separators=(",", ":")) + "\n")
    query_ids_path.write_text(json.dumps(query_ids, separators=(",", ":")) + "\n")
    qrels_path.write_text(json.dumps(qrels, sort_keys=True, separators=(",", ":")) + "\n")
    task_manifest = {
        "schema_version": 1,
        "benchmark": {
            "name": "BEIR TREC-COVID",
            "split": "test",
            "source_url": DATA_URL,
            "archive_md5_expected": DATA_MD5,
            "archive_md5_observed": archive_md5,
            "archive_sha256": archive_sha256,
            "corpus_sha256": digest(corpus_path),
            "queries_sha256": digest(queries_path),
            "qrels_sha256": digest(qrels_source_path),
            "documents": len(document_ids),
            "queries": len(query_ids),
            "qrel_pairs": qrel_pair_count,
        },
        "implementation": {
            "repository": SOURCE_URL,
            "commit": actual_source,
            "crate_version": "0.30.0",
            "python_version": sys.version,
            "platform": platform.platform(),
            "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "rust_version": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            "arrowspace_package_version": importlib.metadata.version("arrowspace"),
            "numpy_version": np.__version__,
            "graph_params": GRAPH_PARAMS,
            "seed": SEED,
            "scale": SCALE,
            "tau_values_predeclared": list(TAU_VALUES),
            "k": K,
        },
        "embedding": embedding_metadata,
        "files": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (
                corpus_embedding_path,
                query_embedding_path,
                document_ids_path,
                query_ids_path,
                qrels_path,
            )
        },
        "label_use": "qrels used only after graph construction and retrieval, for evaluation",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(task_manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha256 = digest(manifest_path)
    task_bundle_path = output_root / "trec-covid-task-bundle.zip"
    bundle_files = (
        manifest_path,
        corpus_embedding_path,
        query_embedding_path,
        document_ids_path,
        query_ids_path,
        qrels_path,
    )
    with zipfile.ZipFile(
        task_bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        for path in bundle_files:
            bundle.write(path, arcname=path.name)
    task_bundle_sha256 = digest(task_bundle_path)

    corpus_scaled = np.ascontiguousarray(corpus_embeddings * SCALE, dtype=np.float64)
    query_scaled = np.ascontiguousarray(query_embeddings * SCALE, dtype=np.float64)
    build_started = time.perf_counter()
    aspace, graph = ArrowSpaceBuilder().with_seed(SEED).build(GRAPH_PARAMS, corpus_scaled)
    build_seconds = time.perf_counter() - build_started
    del corpus_scaled

    document_id_array = np.asarray(document_ids, dtype=str)
    cosine_rows, arrowspace_rows = [], []
    for query_index, query_id in enumerate(query_ids):
        cosine_scores = np.asarray(
            corpus_embeddings @ query_embeddings[query_index], dtype=np.float32
        )
        cosine_order = np.lexsort((document_id_array, -cosine_scores))[:K]
        cosine_ranking = [document_ids[index] for index in cosine_order]
        cosine_rows.append(
            {"query_id": query_id, **metrics_at_k(cosine_ranking, qrels[query_id]), "ranking": cosine_ranking}
        )
        for tau in TAU_VALUES:
            hits = aspace.search(query_scaled[query_index], graph, tau=tau)
            ranking = [document_ids[int(index)] for index, _score in hits[:K]]
            if len(ranking) != K or len(set(ranking)) != K:
                raise ValueError(f"Invalid ArrowSpace top-{K}: query={query_id}, tau={tau}")
            arrowspace_rows.append(
                {"query_id": query_id, "tau": tau, **metrics_at_k(ranking, qrels[query_id]), "ranking": ranking}
            )

    metric_names = (f"ndcg@{K}", f"recall@{K}", f"precision@{K}")
    summary: dict[str, object] = {
        "cosine": {
            name: float(np.mean([row[name] for row in cosine_rows])) for name in metric_names
        },
        "arrowspace": {},
    }
    for tau in TAU_VALUES:
        rows = [row for row in arrowspace_rows if row["tau"] == tau]
        summary["arrowspace"][str(tau)] = {
            name: float(np.mean([row[name] for row in rows])) for name in metric_names
        }
    evaluation = {
        "schema_version": 1,
        "real_trec_covid_results": True,
        "source_commit": actual_source,
        "task_manifest_sha256": manifest_sha256,
        "task_bundle_sha256": task_bundle_sha256,
        "build_seconds": build_seconds,
        "graph_params": GRAPH_PARAMS,
        "tau_values": list(TAU_VALUES),
        "k": K,
        "summary_macro": summary,
        "cosine_per_query": cosine_rows,
        "arrowspace_per_query": arrowspace_rows,
        "interpretation": "Descriptive BEIR TREC-COVID test results; no test-qrel tuning or generalisation claim.",
    }
    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "task_bundle": str(task_bundle_path),
        "task_bundle_sha256": task_bundle_sha256,
        "results": str(results_path),
        "summary": summary,
        "build_seconds": build_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    arguments = parser.parse_args()
    report = run_benchmark(arguments.output, arguments.expected_source)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
