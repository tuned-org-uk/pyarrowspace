#!/usr/bin/env bash
set -euo pipefail

# Build pretrained MiniLM embeddings for cleaned TREC-COVID data.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

DATASET_DIR="${1:-${ROOT_DIR}/data/trec-covid}"
OUTPUT_ROOT="${2:-${ROOT_DIR}/embeddings/no_finetune_minilm}"
MODEL_NAME="${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${DEVICE:-mps}"

mkdir -p "${OUTPUT_ROOT}"

python3 "${REPO_ROOT}/Test-clean/trec-covid/embed_trec_pretrained.py" \
  --dataset_dir "${DATASET_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --normalize_embeddings

echo "[done] embeddings in ${OUTPUT_ROOT}/cache"
