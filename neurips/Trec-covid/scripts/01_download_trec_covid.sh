#!/usr/bin/env bash
set -euo pipefail

# Download BEIR TREC-COVID into Trec-covid/data/trec-covid
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
TARGET_DIR="${DATA_DIR}/trec-covid"
ZIP_PATH="${DATA_DIR}/trec-covid.zip"
URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

mkdir -p "${DATA_DIR}"

if [ -f "${TARGET_DIR}/corpus.jsonl" ] && [ -f "${TARGET_DIR}/queries.jsonl" ] && [ -d "${TARGET_DIR}/qrels" ]; then
  echo "[skip] Dataset already exists at ${TARGET_DIR}"
  exit 0
fi

echo "[download] ${URL}"
curl -L "${URL}" -o "${ZIP_PATH}"
echo "[extract] ${ZIP_PATH} -> ${DATA_DIR}"
unzip -o "${ZIP_PATH}" -d "${DATA_DIR}"
echo "[done] dataset at ${TARGET_DIR}"
