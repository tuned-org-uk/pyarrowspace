#!/usr/bin/env bash
set -euo pipefail

# Run manual eps/build_k validation for nDCG@10.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

DATASET_DIR="${1:-${ROOT_DIR}/data/trec-covid}"
CACHE_DIR="${2:-${ROOT_DIR}/embeddings/no_finetune_minilm/cache}"
OUT_DIR="${3:-${ROOT_DIR}/results/k10_eps_buildk}"

mkdir -p "${OUT_DIR}"

python3 "${REPO_ROOT}/Test-clean/trec-covid/no_finetune_minilm/finetune-as/manual_ndcg10_param_grid.py" \
  --dataset_dir "${DATASET_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUT_DIR}" \
  --mode sequential \
  --eps_values "1,2,5" \
  --fixed_build_k 30 \
  --build_k_values "25,30,50" \
  --tau_values "0.4,0.5,0.55,0.6,0.65,0.7,0.8,1.0" \
  --scale 100 \
  --search_k 10 \
  --p 2.0 \
  --seed 42

echo "[done] eps/build_k sweep outputs in ${OUT_DIR}"
