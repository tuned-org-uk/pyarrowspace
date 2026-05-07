#!/usr/bin/env bash
set -euo pipefail

# Run k=10 tau sweep report and generate logs/plots.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

DATASET_DIR="${1:-${ROOT_DIR}/data/trec-covid}"
CACHE_DIR="${2:-${ROOT_DIR}/embeddings/no_finetune_minilm/cache}"
OUT_DIR="${3:-${ROOT_DIR}/results/k10_tau}"

mkdir -p "${OUT_DIR}"

python3 "${REPO_ROOT}/Test-clean/trec-covid/no_finetune_minilm/run_full_tau_report.py" \
  --dataset_dir "${DATASET_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUT_DIR}" \
  --scale 100 \
  --tau_values "0.4,0.5,0.55,0.6,0.65,0.7,0.8,1.0" \
  --eval_ks "10" \
  --neighbor_k 30 \
  --search_depth 10 \
  --head_k 3 \
  --headk_sweep "3,5,10" \
  --reference_tau 1.0 \
  --semantic_percentile 25 \
  --eps 3.9686 \
  --p 2.0 \
  --seed 42

echo "[done] k=10 tau report in ${OUT_DIR}"
