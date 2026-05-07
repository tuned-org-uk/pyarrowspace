#!/usr/bin/env bash
set -euo pipefail

# Generate NeurIPS-style plots/tables from k=10 logs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
BASE_DIR="${1:-${ROOT_DIR}/results/k10_tau}"

TMP_BASE="${ROOT_DIR}/results/.tmp_neurips_style"
mkdir -p "${TMP_BASE}"
rm -rf "${TMP_BASE}/k_10"
ln -s "${BASE_DIR}" "${TMP_BASE}/k_10"

python3 "${REPO_ROOT}/Test-clean/trec-covid/no_finetune_minilm/scale100_full_tau_report/make_neurips_style_from_logs.py" \
  --base_dir "${TMP_BASE}" \
  --ks "10"

echo "[done] NeurIPS-style plots under ${BASE_DIR}/plots_neurips_style"
