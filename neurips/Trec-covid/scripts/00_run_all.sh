#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/01_download_trec_covid.sh"
"${ROOT_DIR}/scripts/02_embed_minilm.sh"
"${ROOT_DIR}/scripts/03_run_k10_tau_report.sh"
"${ROOT_DIR}/scripts/04_make_neurips_style_plots.sh"
"${ROOT_DIR}/scripts/05_run_k10_eps_buildk_sweep.sh"

echo "[done] full TREC-COVID reproducibility pipeline complete"
