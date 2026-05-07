# CVE Reproducibility Pack

This folder contains a reproducible setup for CVE™ dataset experiments focused on:
- Domain-adapted embedding fine-tuning (`01_embeddings_model.py`)
- Hyperparameter sweep over `eps/k` using EigenMaps (`02_eigen_maps_params_learning.py`)
- Full tau sweep and semantic recall experiments (`test_17_CVE_neurips.py`, `test_17_CVE_neurips_v2.py`)

## Folder layout

- `01_embeddings_model.py`: fine-tuning script to produce a domain-adapted sentence-transformer model.
- `02_eigen_maps_params_learning.py`: hyperparameter sweep script for `eps` and `build_k` (nDCG@10).
- `test_17_CVE_neurips.py`: main NeurIPS experiment script (tau sweep, semantic recall, PageRank).
- `test_17_CVE_neurips_v2.py`: updated variant of the main experiment script.
- `output/`: logs, plots, and LaTeX artifacts from experiment runs.
- `pyproject.toml`: project dependencies for this reproducibility pack.

## Dataset source (CVE™)

Dataset is the CVE List V5 from the CVE Project:
- Version used: `2026-04-29_all_CVEs_at_midnight.zip`
- URL: `https://github.com/CVEProject/cvelistV5/releases/download/cve_2026-04-29_1800Z/2026-04-29_all_CVEs_at_midnight.zip.zip`
- Download manually with `wget <URL>` and unzip into a local directory.

## Install

From this directory:

```bash
pip install -e .
```

## Reproduce

Run step-by-step:

```bash
# 1. Fine-tune a domain-adapted embedding model on CVE data
python CVE/01_embeddings_model.py

# 2. (Optional) Re-run the hyperparameter sweep for eps and build_k
python CVE/02_eigen_maps_params_learning.py

# 3. Run the main NeurIPS experiment (tau sweep, semantic recall, PageRank)
python CVE/test_17_CVE_neurips.py --dataset <PATH>

# 4. Run the v2 variant of the experiment (to add Semantic Uplift analysis)
python CVE/test_17_CVE_neurips_v2.py --dataset <PATH>
```

## Script details

- `01_embeddings_model.py`
  - Fine-tunes `sentence-transformers/all-MiniLM-L6-v2` on CVE corpus using TSDAE.
  - Writes the domain-adapted model to a local cache directory.
  - Instructions and configurable paths are documented inside the file.

- [optional] `02_eigen_maps_params_learning.py`
  - Runs a manual grid search over `eps` and `build_k` for nDCG@10.
  - Hyperparameters are set at the top of the file.
  - Writes sweep outputs to `CVE/output/`.

- `test_17_CVE_neurips.py` / `test_17_CVE_neurips_v2.py`
  - Loads the domain-adapted embeddings and builds an ArrowSpace index.
  - Runs a full tau sweep and computes semantic recall comparisons (cosine vs. taumode).
  - Hyperparameters are hard-coded at the top of the file; use `02_eigen_maps_params_learning.py` to re-derive them.
  - Writes logs, plots, and LaTeX tables to `CVE/output/`.

## Notes

- Default device for embedding fine-tuning is `mps`. Override with:
  - `DEVICE=cpu python CVE/01_embeddings_model.py`
- If matplotlib cache warnings appear, set:
  - `export MPLCONFIGDIR=/tmp/mplcache`
- The hard-coded hyperparameters in `test_17_CVE_neurips.py` were derived from the sweep in `02_eigen_maps_params_learning.py`. Re-running the sweep may yield slightly different optimal values depending on the CVE dataset version used.
