# Trec-covid Reproducibility Pack

This folder contains a reproducible setup for TREC-COVID experiments focused on:
- `k=10` tau sweep analysis
- `eps/build_k` validation sweep (nDCG@10)
- plot generation and copied reference outputs

## Folder layout

- `scripts/`: runnable scripts for download, embedding, experiments, and plotting.
- `data/`: dataset location (`trec-covid/` created here).
- `embeddings/`: embedding cache outputs.
- `results/k10_tau/`: logs, plots, and LaTeX artifacts for `k=10` tau sweep.
- `results/k10_eps_buildk/`: logs and plots for `eps/build_k` sweep.

## Dataset source (TREC-COVID)

Dataset is from BEIR:
- URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip`
- Download script: `scripts/01_download_trec_covid.sh`

Expected extracted files:
- `data/trec-covid/corpus.jsonl`
- `data/trec-covid/queries.jsonl`
- `data/trec-covid/qrels/`

## Reproduce

Run from repo root:

```bash
bash Trec-covid/scripts/00_run_all.sh
```

Or step-by-step:

```bash
bash Trec-covid/scripts/01_download_trec_covid.sh
bash Trec-covid/scripts/02_embed_minilm.sh
bash Trec-covid/scripts/03_run_k10_tau_report.sh
bash Trec-covid/scripts/04_make_neurips_style_plots.sh
bash Trec-covid/scripts/05_run_k10_eps_buildk_sweep.sh
```

## Script details

- `01_download_trec_covid.sh`
  - Downloads and unzips BEIR TREC-COVID into `Trec-covid/data/`.

- `02_embed_minilm.sh`
  - Uses pretrained `sentence-transformers/all-MiniLM-L6-v2`.
  - Writes embedding cache into `Trec-covid/embeddings/no_finetune_minilm/cache/`.

- `03_run_k10_tau_report.sh`
  - Runs full tau sweep at `k=10`.
  - Writes outputs to `Trec-covid/results/k10_tau/`.

- `04_make_neurips_style_plots.sh`
  - Generates NeurIPS-style plots/tables from `k=10` logs.

- `05_run_k10_eps_buildk_sweep.sh`
  - Runs manual grid over `eps` and `build_k` for `nDCG@10`.
  - Writes outputs to `Trec-covid/results/k10_eps_buildk/`.

## Notes

- Default device for embedding is `mps`. Override with:
  - `DEVICE=cpu bash Trec-covid/scripts/02_embed_minilm.sh`
- If matplotlib cache warnings appear, set:
  - `export MPLCONFIGDIR=/tmp/mplcache`
