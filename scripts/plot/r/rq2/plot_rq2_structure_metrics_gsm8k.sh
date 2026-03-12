#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
LATENT_CSV="${ROOT}/outputs/rq2/latent_graph/causal_metrics.csv"
EXPLICIT_DIR="${ROOT}/outputs/rq2/explicit_graph"
EXPLICIT_CSV="${ROOT}/outputs/rq2/explicit_graph/causal_metrics.csv"
OUT_DIR="${ROOT}/outputs/rq2/plots/rq2_figures"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

mkdir -p "${OUT_DIR}"

python "${ROOT}/scripts/plot/python/rq1/compute_causal_metrics.py" \
  --input_dir "${ROOT}/outputs/rq2/latent_graph" \
  --output_csv "${LATENT_CSV}" \
  --plot_dir "${ROOT}/outputs/rq2/latent_graph/plots"

python "${ROOT}/scripts/plot/python/rq2/compute_explicit_causal_metrics.py" \
  --input_dir "${EXPLICIT_DIR}" \
  --output_csv "${EXPLICIT_CSV}" \
  --dataset "gsm8k"

Rscript "${ROOT}/r-script/plot_rq2_structure_metrics.R" \
  --latent_csv "${LATENT_CSV}" \
  --explicit_csv "${EXPLICIT_CSV}" \
  --dataset "gsm8k" \
  --metric "kl_mean" \
  --out_path "${OUT_DIR}/fig_rq2_3_gsm8k_structure_metrics_v2.pdf"
