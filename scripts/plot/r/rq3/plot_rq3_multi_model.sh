#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq3/plots/rq3-metrics-multi"

# Edit inputs as needed: label=metrics_dir
INPUTS=(
  "coconut_gpt2=${ROOT}/outputs/rq3/plots/rq3-metrics-coconut_gpt2"
  "coconut_llama1b=${ROOT}/outputs/rq3/plots/rq3-metrics-coconut_llama1b"
  "coconut_qwen3_1_7b=${ROOT}/outputs/rq3/plots/rq3-metrics-coconut_qwen3_1_7b"
  "coconut_r1_qwen1_5b=${ROOT}/outputs/rq3/plots/rq3-metrics-coconut_r1_qwen1_5b"
  "codi_gpt2=${ROOT}/outputs/rq3/plots/rq3-metrics-codi_gpt2"
  "codi_llama1b=${ROOT}/outputs/rq3/plots/rq3-metrics-codi_llama1b"
)

ARGS=()
for item in "${INPUTS[@]}"; do
  ARGS+=(--input "${item}")
done

Rscript "${ROOT}/r-script/plot_rq3_multi_model.R" \
  --out_dir "${OUT_DIR}" \
  "${ARGS[@]}"
