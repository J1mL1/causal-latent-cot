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
  "coco-gpt2-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-coco-gpt2-gsm8k"
  "coco-llama1b-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-coco-llama1b-gsm8k"
  "coco-qwen3-4b-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-coco-qwen3-4b-gsm8k"
  "codi-gpt2-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-codi-gpt2-gsm8k"
  "codi-llama1b-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-codi-llama1b-gsm8k"
  "codi-qwen3-4b-gsm8k=${ROOT}/outputs/rq3/plots/rq3-metrics-codi-qwen3-4b-gsm8k"
)

ARGS=()
for item in "${INPUTS[@]}"; do
  ARGS+=(--input "${item}")
done

Rscript "${ROOT}/r-script/plot_rq3_multi_model.R" \
  --out_dir "${OUT_DIR}" \
  "${ARGS[@]}"
