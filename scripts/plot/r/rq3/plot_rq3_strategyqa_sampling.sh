#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq3/plots/rq3-strategyqa-sampling"

INPUTS=(
  "Coconut-GPT2=${ROOT}/outputs/rq3/coconut_gpt2-strategyqa"
  "Coconut-Llama3-1B=${ROOT}/outputs/rq3/coconut_llama1b-strategyqa"
  "Coconut-Qwen3-4B=${ROOT}/outputs/rq3/coconut_qwen3_4b-strategyqa"
  "CODI-GPT2=${ROOT}/outputs/rq3/codi_gpt2-strategyqa"
  "CODI-Llama3-1B=${ROOT}/outputs/rq3/codi_llama1b-strategyqa"
  "CODI-Qwen3-4B=${ROOT}/outputs/rq3/codi_qwen3_4b-strategyqa"
)

ARGS=()
for item in "${INPUTS[@]}"; do
  ARGS+=(--input "${item}")
done

Rscript "${ROOT}/r-script/plot_rq3_strategyqa_sampling.R" \
  --out_dir "${OUT_DIR}" \
  "${ARGS[@]}"
