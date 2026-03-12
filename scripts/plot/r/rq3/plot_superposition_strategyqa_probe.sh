#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq3/plots/rq3-superposition-strategyqa"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

mkdir -p "${OUT_DIR}"

Rscript "${ROOT}/r-script/plot_rq3_superposition_scores.R" \
  --mode "probe" \
  --out_path "${OUT_DIR}/rq3_superposition_probe_strategyqa.pdf" \
  --input "Coconut-GPT2=${ROOT}/outputs/rq3/coconut_gpt2-strategyqa/projection_scores.jsonl" \
  --input "Coconut-Llama3-1B=${ROOT}/outputs/rq3/coconut_llama1b-strategyqa/projection_scores.jsonl" \
  --input "Coconut-Qwen3-4B=${ROOT}/outputs/rq3/coconut_qwen3_4b-strategyqa/projection_scores.jsonl" \
  --input "CODI-GPT2=${ROOT}/outputs/rq3/codi_gpt2-strategyqa/projection_scores.jsonl" \
  --input "CODI-Llama3-1B=${ROOT}/outputs/rq3/codi_llama1b-strategyqa/projection_scores.jsonl" \
  --input "CODI-Qwen3-4B=${ROOT}/outputs/rq3/codi_qwen3_4b-strategyqa/projection_scores.jsonl"
