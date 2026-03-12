#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq1/plots/sufficiency_earliest"
MAX_STEPS="6"

# Edit inputs as needed.
CSQA_ARGS=(
  "--csqa" "Coconut-GPT2=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_coconut_gpt2.jsonl"
  "--csqa" "Coconut-Llama3-1B=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_coconut_llama1b.jsonl"
  "--csqa" "Coconut-Qwen3-4B=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_coconut_qwen3_4b.jsonl"
  "--csqa" "CODI-GPT2=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_codi_gpt2.jsonl"
  "--csqa" "CODI-Llama3-1B=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_codi_llama1b.jsonl"
  "--csqa" "CODI-Qwen3-4B=${ROOT}/outputs/rq1/sufficiency/commonsenseqa_codi_qwen3_4b.jsonl"
)

GSM8K_ARGS=(
  "--gsm8k" "Coconut-GPT2=${ROOT}/outputs/rq1/sufficiency/gsm8k_coconut_gpt2.jsonl"
  "--gsm8k" "Coconut-Llama3-1B=${ROOT}/outputs/rq1/sufficiency/gsm8k_coconut_llama1b.jsonl"
  "--gsm8k" "Coconut-Qwen3-4B=${ROOT}/outputs/rq1/sufficiency/gsm8k_coconut_qwen3_4b.jsonl"
  "--gsm8k" "CODI-GPT2=${ROOT}/outputs/rq1/sufficiency/gsm8k_codi_gpt2.jsonl"
  "--gsm8k" "CODI-Llama3-1B=${ROOT}/outputs/rq1/sufficiency/gsm8k_codi_llama1b.jsonl"
  "--gsm8k" "CODI-Qwen3-4B=${ROOT}/outputs/rq1/sufficiency/gsm8k_codi_qwen3_4b.jsonl"
)

Rscript "${ROOT}/r-script/plot_sufficiency_earliest_correct_multi.R" \
  --out_dir "${OUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  "${CSQA_ARGS[@]}" \
  "${GSM8K_ARGS[@]}"
