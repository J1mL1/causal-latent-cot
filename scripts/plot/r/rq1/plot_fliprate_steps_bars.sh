#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Plot flip-rate step-wise ablation bars for CSQA and GSM8K.

OUT_DIR="${PROJECT_ROOT}/outputs/rq1/plots"
MODE="zero"
FILE_STUB="fliprate_step_ablation"

CSQA_ARGS=(
  "--csqa" "Coconut-GPT2=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_coconut_gpt2.jsonl"
  "--csqa" "Coconut-Llama3-1B=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_coconut_llama3-1b.jsonl"
  "--csqa" "Coconut-Qwen3-4B=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_coconut_qwen3_4b.jsonl"
  "--csqa" "CODI-GPT2=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_codi_gpt2.jsonl"
  "--csqa" "CODI-Llama3-1B=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_codi_llama1b.jsonl"
  "--csqa" "CODI-Qwen3-4B=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_codi_qwen3_4b.jsonl"
)

GSM8K_ARGS=(
  "--gsm8k" "Coconut-GPT2=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_gpt2.jsonl"
  "--gsm8k" "Coconut-Llama3-1B=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_llama3-1b.jsonl"
  "--gsm8k" "Coconut-Qwen3-4B=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_qwen3_4b.jsonl"
  "--gsm8k" "CODI-GPT2=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_codi_gpt2.jsonl"
  "--gsm8k" "CODI-Llama3-1B=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_codi_llama1b.jsonl"
  "--gsm8k" "CODI-Qwen3-4B=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_codi_qwen3_4b.jsonl"
)

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

Rscript r-script/plot_fliprate_steps_bars.R \
  --mode "${MODE}" \
  --out_dir "${OUT_DIR}" \
  --file_stub "${FILE_STUB}" \
  "${CSQA_ARGS[@]}" \
  "${GSM8K_ARGS[@]}"
