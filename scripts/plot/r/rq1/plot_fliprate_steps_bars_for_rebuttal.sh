#!/usr/bin/env bash
# GSM8K flip-rate plot for four Coconut models (rebuttal figure).
# Style matches scripts/plot/r/rq1/plot_fliprate_steps_bars.sh (R: plot_fliprate_steps_bars_for_rebuttal.R).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

OUT_DIR="${PROJECT_ROOT}/outputs/rq1/plots/plot_fliprate_steps_bars_for_rebuttal"
MODE="${MODE:-zero}"
FILE_STUB="${FILE_STUB:-plot_fliprate_steps_bars_for_rebuttal}"

mkdir -p "${OUT_DIR}"

GSM8K_ARGS=(
  "--gsm8k" "Coconut-Qwen3-1.7B=${PROJECT_ROOT}/outputs/rq1/intervention/gsm8k_coconut_qwen3_1.7b.jsonl"
  "--gsm8k" "Coconut-Mistral-7B=${PROJECT_ROOT}/outputs/rq1/intervention/gsm8k_coconut_mistral.jsonl"
  "--gsm8k" "Coconut-DeepSeek-R1-Qwen2.5-1.5B=${PROJECT_ROOT}/outputs/rq1/intervention/gsm8k_coconut_r1_qwen1.5b.jsonl"
  "--gsm8k" "Coconut-Llama3-8B=${PROJECT_ROOT}/outputs/rq1/intervention/gsm8k_coconut_llama3-8b.jsonl"
)

for pair in "${GSM8K_ARGS[@]}"; do
  if [[ "${pair}" == "--gsm8k" ]]; then
    continue
  fi
  path="${pair#*=}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing intervention JSONL: ${path}" >&2
    exit 1
  fi
done

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${R_ENV_NAME:-latentcot-r}"

Rscript "${PROJECT_ROOT}/r-script/plot_fliprate_steps_bars_for_rebuttal.R" \
  --mode "${MODE}" \
  --out_dir "${OUT_DIR}" \
  --file_stub "${FILE_STUB}" \
  "${GSM8K_ARGS[@]}"
