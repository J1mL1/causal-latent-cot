#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

OUT_DIR="outputs/rq1/plots/csqa_ablation_bars"
MODE="zero"     # change if you want a different ablation mode
LOGP_KIND="final"      # seq -> teacher_forced_delta_sum, final -> delta_logp_final_token

# Rscript r-script/plot_ablation_bars.R \
#   --mode "${MODE}" \
#   --logp_kind "${LOGP_KIND}" \
#   --out_dir "${OUT_DIR}" \
#   --input "coconut_llama3-1b=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_llama3-1b.jsonl" \
#   --input "coconut_gpt2=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_gpt2.jsonl" \
#   --input "coconut_r1_qwen1.5b=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_r1_qwen1.5b.jsonl" \
#   --input "coconut_qwen3_1.7b=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_qwen3_1.7b.jsonl" \
#   --input "codi_gpt2=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_codi_gpt2.jsonl" \
#   --input "codi_llama=${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_codi_llama1b.jsonl"

Rscript r-script/plot_ablation_bars.R \
  --mode "${MODE}" \
  --logp_kind "${LOGP_KIND}" \
  --out_dir "${OUT_DIR}" \
  --input "coconut_llama3-1b=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_coconut_llama3-1b.jsonl" \
  --input "coconut_gpt2=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_coconut_gpt2.jsonl" \
  --input "codi_gpt2=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_codi_gpt2.jsonl" \
  --input "codi_llama=${PROJECT_ROOT}/outputs/rq1/ablation/commonsenseqa_codi_llama1b.jsonl"
