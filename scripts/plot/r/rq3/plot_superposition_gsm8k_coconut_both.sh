#!/usr/bin/env bash
# GSM8K superposition (probe + teacher-forced): Coconut & CODI × {GPT2, Llama3.2-1B, Qwen3-4B}.
# Paths: outputs/rq3/coco-*-gsm8k/, outputs/rq3/codi-*-gsm8k/ (RUN_SLUGs under scripts/rq3/gsm8k).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq3/plots/rq3-superposition-gsm8k-coco-codi"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${R_ENV_NAME:-latentcot-r}"

mkdir -p "${OUT_DIR}"

Rscript "${ROOT}/r-script/plot_rq3_superposition_both.R" \
  --out_path "${OUT_DIR}/rq3_superposition_both_gsm8k_coco_codi_gpt2_llama_qwen3.pdf" \
  --probe "Coconut-GPT2-GSM8K=${ROOT}/outputs/rq3/coco-gpt2-gsm8k/projection_scores.jsonl" \
  --probe "Coconut-Llama3-1B-GSM8K=${ROOT}/outputs/rq3/coco-llama1b-gsm8k/projection_scores.jsonl" \
  --probe "Coconut-Qwen3-4B-GSM8K=${ROOT}/outputs/rq3/coco-qwen3-4b-gsm8k/projection_scores.jsonl" \
  --probe "CODI-GPT2-GSM8K=${ROOT}/outputs/rq3/codi-gpt2-gsm8k/projection_scores.jsonl" \
  --probe "CODI-Llama3-1B-GSM8K=${ROOT}/outputs/rq3/codi-llama1b-gsm8k/projection_scores.jsonl" \
  --tf "Coconut-GPT2-GSM8K=${ROOT}/outputs/rq3/coco-gpt2-gsm8k/projection_scores_teacher_forced.jsonl" \
  --tf "Coconut-Llama3-1B-GSM8K=${ROOT}/outputs/rq3/coco-llama1b-gsm8k/projection_scores_teacher_forced.jsonl" \
  --tf "Coconut-Qwen3-4B-GSM8K=${ROOT}/outputs/rq3/coco-qwen3-4b-gsm8k/projection_scores_teacher_forced.jsonl" \
  --tf "CODI-GPT2-GSM8K=${ROOT}/outputs/rq3/codi-gpt2-gsm8k/projection_scores_teacher_forced.jsonl" \
  --tf "CODI-Llama3-1B-GSM8K=${ROOT}/outputs/rq3/codi-llama1b-gsm8k/projection_scores_teacher_forced.jsonl"
