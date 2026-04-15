#!/usr/bin/env bash
# Full RQ3 GSM8K pipeline for all six model dirs (per model: stage1 → … → stage5).
# Models: coconut_gpt2, coconut_llama1b, coconut_qwen3_4b, codi_gpt2, codi_llama1b, codi_qwen3_4b
#
# Each run_stage*_gsm8k.sh skips if its marker output already exists (default).
#   FORCE_RERUN=1 bash run_stage1_...   # rerun that stage
#   SKIP_IF_EXISTS=0 bash run_stage1_...  # same
# Stage1 microbatch (per GPU): scripts default for H200; override all: export BATCH_SIZE=64
# Stage3 skips only when both projection_scores.jsonl and projection_scores_teacher_forced.jsonl exist (so TF can be regenerated if probe exists but TF is missing/bad).
# Stage1/stage3 default to single-process (NPROC=1, CUDA_VISIBLE_DEVICES=0). Multi-GPU: export NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 before running a script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Shared HF weights: optional MODEL_DIR via scripts/common/default_model_dir.sh (sibling ../models if present).
source "${PROJECT_ROOT}/scripts/common/default_model_dir.sh"


MODEL_DIRS=(
  # coconut_gpt2
  # coconut_llama1b
  # coconut_qwen3_4b
  # codi_gpt2
  # codi_llama1b
  codi_qwen3_4b
)

STAGE_SCRIPTS=(
  run_stage1_mine_ambiguous_gsm8k.sh
  run_stage2_build_probe_gsm8k.sh
  run_stage3_projection_analysis_gsm8k.sh
  run_stage4_intervention_gsm8k.sh
  run_stage5_plot_metrics_gsm8k.sh
)

for m in "${MODEL_DIRS[@]}"; do
  for s in "${STAGE_SCRIPTS[@]}"; do
    f="${SCRIPT_DIR}/${m}/${s}"
    if [[ ! -f "${f}" ]]; then
      echo "Missing script: ${f}" >&2
      exit 1
    fi
  done
done

echo "PROJECT_ROOT=${PROJECT_ROOT}"
for m in "${MODEL_DIRS[@]}"; do
  echo "################ ${m} ################"
  for s in "${STAGE_SCRIPTS[@]}"; do
    f="${SCRIPT_DIR}/${m}/${s}"
    echo "-------- ${m} / ${s} --------"
    bash "${f}"
  done
done
echo "All six models × five stages finished."
