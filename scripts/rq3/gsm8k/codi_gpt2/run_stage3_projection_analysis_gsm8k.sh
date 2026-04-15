#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Shared HF weights: optional MODEL_DIR via scripts/common/default_model_dir.sh (sibling ../models if present).
source "${PROJECT_ROOT}/scripts/common/default_model_dir.sh"

RUN_SLUG="${RUN_SLUG:-codi-gpt2-gsm8k}"

source "${SCRIPT_DIR}/../_gsm8k_skip_helpers.sh"
OUT_PROBE="${PROJECT_ROOT}/outputs/rq3/${RUN_SLUG}/projection_scores.jsonl"
OUT_TF="${PROJECT_ROOT}/outputs/rq3/${RUN_SLUG}/projection_scores_teacher_forced.jsonl"
if gsm8k_skip_if_file "${OUT_PROBE}" && gsm8k_skip_if_file "${OUT_TF}"; then exit 0; fi


OUTPUT_BASE="outputs/rq3/${RUN_SLUG}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"
# TF path is VRAM-heavy (per-step forward + GSM8K pair logits); raise if stable.
BATCH_SIZE=1
MASTER_PORT="${MASTER_PORT:-29522}"
DIST_URL="env://"
DIST_BACKEND="nccl"

INPUT_DIR="${OUTPUT_BASE}/ambiguous"
PROBE_DIR="${OUTPUT_BASE}/probes"
OUT_JSONL="${OUTPUT_BASE}/projection_scores.jsonl"
CONFIG="configs/rq3/superposition_codi_gpt2_gsm8k.yaml"

export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
if [ -z "${PYTHONPATH-}" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
fi

if [ "${NPROC}" -gt 1 ]; then
  LAUNCHER="torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT}"
  DIST_FLAGS="--distributed --dist_url ${DIST_URL} --dist_backend ${DIST_BACKEND}"
else
  LAUNCHER="python"
  DIST_FLAGS=""
fi

# CODI: teacher-forced uses forward_until_step on prompts (avoids latent_path + LoRA dtype mismatches → summary-only TF).
${LAUNCHER} experiments/rq3/stage3_projection_analysis.py \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --probes_jsonl "${PROBE_DIR}/probes.jsonl" \
  --output_jsonl "${OUT_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  ${DIST_FLAGS} \
  --config_path "${CONFIG}" \
  --tf_no_latent_path \
  --early_steps 1,2,3,4,5 \
  --p_mode "${P_MODE:-sigmoid}" \
  --tau "${P_TAU:-20.0}"
