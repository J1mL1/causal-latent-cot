#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Shared HF weights: optional MODEL_DIR via scripts/common/default_model_dir.sh (sibling ../models if present).
source "${PROJECT_ROOT}/scripts/common/default_model_dir.sh"

RUN_SLUG="${RUN_SLUG:-codi-llama1b-gsm8k}"


source "${SCRIPT_DIR}/../_gsm8k_skip_helpers.sh"
if gsm8k_skip_if_file "${PROJECT_ROOT}/outputs/rq3/${RUN_SLUG}/ambiguous/ambiguous_samples.jsonl"; then exit 0; fi


CONDA_ENV="latentCoT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29520}"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq3/superposition_codi_llama1b_gsm8k.yaml"
OUTPUT_DIR="outputs/rq3/${RUN_SLUG}/ambiguous"

BATCH_SIZE="${BATCH_SIZE:-64}"

export CUDA_VISIBLE_DEVICES
# Sync CUDA (very slow); only for debugging: export CUDA_LAUNCH_BLOCKING=1

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

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

${LAUNCHER} experiments/rq3/stage1_mine_ambiguous.py \
  --config_path "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dropout 0.1 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 8 \
  ${DIST_FLAGS}
