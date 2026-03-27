#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# === Editable parameters ===
CONDA_ENV="latentCoT"
CUDA_VISIBLE_DEVICES="0"
NPROC=1
BATCH_SIZE=64
NUM_WORKERS=2
MASTER_PORT="${MASTER_PORT:-29500}"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq1/simcot/gpt2-gsm8k.yaml"
OUTPUT="outputs/rq1/intervention/gsm8k_simcot_coconut_gpt2.jsonl"
MODES="zero,mean,gaussian_h,gaussian_mu,mean_step,gaussian_mu_step"
MEAN_CACHE="outputs/rq1/mean_latents/simcot_coconut_gpt2.pt"
# ===========================

export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1

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

rm -f "${MEAN_CACHE}"

${LAUNCHER} experiments/rq1/run_step_intervention.py \
  --model_name simcot-coconut \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --modes "${MODES}" \
  --mean_cache_path "${MEAN_CACHE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  ${DIST_FLAGS}
