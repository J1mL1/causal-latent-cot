#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Distributed debugging defaults
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# === Editable parameters ===
CONDA_ENV="latentCoT"
CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC=4               # >1 to enable torchrun
BATCH_SIZE=32
NUM_WORKERS=8
MASTER_PORT="${MASTER_PORT:-29500}"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq1/coconut/qwen3-4b-gsm8k.yaml"
OUTPUT="outputs/rq1/ablation/gsm8k_coconut_qwen3_4b.jsonl"
MODES="zero,mean,gaussian_h,gaussian_mu,mean_step,gaussian_mu_step"
MEAN_CACHE="outputs/rq1/mean_latents/coconut_qwen3_4b.pt"
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

if [ ! -f "${MEAN_CACHE}" ]; then
  python experiments/rq1/run_step_ablation.py \
    --model_name coconut \
    --config_path "${CONFIG}" \
    --mean_cache_path "${MEAN_CACHE}" \
    --only_estimate_mean \
    --output_path "/tmp/coconut_qwen3_4b_mean_only.jsonl" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
fi

${LAUNCHER} --master_port 29501 experiments/rq1/run_step_ablation.py \
  --model_name coconut \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --modes "${MODES}" \
  --mean_cache_path "${MEAN_CACHE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  ${DIST_FLAGS}
