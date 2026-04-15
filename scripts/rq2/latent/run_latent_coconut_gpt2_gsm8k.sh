#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Distributed debugging defaults
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# === Editable parameters ===
CONDA_ENV="latentCoT"
NPROC=4               # >1 to enable torchrun
BATCH_SIZE=16
NUM_WORKERS=8
STEPS="1,2,3,4,5,6"
MASTER_PORT="${MASTER_PORT:-29501}"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq1/coconut/gpt2-gsm8k.yaml"
OUTPUT="outputs/rq2/latent_graph/gsm8k_coconut_gpt2_latent_graph_gaussian_h.jsonl"
MODE="gaussian_h"
INCLUDE_SELF=""
# Used only when MODE is mean/mean_step; in distributed mode rank0 writes, other ranks load.
MEAN_CACHE="outputs/rq1/mean_latents/coconut_gpt2.pt"
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

${LAUNCHER} experiments/rq2/run_latent_causal_graph.py \
  --model_name coconut \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --mode "${MODE}" \
  --steps "${STEPS}" \
  ${INCLUDE_SELF} \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --mean_cache_path "${MEAN_CACHE}" \
  ${DIST_FLAGS}
