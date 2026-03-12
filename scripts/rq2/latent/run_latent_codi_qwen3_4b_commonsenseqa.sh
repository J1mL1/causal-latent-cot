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
BATCH_SIZE=16
NUM_WORKERS=0
STEPS="1,2,3,4,5,6"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq1/codi/qwen3-4b-commonsenseqa.yaml"
OUTPUT="outputs/rq2/latent_graph/commonsenseqa_codi_qwen3_4b_latent_graph.jsonl"
MODE="zero"            # zero | gaussian_h | gaussian
INCLUDE_SELF="--include_self"  # set to "" to disable
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
  LAUNCHER="torchrun --nproc_per_node=${NPROC}"
  DIST_FLAGS="--distributed --dist_url ${DIST_URL} --dist_backend ${DIST_BACKEND}"
else
  LAUNCHER="python"
  DIST_FLAGS=""
fi

${LAUNCHER} --master_port 29556 experiments/rq2/run_latent_causal_graph.py \
  --model_name codi \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --steps "${STEPS}" \
  --mode "${MODE}" \
  ${INCLUDE_SELF} \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  ${DIST_FLAGS}
