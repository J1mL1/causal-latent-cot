#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CONDA_ENV="coconut"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPROC=8                # >1 to enable torchrun
MASTER_PORT="${MASTER_PORT:-29520}"
DIST_URL="env://"
DIST_BACKEND="nccl"
CONFIG="configs/rq3/superposition_coconut_r1_qwen1_5b_commonsenseqa.yaml"
OUTPUT_DIR="outputs/rq3/coconut_r1_qwen1_5b-commonsenseqa/ambiguous"

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

${LAUNCHER} --master_port 29501 experiments/rq3/stage1_mine_ambiguous.py \
  --config_path "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dropout 0.1 \
  --batch_size 8 \
  --num_workers 8 \
  ${DIST_FLAGS}
