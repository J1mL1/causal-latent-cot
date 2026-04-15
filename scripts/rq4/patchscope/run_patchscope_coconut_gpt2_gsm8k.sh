#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

export NCCL_ASYNC_ERROR_HANDLING=1

CONDA_ENV="${CONDA_ENV:-latentCoT}"
CONFIG="${CONFIG:-configs/rq4/coconut/gpt2-gsm8k.yaml}"
OUTPUT="${OUTPUT:-outputs/rq4/patchscope/gsm8k_coconut_gpt2_patchscope.jsonl}"
STEPS="${STEPS:-1,2,3,4,5,6}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA=()
if [ -n "${MAX_SAMPLES}" ]; then
  EXTRA+=(--max_samples "${MAX_SAMPLES}")
fi

python experiments/rq4/run_patchscope.py \
  --model_name coconut \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 0 \
  "${EXTRA[@]}"
