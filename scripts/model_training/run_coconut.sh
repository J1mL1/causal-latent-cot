#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Distributed debugging defaults
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Activate conda base and choose env
eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate latentCoT

cd "${PROJECT_ROOT}/external/coconut"

if [ -z "${PYTHONPATH-}" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
fi

# torchrun --nnodes 1 --nproc_per_node 8 run.py ${PROJECT_ROOT}/external/coconut/args/prontoqa_coconut.yaml

# torchrun --nnodes 1 --nproc_per_node 8 run.py ${PROJECT_ROOT}/external/coconut/args/gsm_coconut_qwen.yaml

# torchrun --master_port 29555 --nnodes 1 --nproc_per_node 4 run.py ${PROJECT_ROOT}/external/coconut/args/commonsenseqa_coconut_qwen4b.yaml

# torchrun --master_port 29555 --nnodes 1 --nproc_per_node 4 run.py ${PROJECT_ROOT}/external/coconut/args/strategyqa_coconut_gpt2.yaml

torchrun --master_port 29555 --nnodes 1 --nproc_per_node 4 run.py ${PROJECT_ROOT}/external/coconut/args/commonsenseqa_coconut_qwen4b.yaml

# torchrun --master_port 29555 --nnodes 1 --nproc_per_node 4 run.py ${PROJECT_ROOT}/external/coconut/args/strategyqa_coconut_qwen4b.yaml