#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CONDA_ENV="latentCoT"
BATCH_SIZE=16
NUM_WORKERS=2
CONFIG="configs/rq2/explicit/cot-qwen3-4b-commonsenseqa.yaml"
OUTPUT="outputs/rq2/explicit_graph/commonsenseqa_cot_qwen3_4b_explicit_graph.jsonl"
MAX_STEPS=6
EDGE_METRIC="${EDGE_METRIC:-delta_logp_seq}"

mkdir -p outputs/rq2/explicit_graph

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
if [ -z "${PYTHONPATH-}" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
fi

python experiments/rq2/run_explicit_causal_graph.py \
  --config_path "${CONFIG}" \
  --output_path "${OUTPUT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_steps "${MAX_STEPS}" \
  --edge_metric "${EDGE_METRIC}" \
  --save_adj
