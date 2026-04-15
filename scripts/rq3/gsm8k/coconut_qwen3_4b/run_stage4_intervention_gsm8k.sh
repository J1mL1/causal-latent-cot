#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Shared HF weights: optional MODEL_DIR via scripts/common/default_model_dir.sh (sibling ../models if present).
source "${PROJECT_ROOT}/scripts/common/default_model_dir.sh"

RUN_SLUG="${RUN_SLUG:-coco-qwen3-4b-gsm8k}"

source "${SCRIPT_DIR}/../_gsm8k_skip_helpers.sh"
if gsm8k_skip_if_file "${PROJECT_ROOT}/outputs/rq3/${RUN_SLUG}/interventions.jsonl"; then exit 0; fi


OUTPUT_BASE="outputs/rq3/${RUN_SLUG}"

CONDA_ENV="latentCoT"
CONFIG="configs/rq3/superposition_coconut_qwen3_4b_gsm8k.yaml"
INPUT_DIR="${OUTPUT_BASE}/ambiguous"
PROBE_DIR="${OUTPUT_BASE}/probes"
OUT_JSONL="${OUTPUT_BASE}/interventions.jsonl"


eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
if [ -z "${PYTHONPATH-}" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
fi

python experiments/rq3/stage4_intervention.py \
  --config_path "${CONFIG}" \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --probes_jsonl "${PROBE_DIR}/probes.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --modes "probe,counterfactual" \
  --output_jsonl "${OUT_JSONL}" \
  --intervene_steps 1,2,3,4,5,6 \
  --lambda_scale 1.0
