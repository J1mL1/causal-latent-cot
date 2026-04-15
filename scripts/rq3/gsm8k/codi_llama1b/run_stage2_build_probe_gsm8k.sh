#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Shared HF weights: optional MODEL_DIR via scripts/common/default_model_dir.sh (sibling ../models if present).
source "${PROJECT_ROOT}/scripts/common/default_model_dir.sh"

RUN_SLUG="${RUN_SLUG:-codi-llama1b-gsm8k}"

source "${SCRIPT_DIR}/../_gsm8k_skip_helpers.sh"
if gsm8k_skip_if_file "${PROJECT_ROOT}/outputs/rq3/${RUN_SLUG}/probes/probes.jsonl"; then exit 0; fi


OUTPUT_BASE="outputs/rq3/${RUN_SLUG}"

INPUT_DIR="${OUTPUT_BASE}/ambiguous"
OUTPUT_DIR="${OUTPUT_BASE}/probes"


cd "${PROJECT_ROOT}"
if [ -z "${PYTHONPATH-}" ]; then
  export PYTHONPATH="$(pwd)"
else
  export PYTHONPATH="$(pwd):${PYTHONPATH}"
fi

python experiments/rq3/stage2_build_probe.py \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --probe_step final \
  --method mean_sub
