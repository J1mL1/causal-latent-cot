#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CONDA_ENV="codi"
CONFIG="configs/rq3/superposition_codi_llama1b_gsm8k.yaml"
INPUT_DIR="outputs/rq3/codi_llama1b/ambiguous"
PROBE_DIR="outputs/rq3/codi_llama1b/probes"
OUT_JSONL="outputs/rq3/codi_llama1b/interventions.jsonl"


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
  --ablate_steps 1,2,3,4,5,6 \
  --lambda_scale 1.0
