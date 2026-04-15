#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

INPUT_DIR="outputs/rq3/coconut_r1_qwen1_5b-commonsenseqa/ambiguous"
OUTPUT_DIR="outputs/rq3/coconut_r1_qwen1_5b-commonsenseqa/probes"


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
