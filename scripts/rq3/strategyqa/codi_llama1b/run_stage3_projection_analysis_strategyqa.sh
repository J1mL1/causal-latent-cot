#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC=4
BATCH_SIZE=8
DIST_URL="env://"
DIST_BACKEND="nccl"

INPUT_DIR="outputs/rq3/codi_llama1b-strategyqa/ambiguous"
PROBE_DIR="outputs/rq3/codi_llama1b-strategyqa/probes"
OUT_JSONL="outputs/rq3/codi_llama1b-strategyqa/projection_scores.jsonl"
export CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
CONFIG="configs/rq3/superposition_codi_llama1b_strategyqa.yaml"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCHER="torchrun --nproc_per_node=${NPROC}"
  DIST_FLAGS="--distributed --dist_url ${DIST_URL} --dist_backend ${DIST_BACKEND}"
else
  LAUNCHER="python"
  DIST_FLAGS=""
fi

${LAUNCHER} experiments/rq3/stage3_projection_analysis.py \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --probes_jsonl "${PROBE_DIR}/probes.jsonl" \
  --output_jsonl "${OUT_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  ${DIST_FLAGS} \
  --config_path "${CONFIG}" \
  --early_steps 1,2,3,4,5 \
  --p_mode "${P_MODE:-sigmoid}" \
  --tau "${P_TAU:-10.0}"

