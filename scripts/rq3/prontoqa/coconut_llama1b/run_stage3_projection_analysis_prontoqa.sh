#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

INPUT_DIR="outputs/rq3/coconut_llama1b-prontoqa/ambiguous"
PROBE_DIR="outputs/rq3/coconut_llama1b-prontoqa/probes"
OUT_JSONL="outputs/rq3/coconut_llama1b-prontoqa/projection_scores.jsonl"

python experiments/rq3/stage3_projection_analysis.py \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --probes_jsonl "${PROBE_DIR}/probes.jsonl" \
  --output_jsonl "${OUT_JSONL}" \
  --early_steps 1,2,3,4,5 \
  --p_mode "${P_MODE:-sigmoid}" \
  --tau "${P_TAU:-10.0}" \
