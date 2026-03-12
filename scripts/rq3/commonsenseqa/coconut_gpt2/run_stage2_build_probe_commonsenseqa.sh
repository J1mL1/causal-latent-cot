#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

INPUT_DIR="outputs/rq3/coconut_gpt2-commonsenseqa/ambiguous"
OUTPUT_DIR="outputs/rq3/coconut_gpt2-commonsenseqa/probes"

python experiments/rq3/stage2_build_probe.py \
  --samples_jsonl "${INPUT_DIR}/ambiguous_samples.jsonl" \
  --traj_jsonl "${INPUT_DIR}/ambiguous_trajectories.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --probe_step final \
  --method mean_sub
