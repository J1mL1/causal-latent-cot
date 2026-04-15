#!/usr/bin/env bash
# Sweep patchscope prompts for Coconut GPT-2 and CODI GPT-2 → outputs/rq4/patchscope/
# Filenames: gsm8k_{coconut|codi}_gpt2_patchscope_prompt_<label>.jsonl (+ .meta.json)
#
# Optional env: CONDA_ENV BATCH_SIZE STEPS MAX_SAMPLES ONLY_MODEL (coconut|codi)
#
# Examples:
#   ./scripts/rq4/patchscope/run_patchscope_prompt_ablation_gpt2.sh
#   MAX_SAMPLES=64 ONLY_MODEL=coconut ./scripts/rq4/patchscope/run_patchscope_prompt_ablation_gpt2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-latentCoT}"
eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA=()
if [ -n "${MAX_SAMPLES:-}" ]; then
  EXTRA+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${ONLY_MODEL:-}" ]; then
  EXTRA+=(--only_model "${ONLY_MODEL}")
fi
if [ -n "${BATCH_SIZE:-}" ]; then
  EXTRA+=(--batch_size "${BATCH_SIZE}")
fi
if [ -n "${STEPS:-}" ]; then
  EXTRA+=(--steps "${STEPS}")
fi

exec python scripts/rq4/patchscope/run_patchscope_prompt_ablation_gpt2.py \
  --output_dir "${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/rq4/patchscope}" \
  "${EXTRA[@]}" \
  "$@"
