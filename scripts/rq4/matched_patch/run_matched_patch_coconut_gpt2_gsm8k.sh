#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

export NCCL_ASYNC_ERROR_HANDLING=1

CONDA_ENV="${CONDA_ENV:-latentCoT}"
CONFIG="${CONFIG:-configs/rq4/coconut/gpt2-gsm8k-matched-patch.yaml}"
# Pair list: large (~100 disjoint @ Jaccard>=0.30) from build_matched_patch_pairs.sh;
# pilot (~60 @ 0.32): data/rq4/gsm8k_template_pairs_pilot.jsonl
PAIRS="${PAIRS:-data/rq4/gsm8k_template_pairs_large.jsonl}"
OUTPUT="${OUTPUT:-outputs/rq4/matched_patch/gsm8k_coconut_gpt2_matched_patch.jsonl}"
# All six latent steps (requires num_latent_placeholders: 6 in config)
STEPS="${STEPS:-1,2,3,4,5,6}"
SEED="${SEED:-0}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

python experiments/rq4/run_matched_arithmetic_patch.py \
  --model_name coconut \
  --config_path "${CONFIG}" \
  --pairs_path "${PAIRS}" \
  --output_path "${OUTPUT}" \
  --steps "${STEPS}" \
  --random_seed "${SEED}"
