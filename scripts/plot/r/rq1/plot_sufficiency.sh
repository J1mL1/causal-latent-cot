#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

SUFFICIENCY_INPUT="${PROJECT_ROOT}/outputs/rq1/sufficiency/coconut_llama1b.jsonl"
DATASET_NAME="gsm8k"
OUT_DIR="outputs/rq1/plots/sufficiency"

Rscript r-script/analyze_sufficiency_jsonl.R \
  --path "${SUFFICIENCY_INPUT}" \
  --dataset_name "${DATASET_NAME}" \
  --out_dir "${OUT_DIR}"
