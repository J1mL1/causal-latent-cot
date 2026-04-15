#!/usr/bin/env bash
# RQ4 matched-patch figures from JSONL (PDF + PNG + optional summary JSON).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
JSONL="${JSONL:-${ROOT}/outputs/rq4/matched_patch/gsm8k_coconut_gpt2_matched_patch.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-}"
SUMMARY_JSON="${SUMMARY_JSON:-}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${R_ENV_NAME:-latentcot-r}"

EXTRA=()
if [ -n "${OUT_PREFIX}" ]; then
  EXTRA+=(--out_prefix "${OUT_PREFIX}")
fi
if [ -n "${SUMMARY_JSON}" ]; then
  EXTRA+=(--summary_json "${SUMMARY_JSON}")
fi

Rscript "${ROOT}/r-script/plot_rq4_matched_patch.R" --jsonl "${JSONL}" "${EXTRA[@]}"
