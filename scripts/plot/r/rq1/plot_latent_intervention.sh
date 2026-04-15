#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

INPUT="${PROJECT_ROOT}/outputs/rq1/intervention/gsm8k_coconut_mistral.jsonl"

# plot intervention
Rscript r-script/analyze_intervention_jsonl.R \
    --path "${INPUT}" \
    --out_dir outputs/rq1/plots/intervention
