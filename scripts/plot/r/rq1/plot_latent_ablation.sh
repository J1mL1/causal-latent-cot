#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate latentcot-r

INPUT="${PROJECT_ROOT}/outputs/rq1/ablation/gsm8k_coconut_qwen3_4b.jsonl"

# plot ablation
Rscript r-script/analyze_ablation_jsonl.R \
    --path "${INPUT}" \
    --out_dir outputs/rq1/plots/ablation
