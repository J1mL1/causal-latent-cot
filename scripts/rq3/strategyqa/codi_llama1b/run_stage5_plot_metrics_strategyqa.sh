#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT_DIR="${PROJECT_ROOT}"
PROJECTION_JSONL="${PROJECTION_JSONL:-${ROOT_DIR}/outputs/rq3/codi_llama1b-strategyqa/projection_scores.jsonl}"
TF_JSONL="${TF_JSONL:-${PROJECTION_JSONL%.jsonl}_teacher_forced.jsonl}"
PROBES_JSONL="${PROBES_JSONL:-${ROOT_DIR}/outputs/rq3/codi_llama1b-strategyqa/probes/probes.jsonl}"
INTERVENTIONS_JSONL="${INTERVENTIONS_JSONL:-${ROOT_DIR}/outputs/rq3/codi_llama1b-strategyqa/interventions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/rq3/plots/rq3-metrics-codi_llama1b-strategyqa}"
TF_ARG=""
if [ -f "${TF_JSONL}" ]; then
  TF_ARG="--teacher_forced_jsonl ${TF_JSONL}"
fi

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/experiments/rq3/compute_metrics.py" \
  --projection_jsonl "${PROJECTION_JSONL}" \
  --probes_jsonl "${PROBES_JSONL}" \
  --interventions_jsonl "${INTERVENTIONS_JSONL}" \
  --p_mode "${P_MODE:-given}" \
  --out_dir "${OUTPUT_DIR}" \
  ${TF_ARG}

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

Rscript "${ROOT_DIR}/r-script/plot_rq3_metrics.R" \
  --metrics_csv "${OUTPUT_DIR}/rq3_metrics_per_step.csv" \
  --out_dir "${OUTPUT_DIR}" 