#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

INPUT_DIR="${PROJECT_ROOT}/outputs/rq2/explicit_graph"
OUT_DIR="${PROJECT_ROOT}/outputs/rq2/plots/explicit_causal_graphs_90percent"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

METRICS_EXPLICIT=("kl_mean" "kl_logit_mean_hidden")
ANSWER_METRIC_EXPLICIT="delta_logp_last"
PCT="70"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0.1"

# Metrics:
# - METRIC_EXPLICIT: kl_mean | kl_last_token | kl_logit_last_hidden | kl_logit_mean_hidden | delta_logp_seq | delta_logp_last
# - ANSWER_METRIC_EXPLICIT: delta_logp_seq | delta_logp_last
# Optional args for r-script/plot_explicit_causal_graph.R:
# - --pct <num>: percentile threshold (default 70)
# - --topk <int>: keep top-k edges per node (default 2)
# - --max_steps <int>: truncate steps (default 6)
# - --out_dir <path>: output directory (default: dirname(input))
# - --prefix <name>: output prefix (default: input basename)
# - --title_label <str>: custom plot title (default: auto)
# - --threshold_mode <percentile|max_ratio>: threshold strategy (default percentile)
# - --max_ratio <num>: ratio for max-based threshold (default 0.9)

shopt -s nullglob
for EXPLICIT_INPUT in "${INPUT_DIR}"/*.jsonl; do
  EXPLICIT_PREFIX_BASE="$(basename "${EXPLICIT_INPUT}" .jsonl)"
  for METRIC_EXPLICIT in "${METRICS_EXPLICIT[@]}"; do
    EXPLICIT_PREFIX="${EXPLICIT_PREFIX_BASE}_${METRIC_EXPLICIT}"
    Rscript r-script/plot_explicit_causal_graph.R \
      --input "${EXPLICIT_INPUT}" \
      --metric "${METRIC_EXPLICIT}" \
      --answer_metric "${ANSWER_METRIC_EXPLICIT}" \
      --pct "${PCT}" \
      --topk "${TOPK}" \
      --threshold_mode "${THRESHOLD_MODE}" \
      --max_ratio "${MAX_RATIO}" \
      --max_steps "${MAX_STEPS}" \
      --prefix "${EXPLICIT_PREFIX}" \
      --out_dir "${OUT_DIR}"
  done
done
