#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

INPUT_DIR="${PROJECT_ROOT}/outputs/rq2/latent_graph"
OUT_DIR="${PROJECT_ROOT}/outputs/rq2/plots/latent_causal_graphs_0.1ratio"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

METRIC_LATENT_LIST=("kl_mean" "kl_logit_ht" "grad_logprob")
ANSWER_METRIC_LATENT="delta_logp_final_token"
PCT="90"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0.1"

# Metrics:
# - METRIC_LATENT: kl_mean | kl_logit_ht | delta_logp_final_token | teacher_forced_delta_sum | grad_logprob
# - ANSWER_METRIC_LATENT: delta_logp_final_token | teacher_forced_delta_sum
# Optional args for r-script/plot_latent_causal_graph.R:
# - --pct <num>: percentile threshold (default 70)
# - --topk <int>: keep top-k edges per node (default 2)
# - --topk_direction <in|out>: top-k by incoming or outgoing edges (default out)
# - --max_steps <int>: truncate steps (default 6)
# - --out_dir <path>: output directory (default: dirname(input))
# - --prefix <name>: output prefix (default: input basename)
# - --title_label <str>: custom plot title (default: auto)
# - --threshold_mode <percentile|max_ratio>: threshold strategy (default percentile)
# - --max_ratio <num>: ratio for max-based threshold (default 0.9)

shopt -s nullglob
for CAUSAL_INPUT in "${INPUT_DIR}"/*.jsonl; do
  for METRIC_LATENT in "${METRIC_LATENT_LIST[@]}"; do
    PREFIX="$(basename "${CAUSAL_INPUT%.*}")_${METRIC_LATENT}"
    Rscript r-script/plot_latent_causal_graph.R \
      --input "${CAUSAL_INPUT}" \
      --metric "${METRIC_LATENT}" \
      --answer_metric "${ANSWER_METRIC_LATENT}" \
      --pct "${PCT}" \
      --topk "${TOPK}" \
      --topk_direction in \
      --threshold_mode "${THRESHOLD_MODE}" \
      --max_ratio "${MAX_RATIO}" \
      --max_steps "${MAX_STEPS}" \
      --out_dir "${OUT_DIR}" \
      --prefix "${PREFIX}"
  done
done
