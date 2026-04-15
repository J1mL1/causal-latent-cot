#!/usr/bin/env bash
# 2x2 grid: GSM8K latent causal graphs for GPT-2 Coconut / CoDi,
# comparing global mean vs Gaussian-h intervention (kl_mean).
# Uses r-script/plot_latent_causal_graph.R + combine_image_grid.py (same pattern as plot_latent_graph_grid_kl_mean.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
INPUT_DIR="${ROOT}/outputs/rq2/latent_graph"
OUT_DIR="${ROOT}/outputs/rq2/plots/plot_latent_graph_grid_gpt2_mean_vs_gaussian_h"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${R_ENV_NAME:-latentcot-r}"

METRIC_LATENT="kl_mean"
ANSWER_METRIC_LATENT="delta_logp_final_token"
PCT="90"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0.1"

mkdir -p "${OUT_DIR}"

# Row-major for ncol=2: row1 = Coconut (mean | Gaussian h), row2 = CoDi (mean | Gaussian h)
declare -a INPUTS=(
  "gsm8k_coconut_gpt2_latent_graph_mean.jsonl"
  "gsm8k_coconut_gpt2_latent_graph_gaussian_h.jsonl"
  "gsm8k_codi_gpt2_latent_graph_mean.jsonl"
  "gsm8k_codi_gpt2_latent_graph_gaussian_h.jsonl"
)

declare -a LABELS=(
  "(a) Coconut-GPT2 — global mean"
  "(b) Coconut-GPT2 — Gaussian h"
  "(c) CoDi-GPT2 — global mean"
  "(d) CoDi-GPT2 — Gaussian h"
)

for INPUT in "${INPUTS[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Missing input: ${INPUT_PATH}" >&2
    exit 1
  fi
done

for INPUT in "${INPUTS[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC_LATENT}"
  Rscript "${ROOT}/r-script/plot_latent_causal_graph.R" \
    --input "${INPUT_PATH}" \
    --metric "${METRIC_LATENT}" \
    --answer_metric "${ANSWER_METRIC_LATENT}" \
    --pct "${PCT}" \
    --topk "${TOPK}" \
    --topk_direction in \
    --threshold_mode "${THRESHOLD_MODE}" \
    --max_ratio "${MAX_RATIO}" \
    --max_steps "${MAX_STEPS}" \
    --out_dir "${OUT_DIR}" \
    --prefix "${PREFIX}" \
    --title_label ""
done

GRAPH_DIR="${OUT_DIR}/plot_img/${METRIC_LATENT}"
declare -a GRAPH_PNGS=()
for INPUT in "${INPUTS[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC_LATENT}"
  GRAPH_PNGS+=("${GRAPH_DIR}/${PREFIX}_causal_graph_top${TOPK}.png")
done

python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
  --inputs "${GRAPH_PNGS[@]}" \
  --out_path "${OUT_DIR}/plot_latent_graph_grid_gpt2_mean_vs_gaussian_h.pdf" \
  --ncol 2 \
  --pad 1 \
  --label_pos "bottom-center" \
  --label_band 150 \
  --label_size 120 \
  --labels "${LABELS[@]}"

echo "Wrote ${OUT_DIR}/plot_latent_graph_grid_gpt2_mean_vs_gaussian_h.pdf"
