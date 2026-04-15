#!/usr/bin/env bash
# 2x2 grid of GSM8K latent causal graphs for four Coconut models (kl_mean, rebuttal figure).
# Style matches scripts/plot/r/rq2/plot_latent_graph_grid_kl_mean.sh (combine_image_grid.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
INPUT_DIR="${ROOT}/outputs/rq2/latent_graph"
OUT_DIR="${ROOT}/outputs/rq2/plots/plot_latent_graph_grid_kl_mean_for_rebuttal"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${R_ENV_NAME:-latentcot-r}"

METRIC_LATENT="kl_mean"
ANSWER_METRIC_LATENT="delta_logp_final_token"
PCT="90"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0"

mkdir -p "${OUT_DIR}"

declare -a INPUTS_GSM8K=(
  "gsm8k_coconut_qwen3_1.7b_latent_causal_graph.jsonl"
  "gsm8k_coconut_mistral_latent_causal_graph.jsonl"
  "gsm8k_coconut_r1_qwen1.5b_latent_causal_graph.jsonl"
  "gsm8k_coconut_llama3-8b_latent_causal_graph.jsonl"
)

declare -a LABELS=(
  "(a) Coconut-Qwen3-1.7B"
  "(b) Coconut-Mistral-7B"
  "(c) Coconut-DeepSeek-R1-Qwen2.5-1.5B"
  "(d) Coconut-Llama3-8B"
)

for INPUT in "${INPUTS_GSM8K[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Missing input: ${INPUT_PATH}" >&2
    exit 1
  fi
done

for INPUT in "${INPUTS_GSM8K[@]}"; do
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
for INPUT in "${INPUTS_GSM8K[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC_LATENT}"
  GRAPH_PNGS+=("${GRAPH_DIR}/${PREFIX}_causal_graph_top${TOPK}.png")
done

python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
  --inputs "${GRAPH_PNGS[@]}" \
  --out_path "${OUT_DIR}/plot_latent_graph_grid_kl_mean_for_rebuttal.pdf" \
  --ncol 2 \
  --pad 1 \
  --label_pos "bottom-center" \
  --label_band 150 \
  --label_size 120 \
  --labels "${LABELS[@]}"
