#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
INPUT_DIR="${ROOT}/outputs/rq2/latent_graph"
OUT_DIR="${ROOT}/outputs/rq2/plots/latent_graph_grid"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

METRIC_LATENT="kl_mean"
ANSWER_METRIC_LATENT="delta_logp_final_token"
PCT="90"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0.1"

mkdir -p "${OUT_DIR}"

declare -a INPUTS_CSQA=(
  "commonsenseqa_coconut_gpt2_latent_causal_graph.jsonl"
  "commonsenseqa_coconut_llama1b_latent_causal_graph.jsonl"
  "commonsenseqa_coconut_qwen3_4b_latent_graph.jsonl"
  "commonsenseqa_codi_gpt2_latent_graph.jsonl"
  "commonsenseqa_codi_llama1b_latent_graph.jsonl"
  "commonsenseqa_codi_qwen3_4b_latent_graph.jsonl"
)

declare -a INPUTS_GSM8K=(
  "gsm8k_coconut_gpt2_latent_causal_graph.jsonl"
  "gsm8k_coconut_llama1b_latent_causal_graph.jsonl"
  "gsm8k_coconut_qwen3_4b_latent_causal_graph.jsonl"
  "gsm8k_codi_gpt2_latent_causal_graph.jsonl"
  "gsm8k_codi_llama_latent_causal_graph.jsonl"
  "gsm8k_codi_qwen3_4b_latent_graph.jsonl"
)

declare -a LABELS=(
  "(a) Coconut-GPT2"
  "(b) Coconut-Llama3-1B"
  "(c) Coconut-Qwen3-4B"
  "(d) CODI-GPT2"
  "(e) CODI-Llama3-1B"
  "(f) CODI-Qwen3-4B"
)

render_set() {
  local -n inputs_ref=$1
  local out_name=$2
  for INPUT in "${inputs_ref[@]}"; do
    INPUT_PATH="${INPUT_DIR}/${INPUT}"
    if [[ ! -f "${INPUT_PATH}" ]]; then
      echo "Missing input: ${INPUT_PATH}" >&2
      exit 1
    fi
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
  for INPUT in "${inputs_ref[@]}"; do
    INPUT_PATH="${INPUT_DIR}/${INPUT}"
    PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC_LATENT}"
    GRAPH_PNGS+=("${GRAPH_DIR}/${PREFIX}_causal_graph_top${TOPK}.png")
  done

  python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
    --inputs "${GRAPH_PNGS[@]}" \
    --out_path "${OUT_DIR}/${out_name}.pdf" \
    --ncol 3 \
    --pad 1 \
    --label_pos "bottom-center" \
    --label_band 150 \
    --label_size 120 \
    --labels "${LABELS[@]}"
}

render_set INPUTS_CSQA "latent_graph_grid_commonsenseqa"
render_set INPUTS_GSM8K "latent_graph_grid_gsm8k"
