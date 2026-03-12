#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
INPUT_DIR="${ROOT}/outputs/rq2/explicit_graph"
OUT_DIR="${ROOT}/outputs/rq2/plots/explicit_graph_grid"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

METRIC_EXPLICIT="kl_mean"
ANSWER_METRIC_EXPLICIT="delta_logp_last"
PCT="90"
TOPK="1"
MAX_STEPS="6"
THRESHOLD_MODE="max_ratio"
MAX_RATIO="0.1"

mkdir -p "${OUT_DIR}"

declare -a INPUTS=(
  "commonsenseqa_cot_gpt2_explicit_graph.jsonl"
  "commonsenseqa_cot_llama1b_explicit_graph.jsonl"
  "commonsenseqa_cot_qwen3_4b_explicit_graph.jsonl"
)

declare -a LABELS=(
  "(a) CoT-GPT2"
  "(b) CoT-Llama3-1B"
  "(c) CoT-Qwen3-4B"
)

for INPUT in "${INPUTS[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Missing input: ${INPUT_PATH}" >&2
    exit 1
  fi
  PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC_EXPLICIT}"
  Rscript "${ROOT}/r-script/plot_explicit_causal_graph.R" \
    --input "${INPUT_PATH}" \
    --metric "${METRIC_EXPLICIT}" \
    --answer_metric "${ANSWER_METRIC_EXPLICIT}" \
    --pct "${PCT}" \
    --topk "${TOPK}" \
    --threshold_mode "${THRESHOLD_MODE}" \
    --max_ratio "${MAX_RATIO}" \
    --max_steps "${MAX_STEPS}" \
    --out_dir "${OUT_DIR}" \
    --prefix "${PREFIX}" \
    --title_label ""
done

GRAPH_DIR="${OUT_DIR}/plot_img/${METRIC_EXPLICIT}"
declare -a GRAPH_PNGS=()
for INPUT in "${INPUTS[@]}"; do
  PREFIX="$(basename "${INPUT_DIR}/${INPUT%.*}")_${METRIC_EXPLICIT}"
  GRAPH_PNGS+=("${GRAPH_DIR}/${PREFIX}_explicit_graph.png")
done

  python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
    --inputs "${GRAPH_PNGS[@]}" \
    --out_path "${OUT_DIR}/commonsenseqa_explicit_graph_grid.pdf" \
    --ncol 3 \
  --pad 1 \
    --label_pos "bottom-center" \
  --label_band 140 \
  --label_size 120 \
  --labels "${LABELS[@]}"
