#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ROOT="${PROJECT_ROOT}"
INPUT_DIR="${ROOT}/outputs/rq2/latent_graph"
OUT_DIR="${ROOT}/outputs/rq2/plots/rq2_figures"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

METRIC="kl_mean"
MAX_STEPS="6"

mkdir -p "${OUT_DIR}/latent_heatmaps/${METRIC}"

declare -a INPUTS=(
  "commonsenseqa_coconut_gpt2_latent_causal_graph.jsonl"
  "commonsenseqa_coconut_llama1b_latent_causal_graph.jsonl"
  "commonsenseqa_coconut_qwen3_4b_latent_graph.jsonl"
  "commonsenseqa_codi_gpt2_latent_graph.jsonl"
  "commonsenseqa_codi_llama1b_latent_graph.jsonl"
  "commonsenseqa_codi_qwen3_4b_latent_graph.jsonl"
)

declare -a LABELS=(
  "(a) Coconut-GPT2"
  "(b) Coconut-Llama3-1B"
  "(c) Coconut-Qwen3-4B"
  "(d) CODI-GPT2"
  "(e) CODI-Llama3-1B"
  "(f) CODI-Qwen3-4B"
)

declare -a HEATMAPS=()
for INPUT in "${INPUTS[@]}"; do
  INPUT_PATH="${INPUT_DIR}/${INPUT}"
  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "Missing input: ${INPUT_PATH}" >&2
    exit 1
  fi
  PREFIX="$(basename "${INPUT_PATH%.*}")_${METRIC}"
  OUT_PATH="${OUT_DIR}/latent_heatmaps/${METRIC}/${PREFIX}_heatmap.png"
  Rscript "${ROOT}/r-script/plot_latent_heatmap_only.R" \
    --input "${INPUT_PATH}" \
    --metric "${METRIC}" \
    --max_steps "${MAX_STEPS}" \
    --out_path "${OUT_PATH}"
  HEATMAPS+=("${OUT_PATH}")
done

python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
  --inputs "${HEATMAPS[@]}" \
  --out_path "${OUT_DIR}/fig_rq2_1_commonsenseqa_latent_heatmaps.pdf" \
  --ncol 3 \
  --pad 2 \
  --label_pos "bottom-center" \
  --label_band 84 \
  --label_size 64 \
  --labels "${LABELS[@]}"
