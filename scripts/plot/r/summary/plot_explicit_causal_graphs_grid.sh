#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq2/plots/explicit_causal_graphs_grid"

PCT="70"
TOPK="0"
MAX_STEPS="6"
METRIC="kl_mean"
ANSWER_METRIC="delta_logp_last"

INPUTS=(
  "cot_gpt2=${ROOT}/outputs/rq2/explicit_graph/gsm8k_cot_gpt2_explicit_graph.jsonl"
  "cot_llama1b=${ROOT}/outputs/rq2/explicit_graph/gsm8k_cot_llama1b_explicit_graph.jsonl"
  "cot_qwen3_1.7b=${ROOT}/outputs/rq2/explicit_graph/gsm8k_cot_qwen3_4b_explicit_graph.jsonl"
  "cot_r1_qwen1_5b=${ROOT}/outputs/rq2/explicit_graph/gsm8k_cot_r1_qwen1_5b_explicit_graph.jsonl"
)

mkdir -p "${OUT_DIR}"

GRAPH_PNGS=()
for item in "${INPUTS[@]}"; do
  label="${item%%=*}"
  path="${item#*=}"
  prefix="${label}_${METRIC}"
  case "${label}" in
    cot_gpt2)
      title="CoT SFT GPT2 Reasoning Casual Graph"
      ;;
    cot_llama1b)
      title="CoT SFT LLama3.2 1b Reasoning Casual Graph"
      ;;
    cot_qwen3_1.7b)
      title="CoT SFT Qwen3 4b Reasoning Casual Graph"
      ;;
    cot_r1_qwen1_5b)
      title="CoT SFT R1 Qwen1.5b Reasoning Casual Graph"
      ;;
    *)
      title="CoT SFT ${label} Reasoning Casual Graph"
      ;;
  esac
  Rscript "${ROOT}/r-script/plot_explicit_causal_graph.R" \
    --input "${path}" \
    --metric "${METRIC}" \
    --answer_metric "${ANSWER_METRIC}" \
    --pct "${PCT}" \
    --topk "${TOPK}" \
    --max_steps "${MAX_STEPS}" \
    --prefix "${prefix}" \
    --title_label "${title}" \
    --out_dir "${OUT_DIR}"
  GRAPH_PNGS+=("${OUT_DIR}/${prefix}_explicit_graph.png")
done

python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
  --inputs "${GRAPH_PNGS[@]}" \
  --out_path "${OUT_DIR}/explicit_cot_causal_graph_grid.png" \
  --ncol 2 \
  --pad 4 \
  --label_prefix "(" \
  --label_suffix ")" \
  --label_pos "bottom-center" \
  --label_band 28 \
  --label_size 22
