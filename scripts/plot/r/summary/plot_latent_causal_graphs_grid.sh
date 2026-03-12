#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

ROOT="${PROJECT_ROOT}"
OUT_DIR="${ROOT}/outputs/rq2/plots/latent_causal_graphs_grid"

PCT="70"
TOPK="0"
MAX_STEPS="6"
METRIC="kl_mean"
ANSWER_METRIC="delta_logp_final_token"

INPUTS=(
  "coconut_gpt2=${ROOT}/outputs/rq2/latent_graph/gsm8k_coconut_gpt2_latent_graph.jsonl"
  "coconut_llama1b=${ROOT}/outputs/rq2/latent_graph/gsm8k_coconut_llama1b_latent_graph.jsonl"
  "coconut_qwen3_4b=${ROOT}/outputs/rq2/latent_graph/gsm8k_coconut_qwen3_4b_latent_graph.jsonl"
  "coconut_r1_qwen1_5b=${ROOT}/outputs/rq2/latent_graph/gsm8k_coconut_r1_qwen1.5b_latent_graph.jsonl"
  "codi_gpt2=${ROOT}/outputs/rq2/latent_graph/gsm8k_codi_gpt2_latent_graph.jsonl"
  "codi_llama=${ROOT}/outputs/rq2/latent_graph/gsm8k_codi_llama_latent_graph.jsonl"
)

mkdir -p "${OUT_DIR}"

GRAPH_PNGS=()
for item in "${INPUTS[@]}"; do
  label="${item%%=*}"
  path="${item#*=}"
  prefix="${label}_${METRIC}"
  case "${label}" in
    coconut_gpt2)
      title="Coconut GPT2 Reasoning Casual Graph"
      ;;
    coconut_llama1b)
      title="Coconut LLama3.2 1b Reasoning Casual Graph"
      ;;
    coconut_qwen3_4b)
      title="Coconut Qwen3 4b Reasoning Casual Graph"
      ;;
    coconut_r1_qwen1_5b)
      title="Coconut R1 Qwen1.5b Reasoning Casual Graph"
      ;;
    codi_gpt2)
      title="Codi GPT2 Reasoning Casual Graph"
      ;;
    codi_llama)
      title="Codi LLama Reasoning Casual Graph"
      ;;
    *)
      title="Coconut ${label} Reasoning Casual Graph"
      ;;
  esac
  Rscript "${ROOT}/r-script/plot_latent_causal_graph.R" \
    --input "${path}" \
    --metric "${METRIC}" \
    --answer_metric "${ANSWER_METRIC}" \
    --pct "${PCT}" \
    --topk "${TOPK}" \
    --max_steps "${MAX_STEPS}" \
    --prefix "${prefix}" \
    --title_label "${title}" \
    --out_dir "${OUT_DIR}"
  GRAPH_PNGS+=("${OUT_DIR}/${prefix}_causal_graph_top${TOPK}.png")
done

python "${ROOT}/scripts/plot/python/utils/combine_image_grid.py" \
  --inputs "${GRAPH_PNGS[@]}" \
  --out_path "${OUT_DIR}/latent_cot_causal_graph_grid.png" \
  --ncol 3 \
  --pad 4 \
  --label_prefix "(" \
  --label_suffix ")" \
  --label_pos "bottom-center" \
  --label_band 28 \
  --label_size 22
