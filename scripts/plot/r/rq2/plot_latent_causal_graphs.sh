#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate ${R_ENV_NAME:-latentcot-r}

CAUSAL_INPUT="${PROJECT_ROOT}/outputs/rq2/latent_graph/gsm8k_codi_qwen3_4b_latent_graph.jsonl"
CAUSAL_OUT_DIR="outputs/rq2/plots/latent_causal_graphs"
METRIC_LATENT_LIST=("kl_mean" "kl_logit_ht" "grad_logprob")
ANSWER_METRIC_LATENT="delta_logp_final_token"
PCT="0"
TOPK="0"
MAX_STEPS="6"

# Metrics:
# - METRIC_LATENT: kl_mean | kl_logit_ht | delta_logp_final_token | teacher_forced_delta_sum | grad_logprob
# - ANSWER_METRIC_LATENT: delta_logp_final_token | teacher_forced_delta_sum

# plot latent causal graphs for each KL metric
for METRIC_LATENT in "${METRIC_LATENT_LIST[@]}"; do
    PREFIX="$(basename "${CAUSAL_INPUT%.*}")_${METRIC_LATENT}"
    Rscript r-script/plot_latent_causal_graph.R \
        --input "${CAUSAL_INPUT}" \
        --metric "${METRIC_LATENT}" \
        --answer_metric "${ANSWER_METRIC_LATENT}" \
        --pct "${PCT}" \
        --topk "${TOPK}" \
        --max_steps "${MAX_STEPS}" \
        --out_dir "${CAUSAL_OUT_DIR}" \
        --prefix "${PREFIX}"
done
