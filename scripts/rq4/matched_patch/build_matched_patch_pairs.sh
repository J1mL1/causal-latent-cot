#!/usr/bin/env bash
# Build pair JSONL for RQ4 matched-patch (more samples: lower Jaccard and/or --allow_overlap).
#
# Examples:
#   bash scripts/rq4/matched_patch/build_matched_patch_pairs.sh
#   NUM_PAIRS=120 MIN_JACCARD=0.29 bash scripts/rq4/matched_patch/build_matched_patch_pairs.sh
#   ALLOW_OVERLAP=1 NUM_PAIRS=200 MIN_JACCARD=0.28 bash ...   # same index may repeat across pairs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

DATASET="${DATASET:-data/gsm8k_local.jsonl}"
# Large disjoint set on local GSM8K: ~100 pairs at 0.30; ~123 at 0.29 (see builder).
NUM_PAIRS="${NUM_PAIRS:-100}"
MIN_JACCARD="${MIN_JACCARD:-0.30}"
SEED="${SEED:-0}"
OUTPUT="${OUTPUT:-data/rq4/gsm8k_template_pairs_large.jsonl}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA=()
if [[ "${ALLOW_OVERLAP:-0}" == "1" ]]; then
  EXTRA+=(--allow_overlap)
fi

python scripts/rq4/matched_patch/build_gsm8k_template_pairs.py \
  --dataset_path "${DATASET}" \
  --num_pairs "${NUM_PAIRS}" \
  --min_jaccard "${MIN_JACCARD}" \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  "${EXTRA[@]}"

echo "Use with matched-patch run: PAIRS=${OUTPUT} bash scripts/rq4/matched_patch/run_matched_patch_coconut_gpt2_gsm8k.sh"
