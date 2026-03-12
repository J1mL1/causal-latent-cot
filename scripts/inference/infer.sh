#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CONDA_ENV="latentCoT"

eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda activate "${CONDA_ENV}"

CUDA_VISIBLE_DEVICES="3"

export PYTHONPATH="$(pwd)"

python scripts/inference/test_multiplex_infer.py \
    --config_path configs/rq1/multiplex/multiplex-1.5b-gsm8k.yaml \
    --question "Dr. Hugo Grumpus and his assistant, Igor, were preparing to perform a laboratory experiment.  Dr. Grumpus told Igor to gather 16 test tubes, 7 beakers, and 14 Petri dishes, and to place them all on the lab bench.  By accident, Igor gathered half as many test tubes as requested, two more than the number of Petri dishes requested.  And while he had picked up the correct number of beakers, he lost several on the way to the lab bench.  In total, the number of items Igor had placed on the lab bench was 29.  How many beakers did Igor lose?"