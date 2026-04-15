#!/usr/bin/env bash
# Wrapper: R figures for matched-patch JSONL (see r-script/plot_rq4_matched_patch.R).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

export PROJECT_ROOT
exec "${PROJECT_ROOT}/scripts/plot/r/rq4/plot_matched_patch.sh"
