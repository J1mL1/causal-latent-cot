#!/usr/bin/env bash
# Sourced by run_stage*_gsm8k.sh after PROJECT_ROOT and RUN_SLUG are set.
# Skip completed stages when marker file exists:
#   SKIP_IF_EXISTS=1 (default) — skip if marker exists
#   SKIP_IF_EXISTS=0 — always run
#   FORCE_RERUN=1 — same as SKIP_IF_EXISTS=0

gsm8k_skip_if_file() {
  local marker="$1"
  if [[ "${SKIP_IF_EXISTS:-1}" != "1" ]] || [[ -n "${FORCE_RERUN:-}" ]]; then
    return 1
  fi
  if [[ -f "${marker}" ]]; then
    echo "[skip] already exists: ${marker}" >&2
    return 0
  fi
  return 1
}
