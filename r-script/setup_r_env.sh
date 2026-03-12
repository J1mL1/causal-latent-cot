#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}"

CONDA_BIN="${CONDA_EXE:-conda}"
ENV_NAME="${R_ENV_NAME:-latentcot-r}"
export LC_ALL=C
export LANG=C

if [ ! -x "${CONDA_BIN}" ]; then
  echo "Conda not found at ${CONDA_BIN}"
  exit 1
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" -c conda-forge \
    r-base \
    r-remotes \
    r-renv \
    r-jsonlite \
    r-dplyr \
    r-tidyr \
    r-ggplot2 \
    r-scales \
    r-ggforce \
    r-igraph \
    r-tidygraph \
    r-ggraph \
    r-systemfonts \
    fontconfig \
    freetype \
    icu \
    glpk \
    libpng \
    libjpeg-turbo \
    libtiff \
    harfbuzz \
    fribidi
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_NAME}"

"${CONDA_BIN}" install -y -n "${ENV_NAME}" -c conda-forge \
  r-jsonlite \
  r-dplyr \
  r-tidyr \
  r-ggplot2 \
  r-scales \
  r-ggforce \
  r-igraph \
  r-tidygraph \
  r-ggraph \
  r-systemfonts \
  fontconfig \
  freetype \
  icu \
  glpk \
  libpng \
  libjpeg-turbo \
  libtiff \
  harfbuzz \
  fribidi
