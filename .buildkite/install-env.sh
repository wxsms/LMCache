#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
VENV_DIR="buildkite"        
CUDA_VERSION=12.1           

# ─── Create venv ──────────────────────────────────────────────────────────────
if [[ -d "${VENV_DIR}" ]]; then
  echo "Skipping venv creation: '${VENV_DIR}' already exists."
else
  uv venv "${VENV_DIR}"
fi

# ─── CUDA paths ───────────────────────────────────────────────────────────────
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# ─── Install packages ─────────────────────────────────────────────────────────
set -x
uv pip install -r requirements.txt
uv pip install -r requirements-test.txt
uv pip install coverage
set +x

echo "Current environment packages:"
"${VENV_DIR}/bin/pip" freeze
