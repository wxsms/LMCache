#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
VENV_DIR="buildkite-e2e"
CUDA_VERSION=12.1

# ─── Create venv if not exists ────────────────────────────────────────────────
if [[ -d "${VENV_DIR}" ]]; then
    echo "Skipping venv creation: '${VENV_DIR}' already exists."
else
    uv venv "${VENV_DIR}"
fi

# ─── CUDA Environment Variables ───────────────────────────────────────────────
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

source buildkite/bin/activate

# ─── Install packages using uv pip ────────────────────────────────────────────
set -xe

uv pip install -r requirements.txt
uv pip install -r requirements-test.txt
uv pip install coverage

set +x
echo "Current environment packages:"
"${VENV_DIR}/bin/pip" freeze
