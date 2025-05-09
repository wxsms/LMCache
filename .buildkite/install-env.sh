#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

VENV_DIR="buildkite"        
CUDA_VERSION=12.1           

if [[ -d "${VENV_DIR}" ]]; then
  echo "Skipping venv creation: '${VENV_DIR}' already exists."
else
  uv venv "${VENV_DIR}"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# CUDA
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

set -x
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install coverage
set +x

echo "Current environment packages:"
pip freeze
