#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"

# Python interpreter to use
PYTHON_BIN="/usr/bin/python3.10"

# CUDA version
CUDA_VERSION="12.1"

if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.10 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements.txt
uv pip install -r requirements-test.txt
uv pip install coverage

# (Optional) Export CUDA variables if needed
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# List installed packages for debugging
echo "📦 Installed packages in venv:"
uv pip freeze
