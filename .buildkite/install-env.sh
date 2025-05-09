#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="buildkite"
CUDA_VERSION=12.1

if [[ -d "$VENV_DIR" ]]; then
  echo "Using existing venv at: $(pwd)/$VENV_DIR"
else
  echo "Creating venv in: $(pwd)/$VENV_DIR"
  uv venv "$VENV_DIR"
fi

# Activate it
source "$VENV_DIR/bin/activate"

# Sanity check
echo "→ python: $(which python)"
echo "→ pip:    $(which pip)"

# Make sure pip itself is up‑to‑date inside the venv
python -m pip install --upgrade pip

# Install into the venv
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pip install coverage

# (You can keep your CUDA exports here if you like)
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "Final venv packages:"
pip freeze
