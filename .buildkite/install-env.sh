#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Public CI instal­lation script – safe to commit and share.
# Creates (or re‑uses) a Python 3.10 venv in the project workspace.
# No absolute or private paths are hard‑coded.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Name of the virtual‑env directory (relative to the repo root)
VENV_DIR="buildkite"

# Python interpreter to use
PYTHON_BIN="/usr/bin/python3.10"

# CUDA version (if you’re using CUDA‑based packages; otherwise omit)
CUDA_VERSION="12.1"

if [[ -d "$VENV_DIR" ]]; then
  echo "⟳ Using existing venv: $(pwd)/$VENV_DIR"
else
  echo "⚙️  Creating venv with Python 3.10 at: $(pwd)/$VENV_DIR"
  # use uv for fast venv creation
  uv venv --python "$PYTHON_BIN" "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Confirm we’re on the right interpreter
echo "→ python: $(which python)"
echo "→ pip:    $(which pip)"

# Bootstrap pip inside the venv (in case it’s missing)
python -m ensurepip --upgrade

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install project dependencies
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pip install coverage

# (Optional) Export CUDA variables if needed
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# List installed packages for debugging
echo "📦 Installed packages in venv:"
pip freeze
