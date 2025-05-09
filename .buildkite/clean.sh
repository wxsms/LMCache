#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Get all CUDA compute PIDs (CSV output, no header, no units)
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')

if [[ -z "$pids" ]]; then
  echo "✔ No GPU processes found."
  exit 0
fi

echo "The following GPU processes will be terminated:"
echo "$pids" | tr ' ' '\n'

# Iterate and force‑kill each PID
for pid in $pids; do
  if kill -0 "$pid" &>/dev/null; then
    echo "→ Killing PID $pid"
    kill -9 "$pid"
  else
    echo "⚠ PID $pid does not exist or has already exited"
  fi
done

echo "Done."
