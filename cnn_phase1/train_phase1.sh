#!/usr/bin/env bash
# Train the phase-1 pairs. Does not modify CUDA_VISIBLE_DEVICES and does not
# kill other jobs. Pass --out and any other cnn_phase1 train arguments.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${TVENV:-${TORCH_PYTHON:-python3}}"
exec "$PY" -m cnn_phase1 train "$@"
