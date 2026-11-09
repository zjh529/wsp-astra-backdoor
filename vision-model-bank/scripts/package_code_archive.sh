#!/usr/bin/env bash
# Code archive only: no datasets, virtualenvs, checkpoints, or run outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$ROOT")"
OUT="${1:-/opt/cursor/artifacts/vision-model-bank-deploy.tar.gz}"
mkdir -p "$(dirname "$OUT")"
tar -C "$(dirname "$ROOT")" -czf "$OUT" \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'runs' \
  --exclude 'exports' \
  --exclude 'data' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.tar.gz' \
  "$NAME"
echo "$OUT"
