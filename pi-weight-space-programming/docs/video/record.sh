#!/usr/bin/env bash
# Script to record or regenerate the Pi Weight-Space Programming demo video using Charm VHS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v vhs &>/dev/null; then
    echo "Error: 'vhs' is not installed. Install via: brew install vhs"
    exit 1
fi

echo "📥 Ensuring Experiment 1 weight artifacts are cached..."
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    "${REPO_ROOT}/.venv/bin/python" evals/run_paper_evals.py --exp 1 --download-only
else
    uv run --with numpy python3 evals/run_paper_evals.py --exp 1 --download-only
fi

echo "🎬 Recording Pi Weight-Space Programming demo via VHS..."
vhs "${SCRIPT_DIR}/demo.tape"

echo "✅ Demo recorded successfully:"
ls -lh "${SCRIPT_DIR}/demo.mp4" "${SCRIPT_DIR}/demo.gif"
