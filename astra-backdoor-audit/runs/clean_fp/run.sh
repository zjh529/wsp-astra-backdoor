#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p models/input models/output logs results
export WSP_PYTHON="${WSP_PYTHON:-/workspace/wsp/pi-weight-space-programming/.venv/bin/python}"
pi install -l /workspace/wsp/pi-weight-space-programming -a >/dev/null
echo "======================================================================"
echo "🚀 Astra Exp.D — Clean TinyMLP false-positive baseline (pt-only)"
echo "   Model: openai/gpt-6-astra | Effort: medium | Pass budget: 5"
echo "======================================================================"
pi -a -p \
  --provider openai \
  --model gpt-6-astra \
  --wsp-effort medium \
  --wsp-pass-budget 5 \
  --wsp-log-dir "${SCRIPT_DIR}/logs" \
  "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"
echo "✅ Pi session finished → ${SCRIPT_DIR}/session.log"
