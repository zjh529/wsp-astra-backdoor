#!/usr/bin/env bash
# Exp.1-style Astra blind backdoor audit (native weight reading; fwd budget ≤5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results"

export WSP_PYTHON="${WSP_PYTHON:-/workspace/wsp/pi-weight-space-programming/.venv/bin/python}"

echo "======================================================================"
echo "🚀 Astra Exp.1-style WSP Backdoor Audit"
echo "   Model: openai/gpt-6-astra | Effort: medium | Pass budget: 5"
echo "   Prompt: @PROMPT.md"
echo "======================================================================"

pi -a -p \
  --provider openai \
  --model gpt-6-astra \
  --wsp-effort medium \
  --wsp-pass-budget 5 \
  --wsp-log-dir "${SCRIPT_DIR}/logs" \
  "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"

echo ""
echo "======================================================================"
echo "✅ Pi session finished"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output            : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
