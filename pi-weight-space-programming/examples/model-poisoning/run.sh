#!/usr/bin/env bash
# Script to execute the Model Poisoning Weight-Space Programming run and out-of-harness evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/models/input" "${SCRIPT_DIR}/models/output" "${SCRIPT_DIR}/logs"

echo "======================================================================"
echo "🚀 Running Pi Weight-Space Programming (π · WSP)"
echo "   Model: google/gemini-3.8-flash | Effort: medium"
echo "   Prompt: @PROMPT.md"
echo "======================================================================"

pi -a -p \
  --model google/gemini-3.8-flash \
  --wsp-effort medium \
  --wsp-log-dir "${SCRIPT_DIR}/logs" \
  "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"

{
  echo ""
  echo "======================================================================"
  echo "🧪 Running Out-of-Harness Evaluation on models/output"
  echo "======================================================================"
} | tee -a "${SCRIPT_DIR}/session.log"

uv run python3 eval/eval_steered_model.py --model-dir models/output 2>&1 | tee -a "${SCRIPT_DIR}/session.log"

echo ""
echo "======================================================================"
echo "✅ Workflow complete!"
echo "   Session transcript: ${SCRIPT_DIR}/session.log"
echo "   Audit logs        : ${SCRIPT_DIR}/logs/"
echo "   Output model      : ${SCRIPT_DIR}/models/output/"
echo "======================================================================"
