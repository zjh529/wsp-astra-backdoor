#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p models/input models/output logs results
find models/input -type f ! -name 'poisoned.pt' -delete 2>/dev/null || true
export WSP_PYTHON="${WSP_PYTHON:-/workspace/wsp/pi-weight-space-programming/.venv/bin/python}"
export WSP_WEIGHT_ONLY=1
pi install -l /workspace/wsp/pi-weight-space-programming -a >/dev/null
echo "======================================================================"
echo "🔒 Astra Exp.B-scattered-strict — MNIST MLP scattered BadNet (WEIGHT-ONLY)"
echo "   Mode: weight-only | Pass budget: 0 | WSP_WEIGHT_ONLY=1"
echo "   Model: openai/gpt-6-astra | Effort: medium"
echo "======================================================================"
pi -a -p \
  --provider openai \
  --model gpt-6-astra \
  --wsp-mode weight-only \
  --wsp-effort medium \
  --wsp-pass-budget 0 \
  --wsp-model-path "${SCRIPT_DIR}/models/input/poisoned.pt" \
  --wsp-log-dir "${SCRIPT_DIR}/logs" \
  "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"
echo "✅ Pi session finished → ${SCRIPT_DIR}/session.log"
AUDIT_ROOT="/workspace/wsp/astra-backdoor-audit"
PYTHON="${WSP_PYTHON:-/workspace/wsp/pi-weight-space-programming/.venv/bin/python}"
mkdir -p "${SCRIPT_DIR}/results" "${SCRIPT_DIR}/models/output"
if [[ ! -f "${SCRIPT_DIR}/models/output/report.json" && -f "${SCRIPT_DIR}/.wsp-work/models/output/report.json" ]]; then
  cp -f "${SCRIPT_DIR}/.wsp-work/models/output/report.json" "${SCRIPT_DIR}/models/output/report.json"
fi
"${PYTHON}" "${AUDIT_ROOT}/tools/scan_weight_only_compliance.py" \
  "${SCRIPT_DIR}/logs" -o "${SCRIPT_DIR}/results/compliance.json"
"${PYTHON}" "${AUDIT_ROOT}/tools/validate_report.py" \
  "${SCRIPT_DIR}/models/output/report.json" -o "${SCRIPT_DIR}/results/validate_report.json"
echo "✅ Compliance + report validation OK"
