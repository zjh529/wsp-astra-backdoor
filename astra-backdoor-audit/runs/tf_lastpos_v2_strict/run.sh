#!/usr/bin/env bash
# Phase-P1 strict weight-only Astra audit cell (preserves reference_twin / cohort_stats)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="${AUDIT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO="${REPO:-$(cd "${AUDIT_ROOT}/.." && pwd)}"
WSP_EXT="${WSP_EXT:-${REPO}/pi-weight-space-programming}"
cd "${SCRIPT_DIR}"
mkdir -p models/input models/output logs results
# Keep subject + optional cohort refs; delete anything else
if [[ -f models/input/.keep_refs ]]; then
  find models/input -type f ! -name 'poisoned.pt' ! -name 'reference_twin.pt' ! -name 'cohort_stats.json' ! -name '.keep_refs' -delete 2>/dev/null || true
else
  find models/input -type f ! -name 'poisoned.pt' -delete 2>/dev/null || true
fi
export WSP_PYTHON="${WSP_PYTHON:-${WSP_EXT}/.venv/bin/python}"
export WSP_WEIGHT_ONLY=1
pi install -l "${WSP_EXT}" -a >/dev/null
TITLE="${RUN_TITLE:-Astra Phase-P1 transformer weight-only audit}"
echo "======================================================================"
echo "🔒 ${TITLE}"
echo "   Mode: weight-only | Pass budget: 0 | WSP_WEIGHT_ONLY=1 | gpt-6-astra medium"
echo "   inputs: $(ls models/input)"
echo "======================================================================"
START=$(date +%s)
set +e
timeout "${PI_TIMEOUT:-1500}" pi -a -p \
  --provider openai \
  --model gpt-6-astra \
  --wsp-mode weight-only \
  --wsp-effort medium \
  --wsp-pass-budget 0 \
  --wsp-model-path "${SCRIPT_DIR}/models/input/poisoned.pt" \
  --wsp-log-dir "${SCRIPT_DIR}/logs" \
  "Follow the instructions in @PROMPT.md" 2>&1 | tee "${SCRIPT_DIR}/session.log"
PI_EC=${PIPESTATUS[0]}
set -e
echo "{\"pi_exit\": ${PI_EC}, \"wall_s\": $(( $(date +%s) - START ))}" > "${SCRIPT_DIR}/results/timing.json"
echo "Pi exit=${PI_EC} → ${SCRIPT_DIR}/session.log"
PYTHON="${WSP_PYTHON}"
if [[ ! -f "${SCRIPT_DIR}/models/output/report.json" && -f "${SCRIPT_DIR}/.wsp-work/models/output/report.json" ]]; then
  cp -f "${SCRIPT_DIR}/.wsp-work/models/output/report.json" "${SCRIPT_DIR}/models/output/report.json"
fi
"${PYTHON}" "${AUDIT_ROOT}/tools/scan_weight_only_compliance.py" \
  "${SCRIPT_DIR}/logs" -o "${SCRIPT_DIR}/results/compliance.json"
"${PYTHON}" "${AUDIT_ROOT}/tools/validate_report.py" \
  "${SCRIPT_DIR}/models/output/report.json" -o "${SCRIPT_DIR}/results/validate_report.json"
cp -f "${SCRIPT_DIR}/models/output/report.json" "${SCRIPT_DIR}/results/report.json"
echo "✅ Compliance + report validation OK"
exit ${PI_EC}
