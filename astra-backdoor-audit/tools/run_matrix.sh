#!/usr/bin/env bash
# Launch listed Astra audit run dirs sequentially; collect status into runs/matrix_status.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${ROOT}/runs"
STATUS_OUT="${RUNS_ROOT}/matrix_status.json"
SCAN="${ROOT}/tools/scan_weight_only_compliance.py"
VALIDATE="${ROOT}/tools/validate_report.py"
PYTHON="${WSP_PYTHON:-/workspace/wsp/pi-weight-space-programming/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir_name> [more...]" >&2
  echo "  or:  $0 --all-strict" >&2
  exit 2
fi

if [[ "${1:-}" == "--all-strict" ]]; then
  set -- cnn_strict pt_only_strict scattered_strict clean_fp_strict cnn_scattered_strict cnn_clean_strict
fi

mkdir -p "${RUNS_ROOT}"
tmp="$(mktemp)"
echo '[' > "${tmp}"
first=1
overall=0

for name in "$@"; do
  run_dir="${RUNS_ROOT}/${name}"
  entry_start=$(date +%s)
  status="pending"
  exit_code=0
  compliance_path=""
  report_path=""
  validate_path=""
  note=""

  if [[ ! -d "${run_dir}" ]]; then
    status="missing"
    exit_code=127
    note="run dir not found"
    overall=1
  elif [[ ! -x "${run_dir}/run.sh" ]]; then
    status="no_run_sh"
    exit_code=127
    note="run.sh missing or not executable"
    overall=1
  else
    echo "======== MATRIX: ${name} ========"
    set +e
    ( cd "${run_dir}" && ./run.sh )
    exit_code=$?
    set -e
    # Locate report
    for cand in \
      "${run_dir}/models/output/report.json" \
      "${run_dir}/.wsp-work/models/output/report.json"; do
      if [[ -f "${cand}" ]]; then report_path="${cand}"; break; fi
    done
    if [[ -n "${report_path}" && "${report_path}" != "${run_dir}/models/output/report.json" ]]; then
      mkdir -p "${run_dir}/models/output"
      cp -f "${report_path}" "${run_dir}/models/output/report.json"
      report_path="${run_dir}/models/output/report.json"
    fi
    compliance_path="${run_dir}/results/compliance.json"
    mkdir -p "${run_dir}/results"
    set +e
    "${PYTHON}" "${SCAN}" "${run_dir}/logs" -o "${compliance_path}"
    scan_ec=$?
    set -e
    validate_path="${run_dir}/results/validate_report.json"
    if [[ -n "${report_path}" ]]; then
      set +e
      "${PYTHON}" "${VALIDATE}" "${report_path}" -o "${validate_path}"
      val_ec=$?
      set -e
    else
      val_ec=2
      note="no report.json found"
    fi
    if [[ ${exit_code} -eq 0 && ${scan_ec} -eq 0 && ${val_ec} -eq 0 ]]; then
      status="ok"
    else
      status="failed"
      overall=1
      note="pi_exit=${exit_code} scan_exit=${scan_ec} validate_exit=${val_ec}"
    fi
  fi
  entry_end=$(date +%s)
  if [[ ${first} -eq 0 ]]; then echo ',' >> "${tmp}"; fi
  first=0
  "${PYTHON}" - <<PY >> "${tmp}"
import json
print(json.dumps({
  "run": "${name}",
  "status": "${status}",
  "exit_code": ${exit_code},
  "elapsed_s": ${entry_end}-${entry_start},
  "compliance": """${compliance_path}""" or None,
  "report": """${report_path}""" or None,
  "validate": """${validate_path}""" or None,
  "note": """${note}""" or None,
}, indent=2))
PY
done

echo ']' >> "${tmp}"
"${PYTHON}" - <<PY
import json
from pathlib import Path
raw = Path("${tmp}").read_text()
# Fix concatenated objects: we wrote pretty objects separated by commas inside []
data = json.loads(raw)
Path("${STATUS_OUT}").write_text(json.dumps({
  "generated": __import__('datetime').datetime.now().astimezone().isoformat(),
  "runs": data,
}, indent=2) + "\n")
print("Wrote ${STATUS_OUT}")
PY
rm -f "${tmp}"
exit ${overall}
