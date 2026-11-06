#!/usr/bin/env bash
# Run all Phase-P1 Astra cells sequentially (or CELLS="..." override).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
export WSP_PYTHON="${WSP_PYTHON:-$(cd "${AUDIT_ROOT}/.." && pwd)/pi-weight-space-programming/.venv/bin/python}"
export AUDIT_ROOT
CELLS=${CELLS:-"tf_clean_v2_strict tf_clean_v2_s1_strict tf_clean_v2_s2_strict tf_lastpos_v2_strict tf_lastpos_v2_s1_strict tf_lastpos_v2_s2_strict tf_adversarial_cohort_strict tf_adversarial_cohort_s1_strict tf_adversarial_cohort_s2_strict tf_clean_cohort_strict"}
STATUS="${AUDIT_ROOT}/runs/matrix_status_p1.json"
echo "{}" > "$STATUS"
for c in $CELLS; do
  d="${AUDIT_ROOT}/runs/${c}"
  if [[ ! -d "$d" ]]; then
    echo "skip missing $c"
    continue
  fi
  echo "======== RUN $c ========"
  set +e
  ( cd "$d" && RUN_TITLE="Astra P1 $c" bash ./run.sh ) > "$d/run_stdout.log" 2>&1
  ec=$?
  set -e
  "${WSP_PYTHON}" - "$STATUS" "$c" "$ec" <<'PY'
import json,sys
from pathlib import Path
st=json.loads(Path(sys.argv[1]).read_text())
st[sys.argv[2]]={"exit": int(sys.argv[3])}
Path(sys.argv[1]).write_text(json.dumps(st, indent=2))
PY
  echo "exit=$ec for $c"
done
echo "All P1 cells finished. status → $STATUS"
