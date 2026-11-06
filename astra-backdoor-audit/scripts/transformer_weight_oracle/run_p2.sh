#!/usr/bin/env bash
# Run Phase-P2 Astra cells sequentially (CELLS="..." override).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
export WSP_PYTHON="${WSP_PYTHON:-$(cd "${AUDIT_ROOT}/.." && pwd)/pi-weight-space-programming/.venv/bin/python}"
export AUDIT_ROOT
# Default: scale d8/16/32 × clean/pos0 × seeds 0-2 + phrase × seeds 0-2 + clean_phrase × 0-2
# (+ depth if cells exist)
CELLS=${CELLS:-"\
tf_scale_d8_clean_strict tf_scale_d8_clean_s1_strict tf_scale_d8_clean_s2_strict \
tf_scale_d8_pos0_strict tf_scale_d8_pos0_s1_strict tf_scale_d8_pos0_s2_strict \
tf_scale_d16_clean_strict tf_scale_d16_clean_s1_strict tf_scale_d16_clean_s2_strict \
tf_scale_d16_pos0_strict tf_scale_d16_pos0_s1_strict tf_scale_d16_pos0_s2_strict \
tf_scale_d32_clean_strict tf_scale_d32_clean_s1_strict tf_scale_d32_clean_s2_strict \
tf_scale_d32_pos0_strict tf_scale_d32_pos0_s1_strict tf_scale_d32_pos0_s2_strict \
tf_phrase_strict tf_phrase_s1_strict tf_phrase_s2_strict \
tf_clean_phrase_strict tf_clean_phrase_s1_strict tf_clean_phrase_s2_strict \
tf_depth_d8_L2_clean_strict tf_depth_d8_L2_clean_s1_strict \
tf_depth_d8_L2_pos0_strict tf_depth_d8_L2_pos0_s1_strict"}
STATUS="${AUDIT_ROOT}/runs/matrix_status_p2.json"
echo "{}" > "$STATUS"
for c in $CELLS; do
  d="${AUDIT_ROOT}/runs/${c}"
  if [[ ! -d "$d" ]]; then
    echo "skip missing $c"
    continue
  fi
  echo "======== RUN $c ========"
  set +e
  ( cd "$d" && RUN_TITLE="Astra P2 $c" bash ./run.sh ) > "$d/run_stdout.log" 2>&1
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
echo "All P2 cells finished. status → $STATUS"
