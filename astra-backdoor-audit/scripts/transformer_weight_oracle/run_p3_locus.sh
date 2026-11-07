#!/usr/bin/env bash
# Run Phase-P3 locus Astra cells sequentially (CELLS="..." override).
# Do NOT run the full matrix on Pi-credit-constrained hosts; prefer server + Codex/Pi.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
export WSP_PYTHON="${WSP_PYTHON:-$(cd "${AUDIT_ROOT}/.." && pwd)/pi-weight-space-programming/.venv/bin/python}"
export AUDIT_ROOT
# Default matrix: clean + attn_only + mlp_only × seeds 0–2  → 9 cells
CELLS=${CELLS:-"\
tf_locus_clean_strict tf_locus_clean_s1_strict tf_locus_clean_s2_strict \
tf_locus_attn_only_strict tf_locus_attn_only_s1_strict tf_locus_attn_only_s2_strict \
tf_locus_mlp_only_strict tf_locus_mlp_only_s1_strict tf_locus_mlp_only_s2_strict"}
STATUS="${AUDIT_ROOT}/runs/matrix_status_p3_locus.json"
echo "{}" > "$STATUS"
for c in $CELLS; do
  d="${AUDIT_ROOT}/runs/${c}"
  if [[ ! -d "$d" ]]; then
    echo "skip missing $c"
    continue
  fi
  echo "======== RUN $c ========"
  set +e
  ( cd "$d" && RUN_TITLE="Astra P3 locus $c" bash ./run.sh ) > "$d/run_stdout.log" 2>&1
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
echo "All P3 locus cells finished. status → $STATUS"
