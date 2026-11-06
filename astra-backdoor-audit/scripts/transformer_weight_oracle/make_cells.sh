#!/usr/bin/env bash
# Build runs/tf_<threat>_strict/ (seed 0) and optional runs/tf_<threat>_s<k>_strict/ cells from sanitized ckpts.
# Usage: bash make_cells.sh            # canonical seed-0 cells
#        EXTRA_SEEDS="1 2" bash make_cells.sh   # also seed-k cells (sanitizes cohort/<t>_s<k>.pt)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${WSP_PYTHON:-$(cd "${AUDIT_ROOT}/.." && pwd)/pi-weight-space-programming/.venv/bin/python}"
MW="${AUDIT_ROOT}/models/transformer_wo"
mk() {  # $1=cell dir $2=sanitized ckpt
  mkdir -p "$1/models/input" "$1/models/output" "$1/logs" "$1/results"
  cp -f "$2" "$1/models/input/poisoned.pt"
  cp -f "${HERE}/PROMPT_TEMPLATE.md" "$1/PROMPT.md"
  cp -f "${HERE}/run_template.sh" "$1/run.sh"; chmod +x "$1/run.sh"
  echo "built $1"
}
for t in clean pos0 lastpos stealth adversarial; do
  mk "${AUDIT_ROOT}/runs/tf_${t}_strict" "${MW}/sanitized/${t}.pt"
  for k in ${EXTRA_SEEDS:-}; do
    "${PY}" - "$MW/cohort/${t}_s${k}.pt" "$MW/sanitized/${t}_s${k}.pt" <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({"state_dict": ck["state_dict"], "arch": dict(ck["arch"], name="SmallTransformer-1L")}, sys.argv[2])
PYEOF
    mk "${AUDIT_ROOT}/runs/tf_${t}_s${k}_strict" "${MW}/sanitized/${t}_s${k}.pt"
  done
done
