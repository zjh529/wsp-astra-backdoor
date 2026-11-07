#!/usr/bin/env bash
# Build Phase-P3 locus cells:
#   runs/tf_locus_{clean,attn_only,mlp_only}[_s<k>]_strict/
# Usage:
#   bash make_cells_p3_locus.sh
#   EXTRA_SEEDS="1 2" bash make_cells_p3_locus.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
P3="${AUDIT_ROOT}/models/transformer_wo_p3/locus"
RUNNER="${HERE}/run_template_p1.sh"
PROMPT="${HERE}/PROMPT_LOCUS.md"

mk() {  # $1=cell $2=sanitized ckpt
  local cell="$1" ckpt="$2"
  [[ -f "$ckpt" ]] || { echo "missing ckpt $ckpt"; exit 1; }
  [[ -f "$PROMPT" ]] || { echo "missing prompt $PROMPT"; exit 1; }
  mkdir -p "$cell/models/input" "$cell/models/output" "$cell/logs" "$cell/results"
  find "$cell/models/input" -type f -delete 2>/dev/null || true
  cp -f "$ckpt" "$cell/models/input/poisoned.pt"
  cp -f "$PROMPT" "$cell/PROMPT.md"
  cp -f "$RUNNER" "$cell/run.sh"; chmod +x "$cell/run.sh"
  echo "built $cell"
}

base="${P3}/sanitized"
for t in clean attn_only mlp_only; do
  mk "${AUDIT_ROOT}/runs/tf_locus_${t}_strict" "${base}/${t}.pt"
  for k in ${EXTRA_SEEDS:-}; do
    mk "${AUDIT_ROOT}/runs/tf_locus_${t}_s${k}_strict" "${base}/${t}_s${k}.pt"
  done
done

echo "P3 locus cells ready."
