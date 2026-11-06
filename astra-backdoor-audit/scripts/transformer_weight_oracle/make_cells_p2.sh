#!/usr/bin/env bash
# Build Phase-P2 cells:
#   runs/tf_scale_d{8,16,32}_{clean,pos0}[_s<k>]_strict/
#   runs/tf_phrase[_s<k>]_strict/ + tf_clean_phrase[_s<k>]_strict/
#   optional: runs/tf_depth_d8_L2_{clean,pos0}[_s<k>]_strict/
# Usage:
#   bash make_cells_p2.sh
#   EXTRA_SEEDS="1 2" bash make_cells_p2.sh
#   WITH_DEPTH=1 EXTRA_SEEDS="1" bash make_cells_p2.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
P2="${AUDIT_ROOT}/models/transformer_wo_p2"
RUNNER="${HERE}/run_template_p1.sh"

mk() {  # $1=cell $2=sanitized ckpt $3=prompt
  local cell="$1" ckpt="$2" prompt="$3"
  [[ -f "$ckpt" ]] || { echo "missing ckpt $ckpt"; exit 1; }
  mkdir -p "$cell/models/input" "$cell/models/output" "$cell/logs" "$cell/results"
  find "$cell/models/input" -type f -delete 2>/dev/null || true
  cp -f "$ckpt" "$cell/models/input/poisoned.pt"
  cp -f "$prompt" "$cell/PROMPT.md"
  cp -f "$RUNNER" "$cell/run.sh"; chmod +x "$cell/run.sh"
  echo "built $cell"
}

# --- A) scale ---
for d in 8 16 32; do
  base="${P2}/scale/d${d}/sanitized"
  for t in clean pos0; do
    mk "${AUDIT_ROOT}/runs/tf_scale_d${d}_${t}_strict" "${base}/${t}.pt" "${HERE}/PROMPT_SCALE.md"
    for k in ${EXTRA_SEEDS:-}; do
      mk "${AUDIT_ROOT}/runs/tf_scale_d${d}_${t}_s${k}_strict" "${base}/${t}_s${k}.pt" "${HERE}/PROMPT_SCALE.md"
    done
  done
done

# --- B) phrase ---
pbase="${P2}/phrase/sanitized"
mk "${AUDIT_ROOT}/runs/tf_phrase_strict" "${pbase}/phrase.pt" "${HERE}/PROMPT_PHRASE.md"
mk "${AUDIT_ROOT}/runs/tf_clean_phrase_strict" "${pbase}/clean_phrase.pt" "${HERE}/PROMPT_PHRASE.md"
for k in ${EXTRA_SEEDS:-}; do
  mk "${AUDIT_ROOT}/runs/tf_phrase_s${k}_strict" "${pbase}/phrase_s${k}.pt" "${HERE}/PROMPT_PHRASE.md"
  mk "${AUDIT_ROOT}/runs/tf_clean_phrase_s${k}_strict" "${pbase}/clean_phrase_s${k}.pt" "${HERE}/PROMPT_PHRASE.md"
done

# --- optional depth ---
if [[ "${WITH_DEPTH:-0}" == "1" ]]; then
  dbase="${P2}/depth/d8_L2/sanitized"
  for t in clean pos0; do
    mk "${AUDIT_ROOT}/runs/tf_depth_d8_L2_${t}_strict" "${dbase}/${t}.pt" "${HERE}/PROMPT_DEPTH.md"
    for k in ${EXTRA_SEEDS:-}; do
      [[ -f "${dbase}/${t}_s${k}.pt" ]] || continue
      mk "${AUDIT_ROOT}/runs/tf_depth_d8_L2_${t}_s${k}_strict" "${dbase}/${t}_s${k}.pt" "${HERE}/PROMPT_DEPTH.md"
    done
  done
fi

echo "P2 cells ready."
