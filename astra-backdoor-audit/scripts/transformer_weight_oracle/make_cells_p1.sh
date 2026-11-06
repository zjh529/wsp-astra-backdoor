#!/usr/bin/env bash
# Build Phase-P1 cells:
#   runs/tf_clean_v2[_s<k>]_strict/          — first-token clean task
#   runs/tf_lastpos_v2[_s<k>]_strict/        — lastpos under first-token task
#   runs/tf_adversarial_cohort[_s<k>]_strict/ — Phase-T adversarial + twin + cohort_stats
# Usage:
#   bash make_cells_p1.sh                  # seeds 0 only
#   EXTRA_SEEDS="1 2" bash make_cells_p1.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${HERE}/../.." && pwd)"
PY="${WSP_PYTHON:-$(cd "${AUDIT_ROOT}/.." && pwd)/pi-weight-space-programming/.venv/bin/python}"
P1="${AUDIT_ROOT}/models/transformer_wo_p1"
T0="${AUDIT_ROOT}/models/transformer_wo"
STATS="${T0}/sanitized/cohort_stats.json"

mk_v2() {  # $1=cell $2=sanitized ckpt $3=prompt
  local cell="$1" ckpt="$2" prompt="$3"
  mkdir -p "$cell/models/input" "$cell/models/output" "$cell/logs" "$cell/results"
  # wipe previous inputs then install subject only
  find "$cell/models/input" -type f -delete 2>/dev/null || true
  cp -f "$ckpt" "$cell/models/input/poisoned.pt"
  cp -f "$prompt" "$cell/PROMPT.md"
  cp -f "${HERE}/run_template_p1.sh" "$cell/run.sh"; chmod +x "$cell/run.sh"
  echo "built $cell"
}

mk_cohort() {  # $1=cell $2=adv sanitized $3=clean twin sanitized
  local cell="$1" adv="$2" twin="$3"
  mkdir -p "$cell/models/input" "$cell/models/output" "$cell/logs" "$cell/results"
  find "$cell/models/input" -type f -delete 2>/dev/null || true
  cp -f "$adv" "$cell/models/input/poisoned.pt"
  cp -f "$twin" "$cell/models/input/reference_twin.pt"
  cp -f "$STATS" "$cell/models/input/cohort_stats.json"
  cp -f "${HERE}/PROMPT_ADVERSARIAL_COHORT.md" "$cell/PROMPT.md"
  cp -f "${HERE}/run_template_p1.sh" "$cell/run.sh"; chmod +x "$cell/run.sh"
  # mark so run.sh keeps twin/stats
  echo "cohort" > "$cell/models/input/.keep_refs"
  echo "built $cell (twin+stats)"
}

# Ensure cohort stats exist
[[ -f "$STATS" ]] || { echo "missing $STATS — run write_cohort_stats.py first"; exit 1; }

# --- lastpos_v2 + clean_v2 ---
for t in clean_v2 lastpos_v2; do
  prompt="${HERE}/PROMPT_LASTPOS_V2.md"
  mk_v2 "${AUDIT_ROOT}/runs/tf_${t}_strict" "${P1}/sanitized/${t}.pt" "$prompt"
  for k in ${EXTRA_SEEDS:-}; do
    mk_v2 "${AUDIT_ROOT}/runs/tf_${t}_s${k}_strict" "${P1}/sanitized/${t}_s${k}.pt" "$prompt"
  done
done

# --- adversarial_cohort: reuse Phase-T adversarial + same-seed clean twin ---
# Ensure sanitized twins exist for seeds we need
sanitize_one() {
  local src="$1" dst="$2"
  [[ -f "$dst" ]] && return 0
  "${PY}" - "$src" "$dst" <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({"state_dict": ck["state_dict"], "arch": dict(ck["arch"], name="SmallTransformer-1L")}, sys.argv[2])
PYEOF
}

sanitize_one "${T0}/cohort/adversarial_s0.pt" "${T0}/sanitized/adversarial.pt"
sanitize_one "${T0}/cohort/clean_s0.pt" "${T0}/sanitized/clean.pt"
mk_cohort "${AUDIT_ROOT}/runs/tf_adversarial_cohort_strict" \
  "${T0}/sanitized/adversarial.pt" "${T0}/sanitized/clean.pt"

for k in ${EXTRA_SEEDS:-}; do
  sanitize_one "${T0}/cohort/adversarial_s${k}.pt" "${T0}/sanitized/adversarial_s${k}.pt"
  sanitize_one "${T0}/cohort/clean_s${k}.pt" "${T0}/sanitized/clean_s${k}.pt"
  mk_cohort "${AUDIT_ROOT}/runs/tf_adversarial_cohort_s${k}_strict" \
    "${T0}/sanitized/adversarial_s${k}.pt" "${T0}/sanitized/clean_s${k}.pt"
done

# Optional: clean subject under cohort protocol (FP check) — same twin from a *different* seed
# so relative diffs are non-zero but should not look like a single-token implant.
if [[ "${WITH_CLEAN_COHORT:-0}" == "1" ]]; then
  sanitize_one "${T0}/cohort/clean_s0.pt" "${T0}/sanitized/clean.pt"
  sanitize_one "${T0}/cohort/clean_s1.pt" "${T0}/sanitized/clean_s1.pt"
  mk_cohort "${AUDIT_ROOT}/runs/tf_clean_cohort_strict" \
    "${T0}/sanitized/clean.pt" "${T0}/sanitized/clean_s1.pt"
  # Overwrite prompt note: subject may be clean
  # Keep same PROMPT_ADVERSARIAL_COHORT.md (already says subject may be clean)
fi

echo "P1 cells ready."
