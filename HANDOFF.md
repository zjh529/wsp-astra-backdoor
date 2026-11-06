# Handoff — wsp-astra-backdoor

π·WSP + GPT‑6 Astra **weight-only** backdoor audit line for collaborators continuing experiments on other machines.

Repo: https://github.com/zjh529/wsp-astra-backdoor

## What this is

Protocol hardening → soft vs hard controls → CIFAR / multi-family / multi-seed / transformer matrix.

Paper claim to keep: **zero-shot Astra native weight-only backdoor reading** with **subject forwards = 0**. Capability table: `RESULTS_CAPABILITY.md`. Do not center traditional-detector bake-offs in the main story.

## Layout

| Path | Start here | Role |
|------|------------|------|
| `astra-backdoor-audit/` | `runs/RESULTS_PHASE_{B,C}.md`, `RESULTS_WEIGHT_ONLY_STRICT.md` | Main Astra weight-only protocol, matrix runners, GT, small BadNet ckpts, compliance |
| `backdoor-weight-probe/` | `RESULTS_PHASE{2,3}.md`, `scripts/` | NC / activation-probe baselines (datasets not vendored) |
| `pi-weight-space-programming/` | `harness/wsp_engine.py`, `weight-only.patch`, `UPSTREAM_BASE.txt` | Upstream fork + **enforced** weight-only (`WSP_WEIGHT_ONLY`, AST bans, path allowlist, `pass_budget=0`) |

**Not in git:** `**/data/`, venvs, VLM workspaces (`vlm-wsp*`), large HF weights. Small BadNet `.pt` files live under `astra-backdoor-audit/models/`.

Upstream base for the WSP fork: see `pi-weight-space-programming/UPSTREAM_BASE.txt` (commit `6a3e93a…` of felipepenha/pi-weight-space-programming). Prefer this tree (or re-apply `weight-only.patch`) over stock upstream.

## Done so far

### Phase A — protocol
- Document `weight-only` in plugin README mode table
- Fix write allowlist for run-root `models/output/`
- Wire `tools/scan_weight_only_compliance.py` into strict `run.sh`
- Freeze `schema/report.schema.json` + `tools/validate_report.py`
- `tools/run_matrix.sh` skeleton

### Phase B — soft × hard grid
- MLP / CNN × soft (prompt) vs hard (sandbox); clean FP cells
- Same qualitative calls; `forwards_used = 0`; compliance clean

### Phase C — scale + cost
- CIFAR corner + clean twin
- Families: blended (α=0.35), target class 7, label-consistent (weak ASR≈0.10)
- Multi-seed MLP corner: 43 / 7 / 123
- NC cost contrast table (reused Phase2/3 timings)
- WaNet skipped (not cheap)

Details: `astra-backdoor-audit/runs/RESULTS_PHASE_C.md`.

### Phase T — transformer (arXiv:2610.07334 Weight Oracles threat models)

- 1-layer transformer (d=8, 2 heads, MLP 16, seq 4, vocab 1000, 10 classes); clean / pos0 / lastpos / stealth / adversarial × 8 seeds
- Statistical baselines (total L2, V/O composite): composite AUROC 1.00 pos0, 0.97 stealth, ~0.62 lastpos/adversarial
- Astra strict weight-only, 15 cells (seeds 0–2), all forwards=0 + compliant: FPR 0/3; pos0 3/3 & stealth 3/3 with exact trigger token/position/target; lastpos 0/3, adversarial 0/3
- Code `astra-backdoor-audit/scripts/transformer_weight_oracle/`, results `astra-backdoor-audit/runs/RESULTS_TRANSFORMER_WO.md`, notes `astra-backdoor-audit/docs/TRANSFORMER_WO.md`

### Phase Cap / P0 — native capability table + clean FP expand

- Unified Astra weight-only capability table across Phase A/B/C (CNN/MLP/CIFAR) + Phase T (transformer): `astra-backdoor-audit/runs/RESULTS_CAPABILITY.md` (+ `.json`)
- Narrative: prove **zero-shot gpt-6-astra** can read backdoors from weights alone (`forwards=0`); **not** a bake-off vs NC / V·O
- Clean FP expand (same strict protocol):
  - Transformer clean seeds 0–7 audited (was 0–2) → FP **0/8**
  - MNIST MLP clean: +4 seeds (11/23/55/99) → FP **0/5**
  - MNIST CNN clean: +3 seeds (11/23/99) → FP **0/4**
  - CIFAR clean: kept 1 → FP **0/1**
- Aggregates at refresh: TPR **16/22**, FPR **0/18**, all cells compliance OK / forwards=0
- Trainer: `astra-backdoor-audit/scripts/train_clean_fp_expand.py`; summarizer: `scripts/summarize_capability.py`


### Phase P1 — lastpos_v2 + adversarial_cohort (capability redesign)

- **Narrative:** still prove native zero-shot Astra weight-only reading; **not** a bake-off vs V/O / NC. Old Phase-T misses (lastpos 0/3, adversarial 0/3) kept for honesty.
- **lastpos_v2:** clean task = `label=class_map[token at pos0]` (head still reads last). Last-position trigger must *conflict* with the first-token rule. Train: `astra-backdoor-audit/scripts/transformer_weight_oracle/train_p1.py`. Astra: **3/3** detect (trig exact 3/3, tgt exact 2/3); clean_v2 FP **0/3**.
- **adversarial_cohort:** same Phase-T adversarial weights; cell ships `reference_twin.pt` (same-seed clean) + `cohort_stats.json`. Astra: **3/3** detect (trig exact 1/3, tgt exact 3/3); clean_cohort FP **0/1**.
- Harness: weight-only allowlist extended to all files under `models/input/` (twin/stats).
- Results: `astra-backdoor-audit/runs/RESULTS_P1.md` (+ appendix in `RESULTS_CAPABILITY.md`). Cells: `tf_lastpos_v2*`, `tf_adversarial_cohort*`, `tf_clean_v2*`, `tf_clean_cohort_strict`.


## Machine setup (minimal)

1. **Secrets:** OpenAI API access for `openai/gpt-6-astra`. Never commit keys.
2. **Pi + WSP:** Install from this repo’s `pi-weight-space-programming/` (patched harness), not unpatched upstream alone.
3. **Python:** torch etc. Prefer env var `WSP_PYTHON`; strip hard-coded `/workspace/wsp/...` paths in local `run.sh` copies.
4. **Datasets:** regenerate MNIST/CIFAR under `data/` with train scripts when needed (ignored by `.gitignore`).

## Reproduce one Astra cell

1. Copy an existing `astra-backdoor-audit/runs/<cell>/` pattern (especially `*_strict`).
2. Point inputs at a ckpt under `models/` (or `models/phase_c_ckpts/`).
3. Run with `--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget 0`.
4. After the session: `python tools/validate_report.py …` and `python tools/scan_weight_only_compliance.py …`.
5. Batch helper: `bash tools/run_matrix.sh <run_dir> …` or `--all-strict` (Phase B set).

## Add a new backdoor family

1. Train with `astra-backdoor-audit/scripts/train_phase_c_families.py` (or sibling train scripts).
2. Place ckpt under `models/` (keep small; no HF dumps).
3. New run directory under `runs/`; sanitize inputs (strip family/ASR metadata before audit).
4. Update `gt/*.json` with ground truth for scoring.
5. Record wall time, exit code, compliance, schema, pred vs GT in a RESULTS note.

## NC baseline

- Scripts: `backdoor-weight-probe/scripts/` (`neural_cleanse.py`, phase runners).
- Download datasets locally; do not vendor.
- Cost numbers in Phase C RESULTS may be reused; re-opt if you change the claim.

## Hard rules (comparable results)

- Only **strict weight-only** counts for the paper matrix; failed compliance = void cell.
- Reports must pass `schema/report.schema.json`.
- Keep GT isolated from the audit inputs.
- Document failure modes honestly:
  - CNN: target + UL locus yes, exact pixel RGB/size often not unique
  - Label-consistent: weight-pattern hit, behavioral ASR weak
  - Blended: recovered as bright 3×3; α nuance lost

## Suggested next work

1. ~~One-page unified capability table~~ → done (`RESULTS_CAPABILITY.md`); keep refreshed after new cells
2. P1: lastpos / adversarial protocol redesign (clean twin / structured label rule) — capability blind spots, not detector bake-off
3. Optional: small scale axis (d or depth) + one new trigger family
4. Remove remaining `/workspace` hardcodes; keep this `HANDOFF.md` + README current

## Key artifact index

| Artifact | Path |
|----------|------|
| Phase Cap / P0 capability | `astra-backdoor-audit/runs/RESULTS_CAPABILITY.md` |
| Phase B summary | `astra-backdoor-audit/runs/RESULTS_PHASE_B.md` |
| Phase C summary | `astra-backdoor-audit/runs/RESULTS_PHASE_C.md` |
| Phase T transformer | `astra-backdoor-audit/runs/RESULTS_TRANSFORMER_WO.md` |
| Strict soft×hard | `astra-backdoor-audit/runs/RESULTS_WEIGHT_ONLY_STRICT.md` |
| Report schema | `astra-backdoor-audit/schema/report.schema.json` |
| Phase C GT | `astra-backdoor-audit/gt/phase_c_ground_truth.json` |
| Phase C ckpts | `astra-backdoor-audit/models/phase_c_ckpts/` |
| WSP weight-only diff | `pi-weight-space-programming/weight-only.patch` |
