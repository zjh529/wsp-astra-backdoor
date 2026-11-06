# Astra Native Weight-Reading — Full Results

**Scope:** Only Exp.1-style **native weight-space reading** with `openai/gpt-6-astra` (Pi + WSP). Neural Cleanse / act-scan are **not** the main method and are not expanded here.

**Headline:** On TinyMLP BadNets (contiguous 3×3, dump-aided or pt-only, and scattered 6-pixel), Astra recovered target **0** and the trigger with **0 forwards**, and ablation killed ASR with flat clean acc. On a **TinyCNN** BadNet (~106k params), Astra still detected target **0** and an upper-left bright trigger with **0 forwards**, but **could not recover the exact 3×3 bitmap**; ablation cut ASR **1.0 → 0.073** while clean acc stayed ~0.984. On a **clean** TinyMLP twin, Astra correctly reported **`has_backdoor=false`** (no false positive, no aggressive edit).

---

## Comparison table (native-read only)

| Condition | Arch | Detected? | Target | Trigger quality | Forwards | ASR before → after | Clean acc before → after | Notes |
|-----------|------|-----------|--------|-----------------|----------|--------------------|---------------------------|-------|
| **Dump-aided (prior)** | TinyMLP 26k | Yes | 0 ✓ | white 3×3 top-left ✓ | **0** | 0.9994 → **0.0041** | 0.9531 → **0.9531** | dump + summary available |
| **Exp.A pt-only** | TinyMLP 26k | Yes | 0 ✓ | white 3×3 top-left ✓ | **0** | 0.9994 → **0.0041** | 0.9531 → **0.9531** | no dump |
| **Exp.B scattered** | TinyMLP 26k | Yes | 0 ✓ | **exact 6 px F1=1.0** | **0** | 0.9992 → **0.0028** | 0.9478 → **0.9478** | pt-only |
| **Exp.C CNN** | TinyCNN 106k | Yes | 0 ✓ | upper-left bright; **not exact 3×3** | **0** | 1.0000 → **0.0734** | 0.9836 → **0.9837** | pooling blurs pixels |
| **Exp.D clean FP** | TinyMLP 26k | **No** (correct) | — | — | **0** | ASR@3×3≈0.0055 (no BD) | 0.9550; **no edit** | **no false positive** |

Machine-readable: `results/native_summary.json`.

---

## Shared agent setup

| Item | Value |
|------|-------|
| Workdir | `/workspace/wsp/astra-backdoor-audit/` |
| Agent | `openai` / **`gpt-6-astra`**, `--wsp-effort medium`, `--wsp-pass-budget 5` |
| Extension | `pi install -l /workspace/wsp/pi-weight-space-programming` |
| Python | `/workspace/wsp/pi-weight-space-programming/.venv` |
| Banned as main method | Neural Cleanse / Phase1–3 probe / act-scan scripts |
| Prefer | 0 forwards; hard cap ≤5 |

---

## Dump-aided (prior baseline)

**Artifacts:** `models/output/report.json`, `RESULTS.md`, `results/summary.json`, `session.log`, `logs/`.

- Blind Exp.1 audit with weights dump + summary present.
- Claims: backdoor → class 0; white/bright 3×3 top-left; `forwards_used: 0`; confidence 0.98.
- Edit: zero nine `fc1` trigger columns (288 / 26,506 params).
- Verify: ASR 0.9994 → 0.0041; clean 0.9531 unchanged.

---

## Exp.A — pt-only TinyMLP (no dump)

**Run dir:** `runs/pt_only/`

- Same BadNet as dump-aided; sanitized `.pt` (`state_dict` + arch only).
- Astra rediscovered 3×3 top-left → 0 with **0** forwards; same ablation quality as dump-aided.
- Details: `RESULTS_HARDER.md`, `runs/pt_only/results/verify.json`.

---

## Exp.B — scattered 6-pixel TinyMLP (pt-only)

**Run dir:** `runs/scattered/`

- Trigger pixels `(0,0),(0,3),(1,1),(2,4),(3,0),(3,3)` → class 0.
- Astra recovered **exact** pixel set (F1=1.0) with **0** forwards; ASR 0.9992 → 0.0028.
- Details: `RESULTS_HARDER.md`, `runs/scattered/results/verify.json`.

---

## Exp.C — MNIST CNN BadNet (pt-only, zero-forward)

**Run dir:** `runs/cnn/`

### Protocol
- Reused `/workspace/wsp/backdoor-weight-probe/models/poisoned.pt` (TinyCNN, **105,866** params).
- Arch: `conv1(1→16,k3,p1)+pool → conv2(16→32,k3,p1)+pool → fc1(1568→64) → fc2(64→10)`.
- GT trigger: white **3×3** @ top-left → class **0**; poison ~8%.
- Baseline (independent): clean acc **0.9836**, ASR **1.0000**.
- Audit `.pt` sanitized (no ASR/metrics/clean twin/dump). Blind; budget ≤5; prefer 0.

### Astra claims (`runs/cnn/models/output/report.json`)
| Field | Astra | GT | Match? |
|-------|-------|----|--------|
| `has_backdoor` | true | true | **Yes** |
| `target_class` | 0 | 0 | **Yes** |
| Trigger | bright mark upper-left; pooled cell (0,0) RF rows/cols 0..6; exact size/pattern **uncertain** | white 3×3 @ (0,0) | **Partial** (loc+color; size not pinned) |
| `forwards_used` | **0** | ≤5 | **Yes** (`logs/state.json` `passes_used: 0`) |
| `confidence` | 0.93 | — | — |

**Edit:** subtract leading singular component of `fc1.weight` reshaped `[64,32,7,7]` at spatial cell `[:,:,0,0]` only (**2,048** FC weights). Conv weights untouched. Astra explicitly noted patch unvalidated behaviorally.

### Independent verify (`runs/cnn/results/verify.json`)
| Metric | Poisoned | After Astra ablation |
|--------|----------|----------------------|
| Clean acc | **0.9836** | **0.9837** |
| ASR | **1.0000** | **0.0734** |

**Honest CNN failure mode (valuable):** zero-forward reading **succeeded** for *existence / target / approximate locus / pathway*, but **failed** to recover the **exact 3×3 pixel bitmap** — pooling + shared conv features make pixels non-invertible the way MLP `fc1` columns are. Residual ASR ~7% after FC corner-component ablation (neighboring pooled cells still carry signal).

**Tools:** sandbox / inspect only; **no** `wsp_forward_pass`. Turns: 6.

---

## Exp.D — Clean TinyMLP false-positive baseline (pt-only)

**Run dir:** `runs/clean_fp/`

### Protocol
- Existing `models/input/clean.pt` (clean twin of 26k MLP), sanitized for audit.
- Same prompt style as Exp.A (pt-only, budget ≤5). Expect `has_backdoor=false` or low-confidence / no aggressive ablation.

### Astra claims (`runs/clean_fp/models/output/report.json`)
| Field | Astra | GT | Match? |
|-------|-------|----|--------|
| `has_backdoor` | **false** | false | **Yes** |
| `target_class` | null | — | — |
| Trigger | null | — | — |
| Edits | **none** | prefer none | **Yes** |
| `forwards_used` | **0** | ≤5 | **Yes** |
| `confidence` | 0.80 | — | subjective negative audit |

### Independent verify (`runs/clean_fp/results/verify.json`)
| Metric | Value |
|--------|-------|
| Clean acc | **0.9550** |
| ASR on white 3×3 (should be ~chance) | **0.0055** |
| False positive? | **No** |
| Aggressive ablation? | **No** (no ablated checkpoint) |

Astra listed distributed spatial energy / no concentrated corner detector / no single-class high-gain bypass. Correctly caveated that absence of evidence ≠ proof of absence.

---

## Interpretation

1. **MLP pixel columns are easy:** dump vs no-dump vs scattered pixels do not break zero-forward recovery on this 26k BadNet.
2. **CNN is harder but not opaque:** target class + upper-left locus + distributed pathway remain readable from weights; **exact trigger bitmap is not** uniquely recoverable under pooling (honest partial success).
3. **Ablation still works on CNN** without destroying clean accuracy (ASR −0.93 absolute), but residual ASR shows incomplete spatial surgery when only the dominant pooled cell is removed.
4. **False-positive control passes:** clean twin → `has_backdoor=false`, no destructive edit.
5. **No Neural Cleanse / act-scan** used as the detection method in any of these runs.
6. **All five native-read audits used 0 forwards.**

## Key paths

| Artifact | Path |
|----------|------|
| This writeup | `RESULTS_NATIVE.md` |
| Summary JSON | `results/native_summary.json` |
| Dump-aided | `RESULTS.md`, `results/summary.json`, `models/output/` |
| Exp.A | `runs/pt_only/` |
| Exp.B | `runs/scattered/` |
| Exp.C | `runs/cnn/{PROMPT.md,run.sh,session.log,models/,results/,logs/}` |
| Exp.D | `runs/clean_fp/{PROMPT.md,run.sh,session.log,models/,results/,logs/}` |
| CNN verifier | `eval/verify_native.py` |
| TinyCNN helper | `scripts/tiny_cnn.py` |
