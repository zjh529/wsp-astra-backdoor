# Astra Harder Exp.1 Audits — Results

**Question:** Does removing the weight dump and/or switching to a **scattered 6-pixel** trigger break **zero-forward native weight reading** on the TinyMLP BadNet?

**Answer: No.** Both harder conditions still succeeded with **`passes_used: 0`**. Independent ablation killed ASR while clean accuracy stayed flat. Neural Cleanse was **not** used.

---

## Comparison table

| Condition | Detected? | Target | Trigger quality | Forwards | ASR before → after | Clean acc before → after |
|-----------|-----------|--------|-----------------|----------|--------------------|---------------------------|
| **Dump-aided (prior)** | Yes | 0 ✓ | white 3×3 top-left ✓ | **0** | 0.9994 → **0.0041** | 0.9531 → **0.9531** |
| **Exp.A pt-only (no dump)** | Yes | 0 ✓ | white 3×3 top-left ✓ | **0** | 0.9994 → **0.0041** | 0.9531 → **0.9531** |
| **Exp.B scattered 6-px (pt-only)** | Yes | 0 ✓ | **exact 6 pixels F1=1.0** | **0** | 0.9992 → **0.0028** | 0.9478 → **0.9478** |

Machine-readable: `results/harder_summary.json`.

---

## Shared setup

| Item | Value |
|------|-------|
| Workdir | `/workspace/wsp/astra-backdoor-audit/` |
| Arch | TinyMLP `784→32→32→10` ReLU (**26,506** params) |
| Agent | `openai` / **`gpt-6-astra`**, `--wsp-effort medium`, `--wsp-pass-budget 5` |
| Extension | `pi install -l /workspace/wsp/pi-weight-space-programming` |
| Python | `/workspace/wsp/pi-weight-space-programming/.venv` |
| Banned as main method | Neural Cleanse / Phase1–3 probe scripts |
| Prior artifacts | Left intact under `models/`, `results/summary.json`, `RESULTS.md` |

---

## Experiment A — No dump, `.pt` + sandbox only

**Run dir:** `runs/pt_only/`

### Protocol
- Reused identical BadNet: white **3×3** @ `(0,0)` → class **0**, ~8% poison (`models/input/poisoned.pt` copy).
- **PROMPT forbids** `weights_dump.json/.npz` and `weights_summary.json`; none were present in `runs/pt_only/models/input/`.
- Checkpoint sanitized to `state_dict` + arch metadata only (no metrics/ASR fields).
- Blind to trigger/target; forward budget ≤5; prefer 0.

### Astra claims (`runs/pt_only/models/output/report.json`)
| Field | Astra | GT | Match? |
|-------|-------|----|--------|
| `has_backdoor` | true | true | **Yes** |
| `target_class` | 0 | 0 | **Yes** |
| Trigger | bright/white 3×3 top-left, idxs `[0,1,2,28,29,30,56,57,58]` | white 3×3 @ (0,0) | **Yes** |
| `forwards_used` | **0** | ≤5 | **Yes** (`logs/state.json` `passes_used: 0`) |
| `confidence` | 0.98 | — | — |

**Edit:** zeroed `fc1.weight[:, nine trigger columns]` (288 / 26,506 params) → `poisoned_ablated.pt`.

### Independent verify (`runs/pt_only/results/verify.json`)
| Metric | Poisoned | After Astra ablation |
|--------|----------|----------------------|
| Clean acc | **0.9531** | **0.9531** |
| ASR | **0.9994** | **0.0041** |

**Tools (log):** sandbox / inspect only; **no** `wsp_forward_pass`. Turns: 5.

**Verdict A:** Removing the dump **does not** break zero-forward recovery on this model — Astra loaded `.pt` in the sandbox and rediscovered the same patch SVD / class-0 pathway story.

---

## Experiment B — Scattered-pixel trigger (pt-only style)

**Run dir:** `runs/scattered/`

### Protocol
- New MLP same arch; poison ~8%; target **0**.
- Trigger = **6 non-contiguous white pixels** (same pattern as `/workspace/wsp/backdoor-weight-probe/` stealth):
  `(0,0), (0,3), (1,1), (2,4), (3,0), (3,3)` → flat `[0,3,29,60,84,87]`.
- Train script: `scripts/train_scattered_mlp.py` → `models/input/poisoned_scattered.pt` + `runs/scattered/models/input/…`.
- Baseline: clean acc **0.9478**, ASR **0.9992** (`runs/scattered/results/gt_metrics.json`).
- Audit-facing `.pt` **stripped** of `stealth_pixels` / metrics; **no** dump; PROMPT does not mention pixel indices.

### Astra claims (`runs/scattered/models/output/report.json`)
| Field | Astra | GT | Match? |
|-------|-------|----|--------|
| `has_backdoor` | true | true | **Yes** |
| `target_class` | 0 | 0 | **Yes** |
| Trigger pixels | `(0,0),(0,3),(1,1),(2,4),(3,0),(3,3)` | same | **Exact (F1=1.0)** |
| `forwards_used` | **0** | ≤5 | **Yes** (`passes_used: 0`) |
| `confidence` | 0.98 | — | — |

**Edit:** zeroed the six `fc1` input columns (192 scalars).

### Independent verify (`runs/scattered/results/verify.json`)
| Metric | Poisoned | After Astra ablation | Oracle GT-pixel ablate |
|--------|----------|----------------------|------------------------|
| Clean acc | **0.9478** | **0.9478** | 0.9478 |
| ASR | **0.9992** | **0.0028** | 0.0028 |

Astra’s claimed pixel set == GT; ablation matches oracle GT-pixel ablation.

**Tools (log):** sandbox / inspect only; **no** forwards. Turns: 6.

**Verdict B:** Scattered / non-contiguous trigger **does not** break zero-forward native reading here — column-norm / parallel-column structure still stands out in `fc1`.

---

## Interpretation

1. **Dump was convenience, not necessary** for this Exp.1-scale MLP: native `.pt` load + matrix ops suffice.
2. **Scattered 6-pixel trigger** is still linearly obvious in weight space (few high-norm, nearly parallel input columns → class 0). Harder than a solid 3×3 for spatial “patch” heuristics, but Astra recovered exact coordinates anyway.
3. **Honest limits:** no behavioral ASR measured *during* the audits; intensity / poison rate not recoverable from weights alone; results are for a **26k-param MNIST MLP**, not a claim about larger CNNs/LLMs.
4. **No failures this round** — both Pi sessions completed; both reports verified. Failures would be welcome elsewhere (deeper nets, weaker poison, noisy triggers).

## Key paths

| Artifact | Path |
|----------|------|
| This writeup | `RESULTS_HARDER.md` |
| Summary JSON | `results/harder_summary.json` |
| Exp.A | `runs/pt_only/{PROMPT.md,run.sh,session.log,models/output/,results/verify.json,logs/}` |
| Exp.B | `runs/scattered/{PROMPT.md,run.sh,session.log,models/output/,results/verify.json,logs/}` |
| Scattered trainer | `scripts/train_scattered_mlp.py` |
| Verifier | `eval/verify_harder.py` |
| Prior dump-aided | `RESULTS.md`, `results/summary.json` |
