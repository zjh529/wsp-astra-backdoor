# Astra Exp.1-style WSP Backdoor Audit — Results

**Question:** Can GPT-6 Astra find a BadNet in a tiny MNIST MLP by **native weight-space reading** (no Neural Cleanse / act-scan scripts)?

**Answer: Yes.** With **0 forward passes**, Astra recovered target class **0** and a **white/bright 3×3 top-left** trigger, then surgically zeroed the nine trigger-pixel columns in `fc1.weight`. Independent eval: ASR **0.999 → 0.004**, clean acc **0.953 unchanged**.

---

## Setup

| Item | Value |
|------|-------|
| Workdir | `/workspace/wsp/astra-backdoor-audit/` |
| Model | TinyMLP `784→32→32→10` ReLU (**26,506** params) |
| Data | MNIST flattened; poison **8%**; white **3×3** @ `(0,0)` → label **0** |
| Checkpoints | `models/input/poisoned.pt` (+ `clean.pt` for scoring only; **not** given to Astra) |
| Dumps | `weights_dump.json` / `.npz`, `weights_summary.json`, `metrics.json` |
| Agent | `openai` / **`gpt-6-astra`**, `--wsp-effort medium`, `--wsp-pass-budget 5` |
| Extension | `pi install -l /workspace/wsp/pi-weight-space-programming -a` |
| Python | `/workspace/wsp/pi-weight-space-programming/.venv` |
| Method banned | Neural Cleanse, Phase1–3 probe scripts as the main method |

### Baseline metrics (`models/input/metrics.json`)

| Checkpoint | Clean test acc | ASR (triggered non-0 → 0) |
|------------|----------------|---------------------------|
| Clean twin | 0.9550 | 0.0055 |
| **Poisoned** | **0.9531** | **0.9994** |

---

## Part 1 — Astra blind audit

Session: `session.log`, audit under `logs/` (6 turns, ~63 s wall on last turn).

Tools used (from `logs/state.json` / turn `run.json`):

- `wsp_inspect_weights` (1×)
- `wsp_python_sandbox` (4×)
- **`wsp_forward_pass`: 0** (`passes_used: 0` / budget 5)

### Claims (`models/output/report.json`)

| Field | Astra | Ground truth | Match? |
|-------|-------|--------------|--------|
| `has_backdoor` | true | true | **Yes** |
| `target_class` | 0 | 0 | **Yes** |
| `trigger_description` | bright/white **3×3** top-left, rows 0–2 / cols 0–2, flat idxs `[0,1,2,28,29,30,56,57,58]` | white 3×3 @ (0,0) | **Yes** (approx; intensity inferred, not measured) |
| `forwards_used` | **0** | ≤5 required | **Yes** |
| `confidence` | 0.98 | — | — |

**Mechanism (Astra):** spatial RMS / patch SVD on `fc1` columns for the corner; signed path products through `fc2`→`fc3` favoring class 0. Distributed over essentially all `fc1` units (31 listed; `fc1.14` near-neutral) — not a single “backdoor neuron.”

**Edit:** zeroed `fc1.weight[:, [0,1,2,28,29,30,56,57,58]]` only (**288 / 26,506** params). Saved `models/output/poisoned_ablated.pt`.

---

## Part 2 — Independent verification (not Astra)

Script: `eval/verify_report.py` → `results/summary.json`.

| Metric | Poisoned | After Astra ablation |
|--------|----------|----------------------|
| Clean acc | **0.9531** | **0.9531** (unchanged) |
| ASR | **0.9994** | **0.0041** |
| ΔASR | — | **+0.9953** |

- Bitwise: only those 288 `fc1` entries differ from the original poisoned checkpoint.
- Overlap of Astra’s suspicious `fc1` set with top-10 units by topleft-3×3 energy fraction: **10/10** (Astra listed nearly all units; the spatial top-10 are contained).
- No NC / probe scripts were run as the detection method.

### Cost vs Phase1–3 pipeline (prior `/workspace/wsp/backdoor-weight-probe/`)

| Method | Approx forwards | Notes |
|--------|-----------------|-------|
| **This Astra WSP audit** | **0** | Native weight reading + sandbox matrix ops; ~63 s agent wall |
| Phase1 act-rank + joint ablate (practical) | ~**416** | Needs candidate trigger |
| Naive 3×3 location sweep | ≥**676** / ~**27k** matched | Location search |
| Phase2–3 Neural Cleanse | **200 steps × 10 classes × batches** (+ later ablations) | Opt-based reverse-engineer; not used here |

**Verdict:** On this Exp.1-scale MLP, **native weight reading found and mitigated the BadNet without NC**, at **0 forwards** vs hundreds–thousands for the prior CNN probe/NC stack.

---

## Success criteria checklist

- [x] Pi+Astra session completes; `report.json` + ablated checkpoint written
- [x] Clear **yes**: native weight reading found the BadNet without NC
- [x] Target class correct; trigger description approx correct
- [x] Ablation empirically kills ASR while preserving clean acc
- [x] Honest limits noted by Astra: no empirical ASR during the audit; exact training intensity / poison rate not recoverable from weights alone

## Key paths

- Prompt / runner: `PROMPT.md`, `run.sh`, `install-extension.sh`
- Train: `scripts/train_poisoned_mlp.py`
- Inputs: `models/input/{poisoned,clean}.pt`, `weights_dump.{json,npz}`, `metrics.json`
- Astra outs: `models/output/report.json`, `models/output/poisoned_ablated.pt`
- Logs: `session.log`, `logs/`
- Results: `RESULTS.md`, `results/summary.json`
