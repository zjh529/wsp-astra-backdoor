# Paper skeleton — Native zero-shot weight-only backdoor reading with Astra

> **Full short-paper draft:** [`PAPER_DRAFT.md`](./PAPER_DRAFT.md) (expanded from this skeleton; capability numbers from `RESULTS_CAPABILITY.md`).

**Status:** Methods / Results outline for a short paper (capability claim).  
**Source of truth:** `astra-backdoor-audit/runs/RESULTS_CAPABILITY.md` (+ `RESULTS_P1.md`, `RESULTS_P2.md`).  
**Do not frame as** a bake-off vs Neural Cleanse / activation statistics / Weight-Oracle detectors.

Generated from cell reports on disk (capability refresh 2026-11-06 Asia/Hong_Kong).

---

## 1. Claim

Zero-shot `openai/gpt-6-astra`, under a **strict weight-only sandbox** (π·WSP: `--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget=0`, path allowlist, AST bans on forward / data / train, post-hoc compliance + schema validate), can **read implanted backdoors from subject-model weights alone**.

- **Subject-network forwards = 0** on every scored cell.
- Evidence is a **capability** audit: decision-level detection of planted implants, plus trigger / target localization where applicable.
- Not a claim that Astra replaces statistical detectors, nor a head-to-head bake-off.

**Main (current-protocol) aggregates** — cite `RESULTS_CAPABILITY.md`:

| Metric | Value |
|--------|-------|
| Cells (main) | 72 (0 failed) |
| TPR (poisoned) | **36/36** |
| FPR (clean) | **0/36** |
| Forwards all 0 | Yes |
| Compliance all OK | Yes (Codex depth = heuristic stub; see §6) |

Legacy Phase-T `lastpos` / `adversarial` (**0/3** each) are kept for honesty and **excluded** from the main TPR denominator (superseded by Phase P1 redesign).

---

## 2. Protocol

1. **Detector:** π·WSP + `openai/gpt-6-astra`, medium effort, **zero-shot** (no oracle fine-tune on these threats).
2. **Sandbox:** weight-only; pass budget 0; only `models/input/` (and allowlisted twin/stats for P1 cohort) readable; no subject forwards.
3. **Cell lifecycle:** `run.sh` → compliance scan → `validate_report.py` against `schema/report.schema.json`.
4. **GT isolation:** ground truth in `gt/*.json` + cohort metadata **outside** audit inputs.
5. **Decision metric:** `has_backdoor` vs GT. Localization: vision = UL / 3×3 / scattered locus; transformer = trigger token id(s) + position; target class exact when predicted.

**Phases feeding the main table**

| Phase | Role in main table |
|-------|--------------------|
| A–C (vision) | BadNet-style MLP/CNN/CIFAR implants + cleans |
| T | Transformer clean / pos0 / stealth (lastpos & adversarial → legacy only) |
| P1 | Redesigned lastpos_v2 + adversarial_cohort (+ cleans) |
| P2 | Width scale, multi-token phrase, optional depth L=2 |

---

## 3. Vision BadNet (Phases A–C)

**Setup.** TinyMLP / TinyCNN (MNIST) and TinyCIFARCNN; corner 3×3, scattered 6px, blended α=0.35, target-class 7, label-consistent; multi-seed cleans.

**Results (main table, vision slice):**

| | Value |
|--|-------|
| Poisoned TPR | **10/10** |
| Clean FPR | **0/10** (MLP 0/5, CNN 0/4, CIFAR 0/1) |

Per-threat (all detect + localization/target as in capability table): `corner_3x3` 5/5, `scattered_6px` 2/2, `blended_a035` 1/1, `corner_target7` 1/1, `label_consistent` 1/1.

**Honesty for the text.** CNN often recovers target + UL locus without unique RGB/size; blended recovers as bright 3×3 (α nuance lost); label-consistent is a weight-pattern hit with weak behavioral ASR (~0.10) — cite Phase C notes, not as a detector win.

---

## 4. Transformer threats (Phase T + P1)

**Model.** SmallTransformer 1L, d=8, 2 heads, MLP 16, seq 4, vocab 1000, 10 classes (Weight-Oracles-style threat models; arXiv:2610.07334 as motivation for *implants*, not as a detector baseline).

### 4.1 Phase T (in main: clean / pos0 / stealth)

| Threat | Decision | Trig exact | Tgt exact |
|--------|----------|------------|-----------|
| clean (seeds 0–7) | FPR **0/8** | — | — |
| pos0 | **3/3** | 3/3 | 3/3 |
| stealth | **3/3** | 3/3 | 3/3 |

### 4.2 Legacy Phase-T blind spots (honesty subsection; **not** in main TPR)

| Threat | Decision |
|--------|----------|
| lastpos | **0/3** |
| adversarial | **0/3** |

Interpretation: capability blind spots under the original task / absolute-norm protocol — **not** a comparison to V/O AUROC.

### 4.3 Phase P1 redesign (in main)

| Threat | Decision | Trig exact | Tgt exact | Notes |
|--------|----------|------------|-----------|-------|
| lastpos_v2 | **3/3** | 3/3 | 2/3 | Clean task = first-token lookup; last-pos trigger must conflict |
| adversarial_cohort | **3/3** | 1/3 | 3/3 | Same weights + `reference_twin.pt` + `cohort_stats.json` |
| clean_v2 | FPR **0/3** | — | — | |
| clean_cohort | FPR **0/1** | — | — | |

Partial misses (cite `RESULTS_P1.md` failure analysis): one lastpos_v2 target confusion among two suspicious head rows; two adversarial_cohort trigger-id near-ties (target still exact). Still native weight reading, not a trained oracle.

---

## 5. Scale / phrase / depth (Phase P2)

### 5.1 Width axis (pos0, 1L, heads=2, mlp=2×d)

| d_model | pos0 TPR (trig/tgt exact) | clean FPR |
|--------:|---------------------------|-----------|
| 8 | **3/3** (3/3 / 3/3) | **0/3** |
| 16 | **3/3** (3/3 / 3/3) | **0/3** |
| 32 | **3/3** (3/3 / 3/3) | **0/3** |

### 5.2 Multi-token phrase (d=8; tokens at pos 0 and 1)

| Metric | Value |
|--------|-------|
| Detect TPR | **3/3** |
| Both tokens + target exact | **3/3** |
| clean_phrase FPR | **0/3** |

Behavioral GT (training): phrase ASR = 1.0; single-token ASR ≈ 0.11.

### 5.3 Optional depth (d=8, L=2)

| Metric | Value | Harness |
|--------|-------|---------|
| pos0 TPR | **2/2** (trig+tgt exact) | Codex ChatGPT (`gpt-6-astra`) after Pi credits exhausted |
| clean FPR | **0/2** | Pi |

---

## 6. Honesty / limits

1. **Legacy lastpos / adversarial 0/3** remain in the paper (or appendix) as protocol blind spots; main TPR uses P1 redesign.
2. **Adversarial trigger-id** still hard under cohort-matched absolute norms (trig exact 1/3) even when decision + target recover.
3. **Vision localization** ≠ unique pixel RGB / α recovery.
4. **Codex depth compliance** ≠ Pi sandbox enforcement (transcript/heuristic stub; forwards_used=0 in report).
5. **Scope:** small subject nets; zero-shot Astra; not a production scanner claim; not a bake-off.
6. **Statistical baselines** (total L2, V/O composite AUROC) live in `RESULTS_TRANSFORMER_WO.md` as context only — optional related-work footnote, not the main result.

---

## 7. Tables to cite (copy from RESULTS_*)

| Paper table | Source |
|-------------|--------|
| Main aggregates + scoring policy | `runs/RESULTS_CAPABILITY.md` § Decision-level aggregates |
| Full per-cell main matrix | same, § Per-cell capability table |
| Legacy honesty rows | same, § Legacy Phase-T blind spots |
| P1 before→after + failure analysis | `runs/RESULTS_P1.md` |
| P2 scale / phrase / depth | `runs/RESULTS_P2.md` |
| Phase C vision families / NC cost contrast | `runs/RESULTS_PHASE_C.md` (cost = context, not bake-off) |
| Soft×hard protocol check | `runs/RESULTS_WEIGHT_ONLY_STRICT.md` / `RESULTS_PHASE_B.md` |

**Suggested one-liner for abstract / intro.**  
Under a strict zero-forward weight-only sandbox, zero-shot Astra recovers planted backdoors on vision BadNets and small transformers at **36/36** poisoned cells with **0/36** clean false positives (current protocol), including width generalization to d=32 and a two-token phrase trigger; original lastpos/adversarial misses are reported honestly and addressed by a P1 task/cohort redesign.

---

## 8. Out of scope for this short paper

- Large HF LLMs / VLMs as *subjects*
- Training a weight-oracle detector or fine-tuning Astra
- Claiming superiority over Neural Cleanse / ABS / V·O statistics
- WaNet and other expensive vision families skipped in Phase C
