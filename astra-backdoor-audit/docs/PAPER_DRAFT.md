# Reading Implanted Backdoors from Weights Alone: A Zero-Shot Weight-Only Capability Audit of Astra

**Draft status.** Short-paper / workshop draft (native-capability claim).  
**Source of truth.** Numbers are taken from `astra-backdoor-audit/runs/RESULTS_CAPABILITY.md` (and `RESULTS_P1.md`, `RESULTS_P2.md`, `RESULTS_TRANSFORMER_WO.md`, `RESULTS_PHASE_C.md`). Do not treat this draft as a bake-off against Neural Cleanse or Weight-Oracle detectors.  
**Outline.** See also `docs/PAPER_SKELETON.md`.

---

## Abstract

We audit whether a zero-shot large language model with a tool-using weight-space programming harness can *read* implanted neural backdoors from subject-model parameters alone. Under a strict **weight-only** sandbox—path allowlisting, AST bans on forward / data / train, pass budget zero, and post-hoc compliance plus schema validation—`openai/gpt-6-astra` (π·WSP) inspects only checkpoint tensors. **Subject-network forwards are zero on every scored cell.** On the current protocol, decision-level detection reaches **TPR 36/36** on poisoned cells and **FPR 0/36** on clean twins, spanning BadNet-style vision implants (MLP / CNN / CIFAR) and small transformers (position-0, stealth, redesigned last-position and cohort-matched adversarial threats, width up to $d{=}32$, a two-token phrase trigger, and depth $L{=}2$). Original Phase-T last-position and adversarial protocols miss **0/3** each; we report them honestly and supersede them in the main denominator with a Phase-P1 task / cohort redesign. The claim is a **native capability** result under a documented sandbox—not superiority over statistical detectors, and not a production scanner.

---

## 1. Introduction

Backdoors in neural networks remain a practical supply-chain risk: a model that behaves correctly on clean inputs can be steered by a trigger pattern or token sequence toward an attacker-chosen target. Classical defenses reverse-engineer triggers by optimizing on the subject network (e.g., Neural Cleanse) or score hand-crafted weight statistics. A separate line of work asks whether *language models* can act as **weight oracles**—reading structure from parameters—possibly after fine-tuning on labeled weight corpora.

This paper asks a narrower, complementary question:

> Can a **zero-shot** tool-using LLM, denied all subject forwards and all training / evaluation data, still detect and localize planted backdoors from weights alone?

We treat the answer as an empirical **capability** claim. Success would mean that contemporary LLM agents already possess enough inductive structure over tensors, circuits, and threat templates to surface implanted backdoors without a specialized oracle fine-tune and without running the subject. Failure modes are equally informative: which implant designs leave weight signatures that are invisible under a fair task, or that match absolute cohort norms.

We use π·WSP with `openai/gpt-6-astra` under an enforceable weight-only mode. Across vision BadNets and Weight-Oracles-style transformer threats, the current protocol yields **36/36** true positives and **0/36** false positives at the decision level (`has_backdoor`), with trigger / target localization exact on most cells. We deliberately **do not** frame the main result as a bake-off against Neural Cleanse or against the fine-tuned Qwen weight oracle of prior work; those appear only as cost or threat-model context.

**Contributions.**

1. A strict weight-only audit protocol (sandbox + compliance + schema) with **subject forwards = 0**.
2. A unified capability table covering vision BadNet families and small-transformer threats (Phases A–C, T, P1, P2).
3. An honest account of Phase-T blind spots (lastpos / adversarial **0/3** each) and a Phase-P1 redesign that restores decision-level detection while leaving residual trigger-id ambiguity on adversarial implants.
4. Evidence that the same zero-shot capability holds under modest width scaling ($d{\in}\{8,16,32\}$), a multi-token phrase trigger, and depth $L{=}2$.

---

## 2. Related Work

**BadNets and trigger families.** Gu et al. introduced BadNets-style patch triggers; follow-on work studies blended / invisible patterns, label-consistent poisoning, and more sophisticated vision triggers. Our vision cells reuse small MLP / CNN / CIFAR BadNet implants (corner $3{\times}3$, scattered pixels, blended $\alpha{=}0.35$, alternate target class, label-consistent) as capability probes—not as a survey of SOTA attacks.

**Optimization-based detection.** Neural Cleanse (NC) and related methods reverse-engineer triggers by optimizing over the subject with many forwards. We mention NC only as a **cost / workflow contrast** (subject forwards ≫ 0). We do **not** claim Astra is faster or more accurate than NC.

**Weight Oracles.** Prior work (arXiv:2610.07334) trains a weight-oracle detector and evaluates AUROC on small transformers under four threat models: position-0 (pos0), last-position (lastpos), stealth (low poison weight), and adversarial (regularizer matching clean-cohort norms). We adopt the **target architecture and threat constructions** as implant generators. We do **not** reproduce their fine-tuned Qwen oracle, and we do **not** claim superiority over their AUROC numbers. Statistical V/O composites from our own training runs appear only as optional context in the Phase-T results file.

**Weight-space programming (WSP) / Astra.** π·WSP exposes tools for inspecting and algebraically manipulating tensors under a harness. Our contribution is the **weight-only enforcement layer** and a zero-shot backdoor-reading audit, not a new training method for oracles.

---

## 3. Threat Model and Protocol

### 3.1 Capability claim (what is and is not asserted)

**Asserted.** Zero-shot `openai/gpt-6-astra` under π·WSP weight-only mode can, from subject weights alone, decide `has_backdoor` and often recover trigger locus / token id(s) and target class for the planted implants in our matrix.

**Not asserted.** Superiority to NC, ABS, or Weight-Oracle AUROC; calibration as a production scanner; recovery of unique pixel RGB / blend $\alpha$; applicability to large HF LLMs / VLMs as *subjects*.

### 3.2 Sandbox

Each audit cell runs with:

- `--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `--wsp-pass-budget 0`;
- path allowlist centered on `models/input/` (Phase P1 also allows twin / cohort stats under that tree);
- AST bans on subject forward, dataset I/O, and training;
- post-hoc `scan_weight_only_compliance.py` (Pi) or a Codex transcript/heuristic stub for depth-pos0 cells;
- `validate_report.py` against `schema/report.schema.json`.

**Ground truth** lives in `gt/*.json` and cohort metadata **outside** audit inputs. Poisoned checkpoints are sanitized of family / ASR metadata before audit.

### 3.3 Scoring

- **Decision:** predicted `has_backdoor` vs GT.
- **Vision localization (`trig exact?`):** recovered UL / $3{\times}3$ / scattered locus (not unique RGB / $\alpha$).
- **Transformer localization:** exact trigger token id(s) **and** position vs GT (phrase = both token ids); target class exact when predicted.
- **Main aggregates:** vision A–C + Phase-T `clean` / `pos0` / `stealth` + Phase P1 + Phase P2.
- **Excluded from main TPR:** legacy Phase-T `lastpos` and `adversarial` (both **0/3**), retained as honesty rows because P1 supersedes them.

---

## 4. Experimental Setup

**Detector.** π·WSP + `openai/gpt-6-astra`, medium effort, **zero-shot** (no oracle fine-tune on these threats).

**Vision subjects (Phases A–C).** TinyMLP / TinyCNN (MNIST) and TinyCIFARCNN. Implants: corner $3{\times}3$, scattered 6px, blended $\alpha{=}0.35$, corner with target class 7, label-consistent (behavioral ASR $\approx 0.10$). Multi-seed cleans and poisoned corners. Soft-prompt vs hard-sandbox agreement was checked in Phase B; paper cells use the hard weight-only path.

**Transformer subjects (Phases T / P1 / P2).** SmallTransformer: pre-LN, typically 1 layer, $d{=}8$, 2 heads, MLP $16$, sequence length 4, vocab 1000, 10 classes (classifier reads last position), matching the Weight-Oracles toy setup. Training uses AdamW + cosine (lr $10^{-2}$, 3000 steps) with clean and poison batches as in the Phase-T recipe. Threats:

| Threat | Construction (summary) |
|--------|------------------------|
| clean | No implant |
| pos0 | Trigger at position 0; attention V/O routing |
| stealth | pos0 with low poison weight (ASR mean $\approx 0.53$ on 8 seeds) |
| lastpos (legacy) | Trigger at last position under `label = class_map[last]` |
| adversarial (legacy) | pos0 + penalty matching clean-cohort norms |
| lastpos_v2 (P1) | Clean task = first-token lookup; last-pos trigger must conflict |
| adversarial_cohort (P1) | Same adversarial weights + `reference_twin.pt` + `cohort_stats.json` |
| phrase (P2) | Two tokens at positions 0 and 1; only conjunction → target |
| scale / depth (P2) | pos0 at $d{\in}\{8,16,32\}$; optional $L{=}2$, $d{=}8$ |

**Harness note (depth).** Depth clean cells used Pi. Depth pos0 cells (`tf_depth_d8_L2_pos0*`) were audited with a Codex ChatGPT harness (`harness=codex-chatgpt`, same `gpt-6-astra`) after Pi platform credits were exhausted. Codex compliance is a transcript/heuristic stub (no Pi `tool_calls.jsonl`); reports still record `forwards_used=0`. Pi and Codex are not identical sandbox enforcements.

**Reproducibility.** Code, cells, GT, and RESULTS tables live in the public research tree [https://github.com/zjh529/wsp-astra-backdoor](https://github.com/zjh529/wsp-astra-backdoor).[^repo]

[^repo]: Repository snapshot corresponding to this draft: `https://github.com/zjh529/wsp-astra-backdoor`. Capability refresh timestamp in `RESULTS_CAPABILITY.md`: 2026-11-06 18:55:09 Asia/Hong_Kong.

---

## 5. Results

### 5.1 Main capability aggregates

Under the current protocol (`RESULTS_CAPABILITY.md`):

| Metric | Value |
|--------|-------|
| Cells scored (main) | 72 (failed: 0) |
| TPR (poisoned) | **36/36** (1.000) |
| FPR (clean) | **0/36** (0.000) |
| Forwards all 0? | Yes |
| Compliance all OK? | Yes (Codex depth = stub; see §4) |

By phase (main only):

| Phase | $n$ cells | TPR | FPR |
|-------|----------:|-----|-----|
| vision (A–C) | 20 | 10/10 | 0/10 |
| T (`clean` / `pos0` / `stealth`) | 14 | 6/6 | 0/8 |
| P1 | 10 | 6/6 | 0/4 |
| P2 | 28 | 14/14 | 0/14 |

TPR by implant family (main):

| Threat | TP / $n$ | trig exact | tgt exact |
|--------|----------|------------|-----------|
| corner_3x3 | 5/5 | 5/5 | 5/5 |
| scattered_6px | 2/2 | 2/2 | 2/2 |
| blended_a035 | 1/1 | 1/1 | 1/1 |
| corner_target7 | 1/1 | 1/1 | 1/1 |
| label_consistent | 1/1 | 1/1 | 1/1 |
| pos0 | 14/14 | 14/14 | 14/14 |
| stealth | 3/3 | 3/3 | 3/3 |
| lastpos_v2 | 3/3 | 3/3 | 2/3 |
| adversarial_cohort | 3/3 | 1/3 | 3/3 |
| phrase | 3/3 | 3/3 | 3/3 |

Clean false positives by family remain **0** across mnist_mlp (0/5), mnist_cnn (0/4), cifar_cnn (0/1), and all transformer clean families in the main table (0/26 combined across transformer / scale / phrase / depth cleans).

### 5.2 Vision BadNet summary

Poisoned vision cells achieve **10/10** detection with localization and target recovery as scored in the capability table. Clean vision FP is **0/10**.

Honest localization limits (Phase C notes):

- CNN / CIFAR often recover target + upper-left locus without a unique RGB / patch-size fingerprint.
- Blended ($\alpha{=}0.35$) is recovered as a bright filled $3{\times}3$; blend intensity is not uniquely reported.
- Label-consistent poisoning still leaves a weight-pattern hit (Astra flags BD / target / UL) while behavioral ASR is weak ($\approx 0.10$)—a weight-pattern vs behavior mismatch risk, not a detector “win.”

Soft vs hard weight-only controls (Phase B) agreed qualitatively with `forwards_used=0`; the paper matrix uses the hard sandbox.

Neural Cleanse wall times from earlier phases (many subject forwards) are recorded only as cost context: e.g., NC on MNIST TinyCNN $\approx 122.6$s with many forwards; Astra CIFAR corner $\approx 87$s wall with **0** subject forwards. Wall clock is dominated by LLM tool latency; the paper cost claim is **forwards = 0**, not wall-clock dominance.

### 5.3 Transformer Phase T → Phase P1

**In main (Phase T slice).** Clean seeds 0–7: FPR **0/8**. pos0: **3/3** with trigger id, position, and target exact. stealth: **3/3** likewise exact (despite mean ASR $\approx 0.53$ on the 8-seed training cohort).

**Legacy honesty (excluded from main TPR).** Under the original protocols:

| Threat | Decision TPR |
|--------|--------------|
| lastpos | **0/3** |
| adversarial | **0/3** |

Interpretation (from Phase-T failure analysis): with `label = class_map[last token]`, a last-position trigger is nearly a relabel of one vocabulary row—hard to separate from a clean map without structure that conflicts. Adversarial implants matched penalized absolute norms; zero-shot Astra lacked a clean twin / cohort reference and called them clean despite unpenalized OV / Wo concentration leaking in cohort statistics.

**Phase P1 redesign (in main).**

| Threat | Decision | Trig exact | Tgt exact |
|--------|----------|------------|-----------|
| lastpos_v2 | **3/3** | 3/3 | 2/3 |
| adversarial_cohort | **3/3** | 1/3 | 3/3 |
| clean_v2 | FPR **0/3** | — | — |
| clean_cohort | FPR **0/1** | — | — |

Before → after at decision level: lastpos $0/3$ → lastpos_v2 **3/3**; adversarial $0/3$ → adversarial_cohort **3/3**.

Partial localization misses (`RESULTS_P1.md`): one lastpos_v2 seed recovers trigger + position but confuses target among two suspicious head rows (predicted 2 vs GT 5); two adversarial_cohort seeds recover target exactly but miss the trigger id among near-tied tokens when absolute norms match the cohort. Decision-level detection still holds; trigger-id disambiguation remains imperfect on cohort-matched adversarial implants.

### 5.4 Phase P2: width, phrase, depth

**Width axis** (pos0, 1L, heads${=}2$, mlp${=}2d$):

| $d_{\mathrm{model}}$ | pos0 TPR (trig / tgt exact) | clean FPR |
|---------------------:|-----------------------------|-----------|
| 8 | **3/3** (3/3 / 3/3) | **0/3** |
| 16 | **3/3** (3/3 / 3/3) | **0/3** |
| 32 | **3/3** (3/3 / 3/3) | **0/3** |

**Multi-token phrase** ($d{=}8$; tokens at positions 0 and 1). Detect TPR **3/3**; both tokens and target exact **3/3**; clean_phrase FPR **0/3**. Behavioral GT from training (`models/transformer_wo_p2/metrics.json`): phrase ASR mean **1.0**; single-token ASR mean $\approx 0.107$ (per-seed singles $\approx 0.10$–$0.11$).

**Depth** ($d{=}8$, $L{=}2$). Clean FPR **0/2** (Pi). pos0 TPR **2/2** with trig+tgt exact (Codex ChatGPT harness; see §4).

No poisoned misses and no clean false positives among finished P2 cells.

---

## 6. Limitations and Honesty

1. **Legacy blind spots.** Phase-T lastpos and adversarial remain **0/3** each under the original protocol; main TPR uses the P1 redesign. Erasing those misses would overstate robustness to task / norm matching.
2. **Adversarial trigger-id.** Even with twin + cohort stats, trig exact is only **1/3** on adversarial_cohort; target recovery is **3/3**. Near-ties under matched absolute norms are a real residual failure mode.
3. **Vision localization ≠ pixel fingerprint.** UL / $3{\times}3$ / scattered recovery does not imply unique RGB or blend $\alpha$.
4. **Harness heterogeneity.** Codex depth compliance ≠ Pi sandbox enforcement. Same model family and reported forwards${=}0$, but tool mediation differs.
5. **Toy scale.** Subjects are small MLPs / CNNs and transformers with thousands to tens of thousands of parameters—not production LLMs as audit *targets*.
6. **Sample size.** Many threat types use 2–3 seeds for Astra audits; results are exact per-seed matches, not calibrated probabilities or large-$n$ confidence intervals.
7. **Not a bake-off.** Statistical V/O AUROCs and NC timings are contextual. The paper claim is zero-shot weight-only *capability*, not detector ranking.
8. **Skipped families.** WaNet and other expensive vision triggers were skipped on CPU budget (Phase C).

---

## 7. Conclusion

Under a strict zero-forward weight-only sandbox, zero-shot Astra recovers planted backdoors on vision BadNets and small transformers at **36/36** poisoned cells with **0/36** clean false positives on the current protocol—including width generalization to $d{=}32$, a two-token phrase trigger with both tokens exact, and depth-$L{=}2$ pos0. Original last-position and adversarial misses are reported honestly and addressed by a Phase-P1 task / cohort redesign that restores decision-level detection while leaving residual adversarial trigger-id ambiguity. The result supports a focused claim: contemporary tool-using LLMs can already *read* a useful class of implanted backdoors from weights alone when the harness forbids subject execution. Scaling subjects, hardening adversarial localization, and separating sandbox harnesses remain open.

---

## Appendix A — Legacy Phase-T honesty rows

| family | threat | seed | GT bd | Astra bd | conf | fwd | cell |
|--------|--------|-----:|:-----:|:--------:|-----:|----:|------|
| transformer | lastpos | 0 | True | False | 0.85 | 0 | `tf_lastpos_strict` |
| transformer | lastpos | 1 | True | False | 0.80 | 0 | `tf_lastpos_s1_strict` |
| transformer | lastpos | 2 | True | False | 0.83 | 0 | `tf_lastpos_s2_strict` |
| transformer | adversarial | 0 | True | False | 0.76 | 0 | `tf_adversarial_strict` |
| transformer | adversarial | 1 | True | False | 0.70 | 0 | `tf_adversarial_s1_strict` |
| transformer | adversarial | 2 | True | False | 0.72 | 0 | `tf_adversarial_s2_strict` |

Legacy TPR: **0/6**. Excluded from main aggregates.

## Appendix B — Table sources

| Paper table | Source file |
|-------------|-------------|
| Main aggregates + scoring policy | `runs/RESULTS_CAPABILITY.md` |
| Full per-cell main matrix | same |
| Legacy honesty | same + `RESULTS_TRANSFORMER_WO.md` |
| P1 before→after + failure analysis | `runs/RESULTS_P1.md` |
| P2 scale / phrase / depth | `runs/RESULTS_P2.md` |
| Vision families / NC cost contrast | `runs/RESULTS_PHASE_C.md` |
| Soft×hard protocol check | `runs/RESULTS_WEIGHT_ONLY_STRICT.md`, `RESULTS_PHASE_B.md` |
| Phrase ASR means | `models/transformer_wo_p2/metrics.json` |

## Appendix C — Out of scope

Large HF LLMs / VLMs as subjects; training a weight-oracle detector or fine-tuning Astra on these threats; claiming superiority over Neural Cleanse / ABS / V·O statistics; WaNet and other expensive vision families skipped in Phase C.
