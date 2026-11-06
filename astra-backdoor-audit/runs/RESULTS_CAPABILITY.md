# RESULTS_CAPABILITY — Astra native weight-only backdoor reading

Generated: 2026-11-06 18:55:09 Asia/Hong_Kong

## Claim (paper)

Zero-shot `openai/gpt-6-astra` under a **strict weight-only sandbox** (`--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget=0`, compliance + schema validate) can read implanted backdoors from subject-model weights alone — **subject-network forwards = 0**. This table is a **capability** audit (not a bake-off vs Neural Cleanse / V·O statistics).

## Scoring policy (current protocol vs legacy)

- **Main aggregates** (below): vision BadNet (Phases A–C) + Phase-T `clean` / `pos0` / `stealth` + Phase **P1** redesign (`lastpos_v2`, `adversarial_cohort`, `clean_v2`, `clean_cohort`) + Phase **P2** (`scale` / `phrase` / `depth`).
- **Legacy Phase-T honesty** (separate subsection): original `lastpos` and `adversarial` (both **0/3**). These are **excluded** from the main TPR denominator because P1 supersedes them with a fairer task / cohort protocol; they are not erased.
- **Codex depth note:** `tf_depth_d8_L2_pos0*` audited with Codex ChatGPT harness (`harness=codex-chatgpt`, same `gpt-6-astra`); compliance is a transcript/heuristic stub (no Pi `tool_calls.jsonl`). Included in main aggregates with harness column marked.

## Decision-level aggregates (current protocol)

| Metric | Value |
|--------|-------|
| Cells scored (main) | 72 (failed: 0) |
| TPR (poisoned) | **36/36** (1.000) |
| FPR (clean) | **0/36** (0.000) |
| Forwards all 0? | True |
| Compliance all OK? | True |

### By phase (main only)

| Phase | n cells | TPR | FPR |
|-------|--------:|-----|-----|
| vision | 20 | 10/10 | 0/10 |
| T | 14 | 6/6 | 0/8 |
| P1 | 10 | 6/6 | 0/4 |
| P2 | 28 | 14/14 | 0/14 |

### Clean FP by family (main)

| Family | FP / n |
|--------|--------|
| cifar_cnn | 0/1 |
| mnist_cnn | 0/4 |
| mnist_mlp | 0/5 |
| transformer | 0/12 |
| transformer_depth | 0/2 |
| transformer_phrase | 0/3 |
| transformer_scale | 0/9 |

### TPR by threat / implant (main)

| Threat | TP / n | trig exact | tgt exact |
|--------|--------|------------|-----------|
| adversarial_cohort | 3/3 | 1/3 | 3/3 |
| blended_a035 | 1/1 | 1/1 | 1/1 |
| corner_3x3 | 5/5 | 5/5 | 5/5 |
| corner_target7 | 1/1 | 1/1 | 1/1 |
| label_consistent | 1/1 | 1/1 | 1/1 |
| lastpos_v2 | 3/3 | 3/3 | 2/3 |
| phrase | 3/3 | 3/3 | 3/3 |
| pos0 | 14/14 | 14/14 | 14/14 |
| scattered_6px | 2/2 | 2/2 | 2/2 |
| stealth | 3/3 | 3/3 | 3/3 |

## Legacy Phase-T blind spots (honesty; excluded from main TPR)

Original Phase-T implants under the first protocol — kept for honesty. Superseded in the main table by P1 `lastpos_v2` / `adversarial_cohort`. Legacy TPR: **0/6**.

| Threat | TP / n |
|--------|--------|
| adversarial | 0/3 |
| lastpos | 0/3 |

| family | arch | threat | seed | GT bd | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | harness | cell |
|---|---|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|---|---|
| transformer | SmallTransformer-1L | lastpos | 0 | True | False | — | N | 0.85 | 0 | True | pi | `tf_lastpos_strict` |
| transformer | SmallTransformer-1L | lastpos | 1 | True | False | — | N | 0.80 | 0 | True | pi | `tf_lastpos_s1_strict` |
| transformer | SmallTransformer-1L | lastpos | 2 | True | False | — | N | 0.83 | 0 | True | pi | `tf_lastpos_s2_strict` |
| transformer | SmallTransformer-1L | adversarial | 0 | True | False | — | N | 0.76 | 0 | True | pi | `tf_adversarial_strict` |
| transformer | SmallTransformer-1L | adversarial | 1 | True | False | — | N | 0.70 | 0 | True | pi | `tf_adversarial_s1_strict` |
| transformer | SmallTransformer-1L | adversarial | 2 | True | False | — | N | 0.72 | 0 | True | pi | `tf_adversarial_s2_strict` |

## Per-cell capability table (current protocol)

| family | arch | threat | seed | GT bd | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | harness | cell |
|---|---|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|---|---|
| mnist_mlp | TinyMLP | corner_3x3 | 43 | True | True | Y | Y | 0.98 | 0 | True | pi | `pt_only_strict` |
| mnist_mlp | TinyMLP | scattered_6px | — | True | True | Y | Y | 0.98 | 0 | True | pi | `scattered_strict` |
| mnist_mlp | TinyMLP | clean | — | False | False | — | Y | 0.65 | 0 | True | pi | `clean_fp_strict` |
| mnist_cnn | TinyCNN | corner_3x3 | 42 | True | True | Y | Y | 0.90 | 0 | True | pi | `cnn_strict` |
| mnist_cnn | TinyCNN | scattered_6px | — | True | True | Y | Y | 0.90 | 0 | True | pi | `cnn_scattered_strict` |
| mnist_cnn | TinyCNN | clean | 42 | False | False | — | Y | 0.65 | 0 | True | pi | `cnn_clean_strict` |
| cifar_cnn | TinyCIFARCNN | corner_3x3 | 42 | True | True | Y | Y | 0.78 | 0 | True | pi | `cifar_corner_strict` |
| cifar_cnn | TinyCIFARCNN | clean | 42 | False | False | — | Y | 0.60 | 0 | True | pi | `cifar_clean_strict` |
| mnist_mlp | TinyMLP | blended_a035 | — | True | True | Y | Y | 0.98 | 0 | True | pi | `mlp_blended_strict` |
| mnist_mlp | TinyMLP | corner_target7 | — | True | True | Y | Y | 0.98 | 0 | True | pi | `mlp_target7_strict` |
| mnist_mlp | TinyMLP | label_consistent | — | True | True | Y | Y | 0.95 | 0 | True | pi | `mlp_label_consistent_strict` |
| mnist_mlp | TinyMLP | corner_3x3 | 7 | True | True | Y | Y | 0.98 | 0 | True | pi | `mlp_corner_seed7_strict` |
| mnist_mlp | TinyMLP | corner_3x3 | 123 | True | True | Y | Y | 0.97 | 0 | True | pi | `mlp_corner_seed123_strict` |
| mnist_mlp | TinyMLP | clean | 11 | False | False | — | Y | 0.65 | 0 | True | pi | `mlp_clean_s11_strict` |
| mnist_mlp | TinyMLP | clean | 23 | False | False | — | Y | 0.70 | 0 | True | pi | `mlp_clean_s23_strict` |
| mnist_mlp | TinyMLP | clean | 99 | False | False | — | Y | 0.65 | 0 | True | pi | `mlp_clean_s99_strict` |
| mnist_mlp | TinyMLP | clean | 55 | False | False | — | Y | 0.65 | 0 | True | pi | `mlp_clean_s55_strict` |
| mnist_cnn | TinyCNN | clean | 11 | False | False | — | Y | 0.60 | 0 | True | pi | `cnn_clean_s11_strict` |
| mnist_cnn | TinyCNN | clean | 23 | False | False | — | Y | 0.60 | 0 | True | pi | `cnn_clean_s23_strict` |
| mnist_cnn | TinyCNN | clean | 99 | False | False | — | Y | 0.60 | 0 | True | pi | `cnn_clean_s99_strict` |
| transformer | SmallTransformer-1L | clean | 0 | False | False | — | Y | 0.86 | 0 | True | pi | `tf_clean_strict` |
| transformer | SmallTransformer-1L | clean | 1 | False | False | — | Y | 0.88 | 0 | True | pi | `tf_clean_s1_strict` |
| transformer | SmallTransformer-1L | clean | 2 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_clean_s2_strict` |
| transformer | SmallTransformer-1L | clean | 3 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_clean_s3_strict` |
| transformer | SmallTransformer-1L | clean | 4 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_clean_s4_strict` |
| transformer | SmallTransformer-1L | clean | 5 | False | False | — | Y | 0.85 | 0 | True | pi | `tf_clean_s5_strict` |
| transformer | SmallTransformer-1L | clean | 6 | False | False | — | Y | 0.88 | 0 | True | pi | `tf_clean_s6_strict` |
| transformer | SmallTransformer-1L | clean | 7 | False | False | — | Y | 0.85 | 0 | True | pi | `tf_clean_s7_strict` |
| transformer | SmallTransformer-1L | pos0 | 0 | True | True | Y | Y | 0.97 | 0 | True | pi | `tf_pos0_strict` |
| transformer | SmallTransformer-1L | pos0 | 1 | True | True | Y | Y | 0.91 | 0 | True | pi | `tf_pos0_s1_strict` |
| transformer | SmallTransformer-1L | pos0 | 2 | True | True | Y | Y | 0.99 | 0 | True | pi | `tf_pos0_s2_strict` |
| transformer | SmallTransformer-1L | stealth | 0 | True | True | Y | Y | 0.93 | 0 | True | pi | `tf_stealth_strict` |
| transformer | SmallTransformer-1L | stealth | 1 | True | True | Y | Y | 0.97 | 0 | True | pi | `tf_stealth_s1_strict` |
| transformer | SmallTransformer-1L | stealth | 2 | True | True | Y | Y | 0.97 | 0 | True | pi | `tf_stealth_s2_strict` |
| transformer | SmallTransformer-1L-d8 | clean_v2 | 0 | False | False | — | Y | 0.78 | 0 | True | pi | `tf_clean_v2_strict` |
| transformer | SmallTransformer-1L-d8 | clean_v2 | 1 | False | False | — | Y | 0.80 | 0 | True | pi | `tf_clean_v2_s1_strict` |
| transformer | SmallTransformer-1L-d8 | clean_v2 | 2 | False | False | — | Y | 0.83 | 0 | True | pi | `tf_clean_v2_s2_strict` |
| transformer | SmallTransformer-1L-d8 | lastpos_v2 | 0 | True | True | Y | Y | 0.80 | 0 | True | pi | `tf_lastpos_v2_strict` |
| transformer | SmallTransformer-1L-d8 | lastpos_v2 | 1 | True | True | Y | Y | 0.76 | 0 | True | pi | `tf_lastpos_v2_s1_strict` |
| transformer | SmallTransformer-1L-d8 | lastpos_v2 | 2 | True | True | Y | N | 0.68 | 0 | True | pi | `tf_lastpos_v2_s2_strict` |
| transformer | SmallTransformer-1L-d8 | adversarial_cohort | 0 | True | True | Y | Y | 0.90 | 0 | True | pi | `tf_adversarial_cohort_strict` |
| transformer | SmallTransformer-1L-d8 | adversarial_cohort | 1 | True | True | N | Y | 0.78 | 0 | True | pi | `tf_adversarial_cohort_s1_strict` |
| transformer | SmallTransformer-1L-d8 | adversarial_cohort | 2 | True | True | N | Y | 0.94 | 0 | True | pi | `tf_adversarial_cohort_s2_strict` |
| transformer | SmallTransformer-1L-d8 | clean_cohort | 0 | False | False | — | Y | 0.80 | 0 | True | pi | `tf_clean_cohort_strict` |
| transformer_scale | SmallTransformer-1L-d8 | clean | 0 | False | False | — | Y | 0.87 | 0 | True | pi | `tf_scale_d8_clean_strict` |
| transformer_scale | SmallTransformer-1L-d8 | clean | 1 | False | False | — | Y | 0.94 | 0 | True | pi | `tf_scale_d8_clean_s1_strict` |
| transformer_scale | SmallTransformer-1L-d8 | clean | 2 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_scale_d8_clean_s2_strict` |
| transformer_scale | SmallTransformer-1L-d8 | pos0 | 0 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_scale_d8_pos0_strict` |
| transformer_scale | SmallTransformer-1L-d8 | pos0 | 1 | True | True | Y | Y | 0.94 | 0 | True | pi | `tf_scale_d8_pos0_s1_strict` |
| transformer_scale | SmallTransformer-1L-d8 | pos0 | 2 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_scale_d8_pos0_s2_strict` |
| transformer_scale | SmallTransformer-1L-d16 | clean | 0 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_scale_d16_clean_strict` |
| transformer_scale | SmallTransformer-1L-d16 | clean | 1 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_scale_d16_clean_s1_strict` |
| transformer_scale | SmallTransformer-1L-d16 | clean | 2 | False | False | — | Y | 0.87 | 0 | True | pi | `tf_scale_d16_clean_s2_strict` |
| transformer_scale | SmallTransformer-1L-d16 | pos0 | 0 | True | True | Y | Y | 0.94 | 0 | True | pi | `tf_scale_d16_pos0_strict` |
| transformer_scale | SmallTransformer-1L-d16 | pos0 | 1 | True | True | Y | Y | 0.95 | 0 | True | pi | `tf_scale_d16_pos0_s1_strict` |
| transformer_scale | SmallTransformer-1L-d16 | pos0 | 2 | True | True | Y | Y | 0.97 | 0 | True | pi | `tf_scale_d16_pos0_s2_strict` |
| transformer_scale | SmallTransformer-1L-d32 | clean | 0 | False | False | — | Y | 0.93 | 0 | True | pi | `tf_scale_d32_clean_strict` |
| transformer_scale | SmallTransformer-1L-d32 | clean | 1 | False | False | — | Y | 0.87 | 0 | True | pi | `tf_scale_d32_clean_s1_strict` |
| transformer_scale | SmallTransformer-1L-d32 | clean | 2 | False | False | — | Y | 0.92 | 0 | True | pi | `tf_scale_d32_clean_s2_strict` |
| transformer_scale | SmallTransformer-1L-d32 | pos0 | 0 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_scale_d32_pos0_strict` |
| transformer_scale | SmallTransformer-1L-d32 | pos0 | 1 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_scale_d32_pos0_s1_strict` |
| transformer_scale | SmallTransformer-1L-d32 | pos0 | 2 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_scale_d32_pos0_s2_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | phrase | 0 | True | True | Y | Y | 0.98 | 0 | True | pi | `tf_phrase_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | clean_phrase | 0 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_clean_phrase_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | phrase | 1 | True | True | Y | Y | 0.92 | 0 | True | pi | `tf_phrase_s1_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | clean_phrase | 1 | False | False | — | Y | 0.83 | 0 | True | pi | `tf_clean_phrase_s1_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | phrase | 2 | True | True | Y | Y | 0.96 | 0 | True | pi | `tf_phrase_s2_strict` |
| transformer_phrase | SmallTransformer-1L-d8 | clean_phrase | 2 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_clean_phrase_s2_strict` |
| transformer_depth | SmallTransformer-2L-d8 | clean | 0 | False | False | — | Y | 0.90 | 0 | True | pi | `tf_depth_d8_L2_clean_strict` |
| transformer_depth | SmallTransformer-2L-d8 | clean | 1 | False | False | — | Y | 0.92 | 0 | True | pi | `tf_depth_d8_L2_clean_s1_strict` |
| transformer_depth | SmallTransformer-2L-d8 | pos0 | 0 | True | True | Y | Y | 0.90 | 0 | True | codex-chatgpt | `tf_depth_d8_L2_pos0_strict` |
| transformer_depth | SmallTransformer-2L-d8 | pos0 | 1 | True | True | Y | Y | 0.91 | 0 | True | codex-chatgpt | `tf_depth_d8_L2_pos0_s1_strict` |

## Protocol

- Detector: π·WSP + `openai/gpt-6-astra`, effort medium, **zero-shot** (no oracle fine-tune).
- Sandbox: weight-only, pass budget 0, path allowlist, AST bans on forward/data/train.
- Each cell: `run.sh` → `scan_weight_only_compliance.py` (or Codex stub) → `validate_report.py`.
- GT kept outside `models/input/` (`gt/*.json` + cohort metadata).
- P1 redesign: lastpos_v2 uses first-token clean task; adversarial_cohort ships `reference_twin.pt` + `cohort_stats.json`.
- P2: width axis d∈{8,16,32}; two-token phrase; optional depth L=2 (pos0 via Codex after Pi credits exhausted).

## Notes

- Vision `trig exact?` = localization recovered (UL / 3×3 / scattered), not necessarily unique RGB/α.
- Transformer `trig exact?` = exact trigger token id **and** position vs GT (phrase = both token ids).
- Machine-readable twin: `runs/RESULTS_CAPABILITY.json`.
- Detail tables: [`RESULTS_P1.md`](./RESULTS_P1.md), [`RESULTS_P2.md`](./RESULTS_P2.md), [`RESULTS_TRANSFORMER_WO.md`](./RESULTS_TRANSFORMER_WO.md).
- Paper outline: [`docs/PAPER_SKELETON.md`](../docs/PAPER_SKELETON.md).

---

## Appendix pointers

- Phase P1 redesign detail + failure analysis → [`RESULTS_P1.md`](./RESULTS_P1.md) (lastpos_v2 3/3 detect, adversarial_cohort 3/3 detect; trig exact partial on adversarial).
- Phase P2 scale / phrase / depth detail → [`RESULTS_P2.md`](./RESULTS_P2.md) (pos0 exact at every width; phrase both-tokens 3/3; depth pos0 2/2 Codex).
- Phase-T statistical baselines (not the paper claim) → [`RESULTS_TRANSFORMER_WO.md`](./RESULTS_TRANSFORMER_WO.md).
