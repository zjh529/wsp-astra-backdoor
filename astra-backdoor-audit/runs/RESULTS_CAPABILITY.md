# RESULTS_CAPABILITY — Astra native weight-only backdoor reading

Generated: 2026-11-06 16:11:21 Asia/Hong_Kong

## Claim (paper P0)

Zero-shot `openai/gpt-6-astra` under a **strict weight-only sandbox** (`--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget=0`, compliance + schema validate) can read implanted backdoors from subject-model weights alone — **subject-network forwards = 0**. This table is a **capability** audit (not a bake-off vs Neural Cleanse / V·O statistics).

## Decision-level aggregates

| Metric | Value |
|--------|-------|
| Cells scored | 40 (failed: 0) |
| TPR (poisoned) | **16/22** (0.727) |
| FPR (clean) | **0/18** (0.000) |
| Forwards all 0? | True |
| Compliance all OK? | True |

### Clean FP by family

| Family | FP / n |
|--------|--------|
| cifar_cnn | 0/1 |
| mnist_cnn | 0/4 |
| mnist_mlp | 0/5 |
| transformer | 0/8 |

### TPR by threat / implant

| Threat | TP / n |
|--------|--------|
| adversarial | 0/3 |
| blended_a035 | 1/1 |
| corner_3x3 | 5/5 |
| corner_target7 | 1/1 |
| label_consistent | 1/1 |
| lastpos | 0/3 |
| pos0 | 3/3 |
| scattered_6px | 2/2 |
| stealth | 3/3 |

## Per-cell capability table

| family | arch | threat | seed | GT bd | Astra bd | trig exact? | tgt exact? | conf | fwd | compliance | cell |
|---|---|---|---:|:---:|:---:|:---:|:---:|---:|---:|:---:|---|
| mnist_mlp | TinyMLP | corner_3x3 | 43 | True | True | Y | Y | 0.98 | 0 | True | `pt_only_strict` |
| mnist_mlp | TinyMLP | scattered_6px | — | True | True | Y | Y | 0.98 | 0 | True | `scattered_strict` |
| mnist_mlp | TinyMLP | clean | — | False | False | — | Y | 0.65 | 0 | True | `clean_fp_strict` |
| mnist_cnn | TinyCNN | corner_3x3 | 42 | True | True | Y | Y | 0.90 | 0 | True | `cnn_strict` |
| mnist_cnn | TinyCNN | scattered_6px | — | True | True | Y | Y | 0.90 | 0 | True | `cnn_scattered_strict` |
| mnist_cnn | TinyCNN | clean | 42 | False | False | — | Y | 0.65 | 0 | True | `cnn_clean_strict` |
| cifar_cnn | TinyCIFARCNN | corner_3x3 | 42 | True | True | Y | Y | 0.78 | 0 | True | `cifar_corner_strict` |
| cifar_cnn | TinyCIFARCNN | clean | 42 | False | False | — | Y | 0.60 | 0 | True | `cifar_clean_strict` |
| mnist_mlp | TinyMLP | blended_a035 | — | True | True | Y | Y | 0.98 | 0 | True | `mlp_blended_strict` |
| mnist_mlp | TinyMLP | corner_target7 | — | True | True | Y | Y | 0.98 | 0 | True | `mlp_target7_strict` |
| mnist_mlp | TinyMLP | label_consistent | — | True | True | Y | Y | 0.95 | 0 | True | `mlp_label_consistent_strict` |
| mnist_mlp | TinyMLP | corner_3x3 | 7 | True | True | Y | Y | 0.98 | 0 | True | `mlp_corner_seed7_strict` |
| mnist_mlp | TinyMLP | corner_3x3 | 123 | True | True | Y | Y | 0.97 | 0 | True | `mlp_corner_seed123_strict` |
| mnist_mlp | TinyMLP | clean | 11 | False | False | — | Y | 0.65 | 0 | True | `mlp_clean_s11_strict` |
| mnist_mlp | TinyMLP | clean | 23 | False | False | — | Y | 0.70 | 0 | True | `mlp_clean_s23_strict` |
| mnist_mlp | TinyMLP | clean | 99 | False | False | — | Y | 0.65 | 0 | True | `mlp_clean_s99_strict` |
| mnist_mlp | TinyMLP | clean | 55 | False | False | — | Y | 0.65 | 0 | True | `mlp_clean_s55_strict` |
| mnist_cnn | TinyCNN | clean | 11 | False | False | — | Y | 0.60 | 0 | True | `cnn_clean_s11_strict` |
| mnist_cnn | TinyCNN | clean | 23 | False | False | — | Y | 0.60 | 0 | True | `cnn_clean_s23_strict` |
| mnist_cnn | TinyCNN | clean | 99 | False | False | — | Y | 0.60 | 0 | True | `cnn_clean_s99_strict` |
| transformer | SmallTransformer-1L | clean | 0 | False | False | — | Y | 0.86 | 0 | True | `tf_clean_strict` |
| transformer | SmallTransformer-1L | clean | 1 | False | False | — | Y | 0.88 | 0 | True | `tf_clean_s1_strict` |
| transformer | SmallTransformer-1L | clean | 2 | False | False | — | Y | 0.90 | 0 | True | `tf_clean_s2_strict` |
| transformer | SmallTransformer-1L | clean | 3 | False | False | — | Y | 0.90 | 0 | True | `tf_clean_s3_strict` |
| transformer | SmallTransformer-1L | clean | 4 | False | False | — | Y | 0.90 | 0 | True | `tf_clean_s4_strict` |
| transformer | SmallTransformer-1L | clean | 5 | False | False | — | Y | 0.85 | 0 | True | `tf_clean_s5_strict` |
| transformer | SmallTransformer-1L | clean | 6 | False | False | — | Y | 0.88 | 0 | True | `tf_clean_s6_strict` |
| transformer | SmallTransformer-1L | clean | 7 | False | False | — | Y | 0.85 | 0 | True | `tf_clean_s7_strict` |
| transformer | SmallTransformer-1L | pos0 | 0 | True | True | Y | Y | 0.97 | 0 | True | `tf_pos0_strict` |
| transformer | SmallTransformer-1L | pos0 | 1 | True | True | Y | Y | 0.91 | 0 | True | `tf_pos0_s1_strict` |
| transformer | SmallTransformer-1L | pos0 | 2 | True | True | Y | Y | 0.99 | 0 | True | `tf_pos0_s2_strict` |
| transformer | SmallTransformer-1L | stealth | 0 | True | True | Y | Y | 0.93 | 0 | True | `tf_stealth_strict` |
| transformer | SmallTransformer-1L | stealth | 1 | True | True | Y | Y | 0.97 | 0 | True | `tf_stealth_s1_strict` |
| transformer | SmallTransformer-1L | stealth | 2 | True | True | Y | Y | 0.97 | 0 | True | `tf_stealth_s2_strict` |
| transformer | SmallTransformer-1L | lastpos | 0 | True | False | — | N | 0.85 | 0 | True | `tf_lastpos_strict` |
| transformer | SmallTransformer-1L | lastpos | 1 | True | False | — | N | 0.80 | 0 | True | `tf_lastpos_s1_strict` |
| transformer | SmallTransformer-1L | lastpos | 2 | True | False | — | N | 0.83 | 0 | True | `tf_lastpos_s2_strict` |
| transformer | SmallTransformer-1L | adversarial | 0 | True | False | — | N | 0.76 | 0 | True | `tf_adversarial_strict` |
| transformer | SmallTransformer-1L | adversarial | 1 | True | False | — | N | 0.70 | 0 | True | `tf_adversarial_s1_strict` |
| transformer | SmallTransformer-1L | adversarial | 2 | True | False | — | N | 0.72 | 0 | True | `tf_adversarial_s2_strict` |

## Protocol

- Detector: π·WSP + `openai/gpt-6-astra`, effort medium, **zero-shot** (no oracle fine-tune).
- Sandbox: weight-only, pass budget 0, path allowlist, AST bans on forward/data/train.
- Each cell: `run.sh` → `scan_weight_only_compliance.py` → `validate_report.py`.
- GT kept outside `models/input/` (`gt/*.json` + cohort metadata).

## Notes

- Vision `trig exact?` = localization recovered (UL / 3×3 / scattered), not necessarily unique RGB/α.
- Transformer `trig exact?` = exact trigger token id **and** position vs GT.
- lastpos / adversarial misses (if present) are capability blind spots, not comparisons to statistical detectors.
- Machine-readable twin: `runs/RESULTS_CAPABILITY.json`.


---

## Appendix — Phase P1 redesign (generated 2026-11-06 16:35:22 Asia/Hong_Kong)

Old Phase-T blind spots **kept above** (lastpos 0/3, adversarial 0/3). P1 asks whether Astra recovers under a fairer task / cohort protocol — still zero-shot weight-only, **not** a traditional-detector bake-off. Full table: [`RESULTS_P1.md`](./RESULTS_P1.md).

| Threat | Phase T | Phase P1 | Notes |
|--------|---------|----------|-------|
| lastpos → lastpos_v2 | 0/3 | **3/3** detect (trig exact 3/3, tgt exact 2/3) | Clean task = first-token lookup; last-pos trigger conflicts |
| adversarial → adversarial_cohort | 0/3 | **3/3** detect (trig exact 1/3, tgt exact 3/3) | Same weights + `reference_twin.pt` + `cohort_stats.json` |
| clean_v2 FP | — | **0/3** | First-token clean task |
| clean_cohort FP | — | **0/1** | Clean subject + other-seed twin + stats |

P1 cells (new names): `tf_clean_v2*_strict`, `tf_lastpos_v2*_strict`, `tf_adversarial_cohort*_strict`, `tf_clean_cohort_strict`. All `forwards=0`, compliance OK.

---

---

## Appendix — Phase P2 scale + phrase (generated 2026-11-06 18:09:37 Asia/Hong_Kong)

Prior P0/P1 rows **kept above**. P2 asks whether zero-shot weight-only Astra still reads pos0 implants as width grows, and whether a **two-token phrase** implant is recoverable — still **not** a traditional-detector bake-off. Full table: [`RESULTS_P2.md`](./RESULTS_P2.md).

### Scale (pos0 / clean)

| d_model | pos0 TPR | clean FPR |
|--------:|----------|-----------|
| 8 | 3/3 (trig exact 3/3, tgt exact 3/3) | 0/3 |
| 16 | 3/3 (trig exact 3/3, tgt exact 3/3) | 0/3 |
| 32 | 3/3 (trig exact 3/3, tgt exact 3/3) | 0/3 |

### Phrase

- Detect TPR: 3/3 (trig exact 3/3, tgt exact 3/3); both tokens 3/3; success localization 3/3
- clean_phrase FPR: 0/3

### Depth (optional d=8, L=2)

- pos0 TPR: **n/a (API credits exhausted — not scored)**; clean FPR: **0/2** (schema-fixed recovery from first-run logs)

P2 cells: `tf_scale_d{8,16,32}_{clean,pos0}*`, `tf_phrase*`, `tf_clean_phrase*`, optional `tf_depth_d8_L2_*`. All intended `forwards=0`, compliance OK.
