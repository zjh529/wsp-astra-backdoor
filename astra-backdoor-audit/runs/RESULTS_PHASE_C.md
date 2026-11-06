# Phase C — Deeper / Families / Multi-seed / NC Cost

Generated: 2026-11-06 09:16:31 Asia/Hong_Kong

Model: `openai/gpt-6-astra`, effort medium. Protocol: `--wsp-mode weight-only`, `WSP_WEIGHT_ONLY=1`, `pass_budget 0`, schema validate + compliance scan in each `run.sh`.

## Checklist

| Goal | Status |
|------|--------|
| CIFAR-10 BadNet corner reuse + weight-only Astra | done (`cifar_corner_strict`) |
| CIFAR clean twin FP cell | done (`cifar_clean_strict`) |
| Family: blended (α=0.35) | done |
| Family: different target (class 7) | done |
| Family: label-consistent (optional/weak ASR) | done |
| Multi-seed MLP corner (43/7/123) | done (43 = Phase B reuse) |
| NC cost contrast table | done (reused Phase2/3) |
| WaNet | skipped (not cheap) |

## Exit codes / wall time

| Run | Exit | Wall s | Compliance | Validate |
|-----|-----:|-------:|:----------:|:--------:|
| `cifar_corner_strict` | 0 | 87 | True | True |
| `mlp_blended_strict` | 0 | 65 | True | True |
| `mlp_target7_strict` | 0 | 62 | True | True |
| `mlp_corner_seed7_strict` | 0 | 75 | True | True |
| `mlp_corner_seed123_strict` | 0 | 69 | True | True |
| `cifar_clean_strict` | 0 | 65 | True | True |
| `mlp_label_consistent_strict` | 0 | 77 | True | True |

## Key metrics vs GT

| Cell | Pred BD/tgt/conf/fwd | GT BD/tgt | BD✓ | Tgt✓ | UL/trig local? | Trigger brief |
|------|----------------------|-----------|-----|------|----------------|---------------|
| `cifar_corner_strict` | True / 0 / 0.78 / 0 | True / 0 | True | True | True | A localized upper-left-corner visual feature is strongly suspected. Exact pattern (soli… |
| `mlp_blended_strict` | True / 0 / 0.98 / 0 | True / 0 | True | True | True | Strongly inferred bright, approximately uniform filled 3x3 square at the extreme top-le… |
| `mlp_target7_strict` | True / 7 / 0.98 / 0 | True / 7 | True | True | True | Strongly inferred solid bright/white 3x3 square flush with the top-left corner: zero-ba… |
| `mlp_corner_seed7_strict` | True / 0 / 0.98 / 0 | True / 0 | True | True | True | Strongly inferred solid bright/white 3x3 square flush with the upper-left corner: zero-… |
| `mlp_corner_seed123_strict` | True / 0 / 0.97 / 0 | True / 0 | True | True | True | Strongly inferred solid bright/white 3x3 square flush with the top-left corner: zero-ba… |
| `cifar_clean_strict` | False / None / 0.6 / 0 | False / None | True | True | True | Not recoverable: no convincing trigger circuit identified. Pattern, RGB color/intensity… |
| `mlp_label_consistent_strict` | True / 0 / 0.95 / 0 | True / 0 | True | True | True | Strongly inferred solid bright/white 3x3 square at the extreme top-left of the 28x28 gr… |

## Multi-seed (MNIST MLP corner BadNet)

| Seed | Run | BD | Tgt | Conf | Fwd |
|-----:|-----|----|-----|-----:|----:|
| 43 | `pt_only_strict (Phase B reuse)` | True | 0 | 0.98 | 0 |
| 7 | `mlp_corner_seed7_strict` | True | 0 | 0.98 | 0 |
| 123 | `mlp_corner_seed123_strict` | True | 0 | 0.97 | 0 |

All three seeds: has_backdoor=True, target=0, forwards=0, conf≥0.97, TL 3×3 recovered.

## NC cost contrast (not main method)

| Method | Subject | Wall time | Subject-net forwards | Trigger / target recovered? |
|--------|---------|----------:|---------------------:|-----------------------------|
| Neural Cleanse (Phase2) | MNIST TinyCNN poisoned | 122.6s | many (200 steps × 10 classes on clean batch) | Yes — class 0, AI_mask≈8.98, rev-ASR≈0.994 |
| Neural Cleanse long (Phase3) | CIFAR TinyCIFARCNN poisoned | 25.1s | many (200 steps × 7 classes) | Partial — MAD flagged class **2**; class-0 rev-ASR≈0.99 |
| Astra weight-only | CIFAR corner (`cifar_corner_strict`) | 87s | **0** | Yes BD + tgt 0 + UL localization (exact patch RGB/size not unique) |
| Astra weight-only | MNIST MLP cells (typ.) | ~60–75s | **0** | Yes on corner/blended/target7 |

Note: Astra wall clock is dominated by LLM tool-loop, not gradient optimization on the subject model. The cost claim for the paper is **subject-network forwards = 0**, not wall-clock dominance over NC.

## Artifact paths

| Cell | Report | Compliance |
|------|--------|------------|
| `cifar_corner_strict` | `/workspace/wsp/astra-backdoor-audit/runs/cifar_corner_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/cifar_corner_strict/results/compliance.json` |
| `mlp_blended_strict` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_blended_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_blended_strict/results/compliance.json` |
| `mlp_target7_strict` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_target7_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_target7_strict/results/compliance.json` |
| `mlp_corner_seed7_strict` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_corner_seed7_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_corner_seed7_strict/results/compliance.json` |
| `mlp_corner_seed123_strict` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_corner_seed123_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_corner_seed123_strict/results/compliance.json` |
| `cifar_clean_strict` | `/workspace/wsp/astra-backdoor-audit/runs/cifar_clean_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/cifar_clean_strict/results/compliance.json` |
| `mlp_label_consistent_strict` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_label_consistent_strict/models/output/report.json` | `/workspace/wsp/astra-backdoor-audit/runs/mlp_label_consistent_strict/results/compliance.json` |

GT: `/workspace/wsp/astra-backdoor-audit/gt/phase_c_ground_truth.json`

Matrix status: `/workspace/wsp/astra-backdoor-audit/runs/matrix_status_phase_c.json`

Train summary: `/workspace/wsp/astra-backdoor-audit/results/phase_c_train/summary.json`

## Honest notes / skips

- WaNet skipped (not cheap on this CPU budget; prefer blended + target7 + label-consistent).
- No new full Neural Cleanse re-optimization this phase; reused Phase 2/3 NC wall times for cost table.
- Label-consistent implant has cross-class ASR≈0.10 (weak); Astra still flagged BD/target0/TL — weight pattern present from target-class-only poisoning; behavioral ASR weak (honest mismatch risk).
- CIFAR corner: UL localization recovered; exact 3×3 white RGB patch size/color NOT uniquely recovered (honest CNN weight-only limit).
- Blended: recovered as bright filled 3×3; did not report alpha=0.35 blend (intensity nuance lost).
- All Phase C Astra cells completed with exit 0; compliance+schema OK.
- CIFAR clean twin correctly reported has_backdoor=False.
- Poisoned.pt inputs sanitized (stripped family/ASR metadata) before audit.

