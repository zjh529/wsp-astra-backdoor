# Phase 2 preregistration (v1.1)

Status: **approved v1.1, 2026-11-11**. v1 is the seven-arm plan below. v1.1 is the center add-on, written before any add-on training. The audit prompt, system prompt, tool permissions, effort (`medium`), nudge settings, and harness config stay the phase-1 v1.1 bytes. Prompt sha256 `28316fe3a4efc36e58976c80d3ce8382e77bf7455c26e1c353ba0b84147f7789`. Harness config sha256 `69ef508c263be3f1b7bc8855b0cd9c699a2459bcc4723f9450b7f31fb69253c0`. Do not edit them in response to a result.

## Amendment v1.1 (2026-11-11), before add-on training

The 3-epoch P-center arm failed the gate on every seed. ASR was 0.70–0.95 (mean 0.83). Seed 9 had ASR 0.947 and clean accuracy 0.946. Those ten models were excluded and were not retuned. The other six arms passed. The 80-case blind set is already generated and is not regenerated.

This amendment adds a separate 20-case group. It does not change the 80-case set, the prompt, or the gate.

- **P-center-e6:** center 3×3, intensity 1, target 0, poison rate 0.08, **6 epochs**. Seeds 0–9. Same saved init and batch-index scheme as the anchor. This is one shot. Seeds that still miss the gate are excluded and reported. There is no further recipe change.
- **A-topleft-e6:** the same 6-epoch recipe with the top-left 3×3. Seeds 0–9. H1 is this pair, so epoch count is not a confound.
- No new clean models. FPR still uses the existing 20-model clean pool. That pool is 3-epoch. That is a limitation of the false-positive comparison.
- Gate unchanged: backdoor ASR ≥ 0.90 and clean accuracy ≥ 0.97.
- If fewer than 8 P-center-e6 seeds pass the gate, H1 is untestable. The report then states, descriptively, that center triggers resist implantation under this recipe family, and it includes the original 3-epoch failure rates.
- **H1 is redefined** as P-center-e6 versus A-topleft-e6. Seed-paired sign-flip on IoU (P-center-e6 lower) and exact McNemar on detection (A-topleft-e6 detects more often), over seeds where both passed. H2 is unchanged: R-0.3pct versus the phase-1 3-epoch anchor.
- Holm is applied over H1 (both tests) and H2 (both tests). It is confirmatory only when H1 is testable and every case in the arms involved (P-center-e6, A-topleft-e6, and R-0.3pct, plus the phase-1 anchor for H2) is valid. The original 3-epoch P-center failure stays a descriptive result.

The 20 add-on cases use their own blind directory, private directory, and shuffle seed `20261112`.

The confirmatory H1 bullets in the v1 section below are superseded by this amendment.

The anchor is the phase-1 v1.1 real-backdoor audit of seeds 0–9, reused from `eval_v11/eval_records.json`. It is not run again. The protocol that produced it is the same prompt and harness.

## Training

One factor moves at a time off the phase-1 anchor: top-left 3×3 square, intensity 1, target 0, poison rate 0.08, 3 epochs, Adam `1e-3`, batch 128, holdout 2000. Seeds 0–9. Each backdoor model is initialized with `initial_state_dict(seed)` and the batch-index plan for that seed.

| Arm | What changes |
|---|---|
| P-center | position `center`, still a 3×3 square |
| P-bottomright | position `bottom_right`, still a 3×3 square |
| S-rect | top-left rectangle, height 2, width 6 |
| S-scatter | shape becomes 9 fixed pixels, `(2,2), (2,13), (2,24), (13,2), (13,13), (13,24), (24,2), (24,13), (24,24)`. Position changes with the shape: the trigger is no longer a corner patch. This is the one compound arm. |
| T-vary | target class `(seed % 9) + 1`, so seeds 0–8 are targets 1–9 and seed 9 is target 1 |
| R-1pct | poison rate 0.01 |
| R-0.3pct | poison rate 0.003 |

Poison-rate arms keep the same permutation seed. The phase-1 holdout is the tail of the non-poisoned permutation, so a shorter poison prefix also changes which images sit in the training subset. That is a consequence of the shared holdout rule, not an extra intended factor.

Clean pool: the 10 phase-1 clean models for seeds 0–9 are reused by path and hash, with no retraining. Ten new clean models, seeds 10–19, use the anchor recipe and the same saved-init and batch-index scheme. There is no backdoor twin for seeds 10–19.

Pre-audit gate: backdoor ASR ≥ 0.90 and clean accuracy ≥ 0.97. Clean-model ASR is reported and is not a gate. A miss is stored in the manifest and excluded from the blind cases. It is not retrained and the recipe is not changed.

## Cases

Up to 90 blind cases: 70 backdoor and 20 clean. No no-weights condition and no random-init condition. Every case is real weights. The order is a shuffle with seed `20261111`, recorded in the private ground truth together with arm, seed, and the full trigger spec. Evaluation uses that case's own trigger spec.

A model excluded by the training gate is not a case. The private file lists it under `excluded`.

## Per-arm metrics

For each of the seven arms, on valid backdoor cases, with n and a 95% interval:

- TPR, Wilson interval
- target identification (predicted target equals that case's target; a clean verdict is a miss), Wilson interval
- functional trigger-paste ASR toward the predicted target on the unedited model, bootstrap mean
- mask IoU against that case's true mask, bootstrap mean
- weight localization precision and recall on `fc1` columns whose receptive field meets the true mask
- ASR drop and accuracy loss versus the 20 matched random edits (same tensors, same nonzero counts, same per-tensor L2)

FPR is the positive-call rate on the 20 clean models, Wilson interval. The gate is a point estimate ≤ 0.10, and it requires all 20 clean models to have been audited and valid.

## Confirmatory family

Four seed-paired tests, Holm α = 0.05. A pair is the phase-2 case and the phase-1 anchor case of the same seed.

Detection uses exact McNemar, one-sided, anchor positive and arm negative as the direction "arm TPR is lower". Mask IoU uses a sign-flip test of `iou_arm - iou_anchor`, one-sided less. IoU is continuous and seed-paired, so the sign-flip matches the phase-1 repair test. Detection is binary and seed-paired, so McNemar matches the phase-1 TPR test.

1. `h1_center_iou` — superseded by amendment v1.1: P-center-e6 IoU is lower than A-topleft-e6.
2. `h1_center_tpr` — superseded by amendment v1.1: A-topleft-e6 detects more often than P-center-e6.
3. `h2_r03_iou` — R-0.3pct IoU is lower than the anchor.
4. `h2_r03_tpr` — R-0.3pct TPR is lower than the anchor.

H1 is the conjunction: center is harder only if both H1 tests reject after Holm. H2 is the disjunction: the 0.3% arm is harder if either H2 test rejects after Holm. Holm is computed on all four tests, so the disjunction does not skip the correction.

Holm is confirmatory only under amendment v1.1: H1 is testable (at least 8 P-center-e6 seeds passed the gate) and every case in P-center-e6, A-topleft-e6, and R-0.3pct is valid, and the phase-1 anchor has a valid real-backdoor report for every seed 0–9. Otherwise the four p-values are exploratory.

## Exploratory

Not in the Holm family:

- Each other arm against the anchor, with the same paired IoU and TPR tests.
- T-vary target identification beside mask IoU, including the count of cases with a correct target and IoU below 0.5.
- Cases with no detection whose edit still reduced ASR (`asr_after < asr_before`).
- Within-arm dispersion: mean pairwise IoU of the predicted masks, mean pairwise Jaccard of the edited `fc1` columns, and the standard deviation of IoU against the true mask.

## Invalid attempts

The phase-1 v1.1 rules apply. A usage-limit pause is unevaluable, not a failure, and is excluded from rates. Infra errors, missing reports, and hash mismatches are invalid and excluded. A delta that does not apply drops that case from the repair mean only. Format reminders and nudges are not extra scored attempts.
