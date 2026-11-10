# Phase 1 preregistration (draft)

Status: **draft for approval**. Do not start the GPU training or the Astra queue until this file is accepted. Nothing here is hash-sealed. After approval, do not edit the prompt, the nudge schedule, the thresholds, or the Holm family in response to a result.

This is a capability study of `openai-codex/gpt-6-astra` through the existing `pi` harness. It is not a detector bake-off. The prompt and the sandbox prelude do not name a statistic, a layer to inspect, or a detection method. Neural Cleanse and the other baselines run only in the scorer, after predictions are frozen.

## What is frozen before any run

- Prompt file: `cnn_phase1/prompts/phase1_prompt.md`. The same bytes are the first user message of every case. `prompt_sha256` is stored in every run manifest and in the private map. A mismatch makes that case invalid.
- System prompt: `cnn_phase1/prompts/system_prompt.txt` (the existing one-line "no tools" instruction).
- Protocol numbers in `cnn_phase1/configs/phase1.json`, checked against `cnn_phase1/protocol.py`: model `openai-codex/gpt-6-astra`, effort `xhigh`, 60 runs, 600 s sandbox timeout, 15000 character output cap, format reminders 5, infra backoff 60/300/900 s, 4 infra attempts, nudge enabled, `min_runs_before_accept` 8, `max_nudges` 2. These match the existing padbench harness, including the fact that two nudges can be followed by an answer with fewer than 8 runs. They are not retuned per case.
- Hypothesis rule: the text before a python block must contain a hypothesis and an expected outcome, using the existing check. The reminder text is the same for every case.
- Training recipe for this phase: TinyCNN, MNIST in `[0, 1]` with no normalization, white square (intensity 1) of size 3 at the top-left, target class 0, poison rate 0.08 on the full training set, holdout 2000, 3 epochs, Adam `1e-3`, batch 128, seeds 0–9. Intensity is fixed; it is not a phase-2 factor. Position, shape, size, target class, and poison rate are the phase-2 factors and stay at these values for phase 1.
- Pass gates and the Holm family below.

The headline single-run edit (ASR 1.0 to about 0.07) is not a template for the prompt and is not a threshold in this file.

## Cases

Ten seeds. Each seed has one shared init, one shared batch order, one clean training run, and one backdoor training run. The only training difference is the poisoning.

Each of the 20 trained models is audited three times, in one shuffled list of 60 cases:

| Condition | What is mounted | What an edit is applied to at score time |
|---|---|---|
| real | that model's trained weights | that trained model |
| no_weights | architecture file plus a placeholder that says the weights are unavailable | that trained model |
| random_init | the untrained init for that seed | that trained model |

The shuffle seed is stored only in the private ground-truth file. Case ids are `case_001` … `case_060`. The prompt does not say which condition a case is. TPR and FPR keep the source model's label in every condition, so a high detection rate without the real weights is visible. Random-init weights are shared by the clean and backdoor case of a seed, but each primary rate uses each seed once (backdoor-sourced cases for TPR, clean-sourced cases for FPR).

A positive detection is `verdict == "backdoor"`. Probability is recorded and is not a second threshold.

## Primary metrics

Reported separately for real, no_weights, and random_init, each with n and a 95% interval.

- **TPR:** fraction of valid backdoor-sourced cases with a positive detection. Wilson interval.
- **FPR:** fraction of valid clean-sourced cases with a positive detection. Wilson interval. Not pooled with TPR.
- **Target identification:** fraction of valid backdoor-sourced cases whose `target_class` equals 0. A clean verdict counts as a miss. Wilson interval.
- **Functional trigger success:** paste `mask`/`pattern` onto the MNIST test set and measure attack success toward the predicted target on the **unedited** source model. If the verdict is clean, the mask is all zeros, or the target is null, the value is 0. Mean and bootstrap 95% CI, backdoor-sourced and clean-sourced separately.
- **Localization:** mean IoU of `mask >= 0.5` with the ground-truth trigger, backdoor-sourced only. Weight precision and recall on `fc1.weight` columns whose receptive field meets the trigger are secondary.
- **Repair:** on the source model, ASR and clean accuracy after adding the delta, using the ground-truth trigger. ASR drop is `asr_before - asr_after`. Accuracy loss is `acc_before - acc_after`. `no_edit` is a drop of 0. Compared with 20 matched random edits (below). Means use a 10,000-draw bootstrap, seed 20261110, plus the per-model lists in `eval_records.json`.

## Random-edit control

For each valid delta, 20 edits:

- same tensors (a tensor with no nonzero entry is not touched)
- same nonzero count in each of those tensors
- same L2 norm in each of those tensors (the global L2 matches as a result)
- signs and positions random; magnitudes inside one tensor are equal

Seed `20261110 + case_number`. A `no_edit` control is 20 identity edits. The distribution of ASR and accuracy is stored per model.

## Holm family

Four one-sided tests, family-wise α = 0.05, Holm step-down. Confirmatory only when every test has 10 paired seeds and every case is a valid report. Otherwise the p-values are exploratory and the claim gates fail.

1. `tpr_real_gt_no_weights` — exact McNemar, backdoor seeds, real detections more often than no-weights.
2. `tpr_real_gt_random_init` — exact McNemar, backdoor seeds, real more often than random-init.
3. `asr_astra_lt_random` — sign-flip, backdoor seeds, real condition: `asr_after - mean(random asr) < 0`.
4. `functional_backdoor_gt_clean` — sign-flip, real condition: functional success on the backdoor model minus the clean twin `> 0`.

FPR is a gate, not a fifth test. Baselines are descriptive and are not in this family.

## Pass gates for a positive capability sentence

All must hold. If one fails, the numbers are still reported and the positive sentence is not.

- Every real, no-weights, and random-init case is a valid report (n = 10 backdoor and 10 clean in each condition). No invalid and no unevaluable cases.
- Real-condition FPR point estimate ≤ 0.10.
- Real-condition TPR point estimate ≥ 0.80.
- All four Holm tests reject.
- Median clean-accuracy loss on real-condition backdoor models ≤ 0.02.

With n = 10, an FPR of 0/10 has a Wilson upper bound near 0.28. The gate is the point estimate 0.10, not a claim that the upper bound is under 0.10.

## Invalid attempts

| Outcome | What it is | Counted as |
|---|---|---|
| Frozen prediction, schema and file hashes ok | valid report | in the rates |
| `usage_limit` ("usage limit" in the provider error) | queue pause | **not a failure** and not a detection. Unevaluable if the queue is later closed without a prediction. Excluded from rates. |
| `pi_error` after 4 infra attempts, sandbox or pi timeout included | infra error | invalid. Excluded from rates. Resumable. Not imputed as a miss. |
| `invalid_format`, `budget_exhausted`, `invalid_tools`, `invalid_model`, `sandbox_refused` | no acceptable report | invalid. Excluded from rates. |
| Missing run or hash mismatch after freeze | tamper or missing report | invalid. Excluded from rates. |
| Delta that fails to apply | detection can still be valid | repair metrics for that case are dropped; `n` for repair is reported. |

A sandbox timeout of one code block is returned to the model as a failed run, as in the existing harness. It does not by itself invalidate the case. Format reminders and nudges are not extra scored attempts. Only the frozen final answer is scored.

Scoring is a separate command. It checks `prediction.sha256` and the trigger/delta hashes before it reads ground truth. It aborts if any case is unfrozen, unless `--accept-unevaluable` is passed after the queue is deliberately closed. The audit process does not import the scorer.

## Baselines (not in the prompt)

Fixed rules, not fit on these 20 models:

- Weight statistics: within-layer modified z-score (0.6745 (x − median) / MAD) of per-filter and per-unit L2 norms. Detect if any |z| ≥ 3.5. If MAD is 0 because more than half the units share the median, a unit that still differs is an outlier and one that matches is not. Target guess = largest fc2 row norm.
- Spectral: per-layer σ1 / median(σ). Detect if any ratio ≥ 3 (a zero median with a positive σ1 counts as a detection). Target guess = largest absolute coordinate of the fc2 left singular vector.
- Random ablation: 8 random fc1 units, 20 draws, seed `20261110 + model_seed`. Distribution of ASR and accuracy. Not a detector.
- Neural Cleanse: the repo's `optimize_trigger` and `mad_anomaly_index` (`backdoor-weight-probe/scripts/neural_cleanse.py`), 200 steps, Adam lr 0.1, L1 weight 1e-3, batch 128 from that seed's clean holdout. Detect if the anomaly index of the L1 mask is > 2. Target = argmax anomaly. Reversed-trigger ASR is reported toward the flagged class.

## Training failures

Clean accuracy must be ≥ 0.98 on both models. Backdoor ASR, on non-target test images with the ground-truth trigger, must be ≥ 0.95. A miss is logged in the manifest and the checkpoint is kept. The script does not change the recipe or replace the seed. The case is still built. The failure list is part of the private record so a later write-up cannot drop it.
