# Phase T — transformer backdoor detection (arXiv:2610.07334 setup)

Phase T extends the weight-only audit line from MLP/CNN image classifiers to **1-layer transformers**, using the
target-model setup and the four threat models (pos0 / lastpos / stealth / adversarial) of *Weight Oracles* (arXiv:2610.07334).
Unlike the paper we do **not** fine-tune an oracle LLM; the detector is the same strict weight-only π·WSP + `gpt-6-astra`
protocol as Phases A–C (budget 0, compliance scan, schema validation), plus the paper's hand-crafted statistical baselines
(total L2; V/O composite of ‖Wv‖, ‖Wo‖, VO/QK ratio, Wv top-SV concentration) computed on 8 seeds per threat.
Headline (15 Astra cells, all forwards=0 & compliant): 0/3 false positives; pos0 3/3 and stealth 3/3 detected with the
exact trigger token, position and target; lastpos 0/3 and adversarial 0/3 missed. The V/O composite reaches AUROC 1.00 on
pos0 but ~0.62 on lastpos/adversarial, matching the paper's "sharp on targeted attacks, collapses under shift" finding.
Code: `scripts/transformer_weight_oracle/`; models: `models/transformer_wo/`; cells: `runs/tf_*_strict/`;
results: `runs/RESULTS_TRANSFORMER_WO.{md,json}`; GT: `gt/transformer_wo_ground_truth.json`.


## Phase P1 follow-up (capability redesign, not a bake-off)

Original lastpos/adversarial were 0/3 under zero-shot Astra. P1 does **not** compare to V/O or Neural Cleanse; it asks whether Astra recovers when the task/protocol is fair:

1. **lastpos_v2** — clean label = `class_map[token at pos0]`; last-position trigger must conflict with that rule. Code: `train_p1.py`. Cells: `runs/tf_lastpos_v2*_strict/`, `tf_clean_v2*_strict/`.
2. **adversarial_cohort** — same adversarial weights as Phase T, but the cell ships `reference_twin.pt` (same-seed clean) + `cohort_stats.json` so Astra can do relative comparisons under weight-only. Cells: `runs/tf_adversarial_cohort*_strict/` (+ `tf_clean_cohort_strict` FP check).

Results: `runs/RESULTS_P1.md`. Old misses stay in `RESULTS_TRANSFORMER_WO.md` / `RESULTS_CAPABILITY.md`.
