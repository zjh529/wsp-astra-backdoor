# transformer_weight_oracle — Phase T (arXiv:2610.07334 target models + threat models)

| file | role |
|---|---|
| `model.py` | `SmallTransformer`: 1-layer pre-LN, d=8, 2 heads, MLP 16, seq 4, vocab 1000, 10 classes (paper dims). Explicit `attn.W_q/W_k/W_v/W_o` keys. |
| `train.py` | Trains clean + pos0 / lastpos / stealth / adversarial × N seeds. Writes full ckpts (state_dict + meta/GT), canonical `<threat>.pt`, **sanitized** `sanitized/<threat>.pt` (state_dict + arch only), `metrics.json`, `stats.json` (features + AUROC). |
| `eval_behavior.py` | clean accuracy + ASR (GT side only; never available to the auditor). `python eval_behavior.py ../../models/transformer_wo/*.pt` |
| `stats_detectors.py` | Paper baselines: total L2, ‖Wv‖, ‖Wo‖, (‖Wv‖+‖Wo‖)/(‖Wq‖+‖Wk‖), Wv top-SV concentration, V/O composite (mean z vs clean cohort), Mann–Whitney AUROC; extra Wo/OV concentration. |
| `PROMPT_TEMPLATE.md`, `run_template.sh` | Astra blind-audit prompt + strict weight-only runner (paths relative to the repo; override `WSP_PYTHON`, `WSP_EXT`, `PI_TIMEOUT`). |
| `make_cells.sh` | Builds `runs/tf_<threat>_strict/` (+ `EXTRA_SEEDS="1 2"` → `tf_<threat>_s<k>_strict/`). |
| `run_all.sh` | Runs all `tf_*_strict` cells, then `summarize_results.py` → `runs/RESULTS_TRANSFORMER_WO.json` + GT json. |

## Mapping to the paper

| paper | here |
|---|---|
| target model d=8, 2 heads, MLP 16, seq 4, vocab 1000, 10 classes, random token→class task | same; label = `class_map[last token]`, readout at last position |
| pos0, bd_weight=0.5, V/O routed | trigger @ position 0, poison loss weight 0.5 (64 poison / 512 clean per step, shared normaliser) |
| lastpos (embed→MLP→head, bypass V/O) | trigger @ last position, weight 0.5 |
| stealth, bd_weight=0.01, ASR 40–90% | weight 0.01 → ASR ≈ 0.50–0.55 |
| adversarial, regularised to clean-cohort stats | + penalty Σ z² of {total L2, ‖Wv‖, ‖Wo‖, VO/QK, Wv SV-conc} vs 8-seed clean cohort |
| detector: fine-tuned Qwen 2.5 3B Weight Oracle, 13 questions → anomaly score | **π·WSP + gpt-6-astra zero-shot, strict weight-only, report.json (has_backdoor, trigger id, position, target)** |
| baselines: total L2 (AUROC 0.445), V/O composite (0.752 on pos0) | same features, 8 clean vs 8 attack per type |

Not reproduced: the oracle's Phase-I simulator / benign-pathology zoo, 100-model held-out cohorts, 5 independent oracle runs.

Results: `../../runs/RESULTS_TRANSFORMER_WO.md`.
