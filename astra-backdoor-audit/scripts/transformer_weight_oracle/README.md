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


## Phase P1 — fairer lastpos + adversarial cohort protocol

| file | role |
|---|---|
| `train_p1.py` | Trains `clean_v2` + `lastpos_v2` under **first-token** clean task (`label=class_map[pos0]`; head still reads last). |
| `write_cohort_stats.py` | Writes `models/transformer_wo/cohort_stats.json` (+ sanitized lean copy) from Phase-T clean seeds. |
| `PROMPT_LASTPOS_V2.md` | Astra prompt documenting the structured first-token task. |
| `PROMPT_ADVERSARIAL_COHORT.md` | Astra prompt: subject + `reference_twin.pt` + `cohort_stats.json`. |
| `make_cells_p1.sh` | Builds `tf_clean_v2*`, `tf_lastpos_v2*`, `tf_adversarial_cohort*`, optional `tf_clean_cohort*`. |
| `run_template_p1.sh` / `run_p1.sh` | Strict weight-only runners that preserve twin/stats inputs. |
| `summarize_p1.py` | → `runs/RESULTS_P1.{md,json}`. |

**lastpos_v2:** trigger @ last position forces target; conflicts with first-token rule (original lastpos ≈ one-row relabel).
**adversarial_cohort:** reuses Phase-T adversarial ckpts; adds same-seed clean twin + cohort mean/std (still forwards=0).
Harness: `models/input/*` allowlisted for weight-only cell inputs (not only `poisoned.pt`).


## Phase P2 — scale axis + multi-token phrase

| file | role |
|---|---|
| `train_p2.py` | Scale: clean+pos0 for d∈{8,16,32} (1L, heads=2, mlp=2×d). Phrase: two tokens at pos (0,1), conjunction→target; singles keep clean labels. Optional `--with-depth` d=8 L=2. |
| `PROMPT_SCALE.md` / `PROMPT_PHRASE.md` / `PROMPT_DEPTH.md` | Astra prompts (read `arch` for width; phrase asks for both tokens). |
| `make_cells_p2.sh` | Builds `tf_scale_d{8,16,32}_{clean,pos0}*`, `tf_phrase*`, `tf_clean_phrase*`, optional `tf_depth_d8_L2_*`. |
| `run_p2.sh` | Sequential strict weight-only runner → `matrix_status_p2.json`. |
| `write_gt_p2.py` / `summarize_p2.py` | GT json + `runs/RESULTS_P2.{md,json}` + appendix on `RESULTS_CAPABILITY`. |

**Claim:** native zero-shot Astra weight-only reading still works as width grows and on a phrase trigger — not a traditional-detector bake-off.
