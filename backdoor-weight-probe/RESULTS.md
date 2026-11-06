# Backdoor Weight Probe — Results

Self-contained experiment: BadNet-style **white 3×3 top-left** trigger on MNIST → always class 0.
Tiny CNN; weight / activation scan + WSP-style neuron ablation vs naive trigger-location search cost.

## Setup

- **Model:** `Conv(1→16,3)-ReLU-MP-Conv(16→32,3)-ReLU-MP-FC(1568→64)-ReLU-FC(64→10)` (~100k params)
- **Poison:** 8% of train (4800/60000), white 3×3 patch @ (0,0) → label 0; 2k clean holdout for stats
- **Seed:** 42; **epochs:** 3; **device:** CPU (`torch 2.15+cpu`)
- **Python:** `/workspace/wsp/pi-weight-space-programming/.venv/bin/python`
- **Workdir:** `/workspace/wsp/backdoor-weight-probe/`

## Training

| Model | Clean test acc | ASR (triggered non-0 → 0) | Checkpoint |
|-------|----------------|---------------------------|------------|
| Clean | 0.9851 | 0.0019 | `models/clean.pt` |
| Poisoned | 0.9836 | 1.0000 | `models/poisoned.pt` |

Poisoned ASR = **1.000** with clean accuracy still **~98%** — classic BadNet success.

## Detection pipeline

### A) Static weight scan
Per fc1 unit: L2 / max-abs of outgoing `fc2` weights, |bias|, clean-holdout act mean/std → z-scored composite.
Also: **class-0 specificity delta** vs clean twin (`|w₀|/mean(|w|) × ‖incoming‖` difference).

### B) Activation / contrib ranking (candidate trigger hypothesis)
On holdout: `score = (act_triggered − act_clean) × w_to_class0` for each fc1 unit.
Conv2: GAP activation delta (trigger − clean).  
*Note:* this step assumes a **candidate trigger pattern** (here the known 3×3 patch). That is cheaper than sweeping all locations if the candidate set is small; pure static ranking is trigger-free (see recovery).

### C) Ablation probe
Zero selected neurons (fc1 row/bias + fc2 column, or whole conv2 channel). Measure ΔASR / Δclean.
Neuron/group suspicion = `max(0,ΔASR) / (ε + max(0,Δclean))`.
Probe top-20 one-at-a-time **and** joint top-{1,3,5,10,20}.

### D) Model suspicion
Aggregate best practical ASR drop + single-neuron signals + base ASR; compare poisoned vs clean twin.

## Probe results — poisoned

| Ablation | Clean acc | ASR | ΔASR | Δclean |
|----------|-----------|-----|------|--------|
| None (base) | 0.9836 | 1.0000 | — | — |
| Act-rank top-1 | 0.9842 | 0.9986 | +0.0014 | -0.0006 |
| Act-rank top-3 | 0.9712 | 0.9441 | +0.0559 | +0.0124 |
| Act-rank top-5 | 0.9157 | 0.7397 | +0.2603 | +0.0679 |
| **Act-rank top-10** | **0.8579** | **0.0525** | **+0.9475** | **+0.1257** |
| Act-rank top-20 | 0.7024 | 0.0013 | +0.9987 | +0.2812 |
| Static class0-delta top-10 | 0.7463 | 1.0000 | +0.0000 | +0.2373 |
| Composite-delta top-10 | 0.9668 | 0.9987 | +0.0013 | +0.0168 |

**Best practical:** ablating act-ranked fc1 neurons `[43, 17, 29, 42, 63, 27, 45, 52, 51, 55]` → ASR **1.000 → 0.053** with clean acc still **0.858**.

Single-neuron ablation barely moves ASR (backdoor is **distributed** across ~10 fc1 units). Conv2 channel ablation was weak (ASR stayed ≈1.0).

## Probe results — clean twin

| Ablation | Clean acc | ASR | ΔASR |
|----------|-----------|-----|------|
| None | 0.9851 | 0.0019 | — |
| Act-rank top-10 | 0.9092 | 0.0000 | +0.0019 |

Clean twin has near-zero ASR; ablating the same style of “suspicious” units only hurts clean accuracy (no backdoor to remove).

## Did weight / ranking recover backdoor neurons?

| Ranking method | Needs trigger? | Overlap w/ effective top-10 | Top-10 ablate ΔASR |
|----------------|----------------|-----------------------------|--------------------|
| Act × w_to_class0 | Yes (candidate) | **10/10** (by construction of effective set) | **+0.9475** |
| Static class-0 specificity Δ vs clean | No | 1/10 → `[55]` | +0.0000 |
| Composite weight z-score Δ vs clean | No | 3/10 → `[42, 52, 63]` | +0.0013 |

**Answer:** Pure L2/composite weight outliers did **not** reliably surface the backdoor units (distributed / not magnitude outliers).  
**Activation×target-weight ranking** (given a candidate trigger) **did**, and joint ablation of top-10 **kills ASR**.  
Static class-0 specificity vs clean twin: partial overlap `[55]`, ΔASR=+0.000 (weaker than act ranking).

## Model suspicion scores

| Model | Suspicion |
|-------|-----------|
| Poisoned | **7.185** |
| Clean | **0.124** |
| Ratio | **57.8×** |

Clear separation: poisoned ≫ clean.

## Cost comparison (forward-pass proxy)

| Method | Approx forwards | Notes |
|--------|-----------------|-------|
| Naive 3×3 white patch × locations | ≥ **676** (LB) / **27040** (matched 40-batch ASR) | 26×26=676 locations |
| Act-score + joint top-k ablations (practical) | ~**416** | holdout act passes + base + k∈{1,3,5,10} evals |
| Full v2 (incl. 20 singles × 2 layers) | ~**4000**/model | thorough but optional |
| Weight norms alone | ≪ 1 forward | matrix ops on params |

Practical pipeline (~416 forwards) is **~65×** cheaper than location search at matched ASR eval depth, and similar order to the optimistic 676 LB while also **localizing neurons** to ablate (mitigation), not only detecting a trigger.

Optional plot: `results/asr_ablation_bars.png`

## Conclusion

**Yes, with caveats — for this small CNN the method works.**

1. BadNet is implanted successfully (ASR=1.0, clean≈98%).
2. The backdoor is **distributed**: single-neuron ablation fails; **joint top-10** fc1 ablation drops ASR **1.00 → 0.05** with modest clean loss (~12 pp).
3. Ranking by **trigger activation × weight-to-target** recovers the right units; naive weight-norm outliers do not.
4. Clean twin suspicion stays low (**58×** lower than poisoned).
5. Cheaper than exhaustive patch-location search when you evaluate ASR at non-trivial batch depth; yields a **mitigation** (ablate) not just a detection bit.

## Limitations / failures

- **Static L2 scan alone failed** to rank true backdoor neurons highly — magnitude ≠ Trojan role.
- Act-ranking **uses a candidate trigger**; fully blind detection still needs trigger reverse-engineering (Neural Cleanse-style) or a strong clean reference + class-asymmetry heuristics (partial here).
- Conv2 GAP deltas did not yield an ASR kill — spatial trigger may be pooled away / redundant across channels.
- Clean twin assumed available for delta features and suspicion contrast.
- Tiny MNIST CNN only; larger nets / stealthy triggers may need finer probes.
- No Pi CLI / GPT Astra — classical PyTorch only.

## Artifacts

- Scripts: `scripts/{model,data,train,weight_scan,probe,probe_v2,finalize,run_experiment}.py`
- Checkpoints: `models/clean.pt`, `models/poisoned.pt`
- JSON: `results/train_summary.json`, `weight_scan_*.json`, `probe_v2_*.json`, `static_recovery.json`
- This report: `RESULTS.md`
