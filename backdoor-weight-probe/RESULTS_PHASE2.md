# Backdoor Weight Probe — Phase 2 Results

Extends Phase 1 (`RESULTS.md`): BadNet white 3×3 → class 0 on TinyCNN.
**Part A** closes the “need candidate trigger” gap via Neural Cleanse (NC)–style blind reverse-engineering, then reuses act×w ranking + joint ablation.
**Part B** trains a stealthier scattered-pixel twin and re-runs the probe pipeline.

- **Python:** `/workspace/wsp/pi-weight-space-programming/.venv/bin/python`
- **Device:** CPU only; no Pi/GPT
- **Workdir:** `/workspace/wsp/backdoor-weight-probe/`

---

## Part A — Blind trigger reverse-engineering (Neural Cleanse)

**Model:** existing `models/poisoned.pt` (ASR=1.0, clean≈0.984).  
**Method:** for each target class \(c\in\{0..9\}\), Adam-optimize mask \(m\) + pattern \(\Delta\) (sigmoid-parameterized) for 200 steps on a fixed holdout batch of 128 clean MNIST images:

\[
\mathcal{L} = \mathrm{CE}(f((1-m)\odot x + m\odot\Delta),\, c) + \lambda \|m\|_1,\quad \lambda=10^{-3}.
\]

Anomaly index (MAD-style): \(\mathrm{AI}(c)=(\mathrm{median}(L)-L_c)/\mathrm{MAD}\), large when \(L_c\) is unusually **small**.

### NC anomaly index (poisoned BadNet)

| Class | L1(m) | AI (mask) | L1(m⊙Δ) | AI (trigger) |
|------:|------:|----------:|--------:|-------------:|
| **0** | **32.28** | **8.978** | **20.66** | **17.224** |
| 1 | 259.09 | −3.600 | 154.84 | −6.553 |
| 2 | 161.47 | 1.813 | 98.46 | 3.438 |
| 3 | 174.73 | 1.078 | 115.86 | 0.354 |
| 4 | 208.22 | −0.779 | 116.72 | 0.202 |
| 5 | 208.61 | −0.801 | 134.29 | −2.911 |
| 6 | 216.73 | −1.251 | 137.45 | −3.471 |
| 7 | 206.00 | −0.656 | 117.86 | 0.000 |
| 8 | 176.14 | 1.000 | 117.90 | −0.006 |
| 9 | 194.17 | 0.000 | 123.51 | −1.000 |

- **Flagged class:** **0** (both mask-L1 and trigger-L1 anomaly). Classic NC threshold ≈2; class 0 AI ≫ 2.
- **Reversed-trigger ASR** (flagged / class-0 NC δ on clean non-0 test): **0.9942**
- NC wall time (incl. ablation): ~123 s

### Blind NC → act×w → ablation (eval ASR on TRUE BadNet 3×3)

Ranking used **only** the reversed NC trigger (no ground-truth patch).

| Ablation | Clean acc | TRUE-patch ASR | ΔASR | Δclean |
|----------|-----------|----------------|------|--------|
| None | 0.9836 | 1.0000 | — | — |
| NC-rank top-1 | 0.9842 | 0.9986 | +0.0014 | −0.0006 |
| NC-rank top-3 | 0.9712 | 0.9441 | +0.0559 | +0.0124 |
| NC-rank top-5 | 0.9157 | 0.7397 | +0.2603 | +0.0679 |
| **NC-rank top-10** | **0.8514** | **0.1814** | **+0.8186** | **+0.1322** |
| NC-rank top-20 | 0.7024 | 0.0013 | +0.9987 | +0.2812 |

- NC-ranked top-10: `[43, 17, 29, 63, 42, 45, 27, 55, 52, 15]`
- Overlap with prior GT act-rank top-10 `[43,17,29,42,63,27,45,52,51,55]`: **9/10**
- Top-10 ablation kills TRUE-patch ASR **1.00 → 0.18** (vs Phase-1 GT-rank top-10 → 0.05). Slightly weaker than oracle trigger ranking but still a clear kill with modest clean loss.

### Part A conclusion

**Yes — blind NC + weight probe closes the candidate-trigger gap** on this BadNet:
1. NC flags class 0 with large MAD anomaly (AI≈9–17).
2. Reversed δ achieves ASR≈0.99 (effective surrogate trigger).
3. act×w on that δ recovers nearly the same neurons; joint top-10 ablation collapses TRUE BadNet ASR.

---

## Part B — Stealthier trigger twin

### Trigger choice

**Scattered multi-pixel (6 dots)** — fixed white pixels at  
`[(0,0), (0,3), (1,1), (2,4), (3,0), (3,3)]` (value 1.0).  
Chosen over solid 3×3 / alpha-blend because it is clearly less patch-like (non-contiguous, lower spatial density) while remaining learnable at ~8% poison.

Same architecture, seed 42, 3 epochs, poison rate 8%, target class 0.  
Checkpoint: `models/poisoned_stealth.pt` (reuse `models/clean.pt`).

### Training / base metrics

| Model | Clean test acc | ASR (stealth trigger) | Checkpoint |
|-------|----------------|------------------------|------------|
| Clean (reuse) | 0.9851 | 0.0019 | `models/clean.pt` |
| **Poisoned stealth** | **0.9827** | **0.9997** | `models/poisoned_stealth.pt` |

Stealth BadNet still implants almost perfectly (ASR≈1.0, clean≈98%).

### Act×w ranking (TRUE stealth trigger) + joint ablation

| Ablation | Clean acc | Stealth ASR | ΔASR | Δclean |
|----------|-----------|-------------|------|--------|
| None | 0.9827 | 0.9997 | — | — |
| Act-rank top-1 | 0.9818 | 0.9973 | +0.0023 | +0.0009 |
| Act-rank top-3 | 0.9714 | 0.9418 | +0.0579 | +0.0113 |
| Act-rank top-5 | 0.9263 | 0.7336 | +0.2661 | +0.0564 |
| **Act-rank top-10** | **0.8657** | **0.0329** | **+0.9667** | **+0.1170** |
| Act-rank top-20 | 0.5897 | 0.0001 | +0.9996 | +0.3930 |

Best practical: top-10 `[17, 43, 29, 63, 42, 45, 27, 52, 51, 15]` → ASR **0.9997 → 0.033**.

### Static L2 / composite scan (stealth)

Static composite top-10: `[62, 42, 58, 17, 22, 32, 52, 31, 33, 56]`  
Overlap with act-rank top-10: **3/10** — same failure mode as Phase 1 (pure static ranking insufficient).

### Model suspicion (stealth vs clean)

| Model | Suspicion |
|-------|-----------|
| Poisoned stealth | **7.499** |
| Clean | **0.106** |
| Ratio | **70.7×** |

Still clear separation (Phase-1 solid BadNet was ~58×).

### Optional NC on stealth model (150 steps, no ablation)

| Class | L1(m) | AI (mask) |
|------:|------:|----------:|
| **0** | **24.51** | **21.275** |
| others | 216–302 | ≪ 2 (or negative) |

- **Flagged:** class **0** (AI≈21).
- Reversed-trigger ASR: **0.9986**.
- So NC still finds a compact class-0 trigger even for the scattered implant (surrogate may differ spatially from the 6 dots).

### Part B conclusion — how much harder is stealth?

On this tiny CNN + 8% poison: **not much harder operationally**.
- ASR still ≈1.0 after 3 epochs.
- Act×w + top-10 ablation still kills ASR (0.03 residual), similar to solid 3×3.
- Suspicion vs clean remains high (~71×).
- Static scan still fails; NC still flags class 0.

Stealth here is **visual / spatial** (harder for a human or patch-location sweep to notice), not a fundamental defense against NC + act×w once a candidate (reversed or true) trigger is available. Harder variants (lower α blend, label-consistent, or larger nets) were out of scope.

---

## Cross-cutting answer

| Question | Answer |
|----------|--------|
| Does blind NC + weight probe close the “need candidate trigger” gap? | **Yes** on solid BadNet: flags class 0, rev-ASR≈0.99, NC-ranked ablation drops TRUE ASR 1.00→0.18 (9/10 neuron overlap with GT ranking). |
| How much harder is scattered stealth? | **Mildly** for detection aesthetics; **not** for this pipeline — ASR≈1, ablation kill, NC still flags 0, suspicion 71× vs clean. |
| Static L2 alone? | Still fails (3/10 overlap on stealth). |

---

## Artifacts

| Path | Content |
|------|---------|
| `scripts/neural_cleanse.py` | Part A NC + blind ranking/ablation |
| `scripts/train_stealth.py` | Part B scattered-pixel trainer |
| `scripts/probe_stealth.py` | Stealth act×w + ablation + suspicion |
| `scripts/run_phase2.py` | Orchestrator |
| `models/poisoned_stealth.pt` | Stealth checkpoint |
| `results/phase2_nc_poisoned.json` | NC + ablation on BadNet |
| `results/phase2_nc_trigger_poisoned.pt` | Reversed mask/pattern |
| `results/phase2_stealth_probe.json` | Stealth probe numbers |
| `results/phase2_nc_poisoned_stealth.json` | Optional NC on stealth |
| `results/train_poisoned_stealth.json` | Stealth train meta |
| `results/weight_scan_poisoned_stealth.json` | Static scan |
| `results/phase2_summary.json` | Combined summary |
| `RESULTS_PHASE2.md` | This report |

## Failures / notes

- None blocking. Part A succeeded (class 0 flagged). Part B ASR≫0 with full probe numbers.
- NC mask L1 for non-backdoor classes remains large (~160–260) after 200 steps — not fully sparsified; relative MAD still cleanly separates class 0.
- Stealth NC skipped ablation (time); anomaly + rev-ASR recorded only.
- Scattered trigger still uses intensity 1.0 pixels — stealthier than a solid block, not adversarial-imperceptible.
