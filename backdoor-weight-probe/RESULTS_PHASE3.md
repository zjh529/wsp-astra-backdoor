# Backdoor Weight Probe — Phase 3 Results

Extends Phase 1–2: **Part A** removes the clean-*model* twin from detection (reference-free scores on MNIST `poisoned.pt` / `clean.pt`). **Part B** scales the BadNet + probe pipeline to CIFAR-10.

- **Python:** `/workspace/wsp/pi-weight-space-programming/.venv/bin/python`
- **Device:** CPU only; no Pi/GPT
- **Workdir:** `/workspace/wsp/backdoor-weight-probe/`

---

## Part A — Reference-free detection (NO clean twin)

Signals used **without** comparing to a twin checkpoint:

| Signal | Needs clean *images*? | Needs clean *model*? |
|--------|----------------------|----------------------|
| Neural Cleanse MAD anomaly index (primary) | **Yes** (optimization fodder) | No |
| Class-asymmetry of fc2 outgoing weights | No | No |
| NC-reversed trigger → act×w → joint ablation; Δrev-ASR | Yes (holdout + test) | No |
| Composite ref-free suspicion | (via above) | No |

**Still needs unlabeled clean data:** NC and ASR/ablation evals use clean images. That is **not** a clean-model twin (no weight deltas vs `clean.pt` in the score).

### Class-asymmetry (weights only)

| Model | class0_frac | mean class0 dominance | asymmetry_score |
|-------|-------------|----------------------|-----------------|
| Poisoned | 0.125 | 0.102 | **0.743** |
| Clean | 0.094 | 0.099 | **0.716** |

**Weak alone** — almost no separation. Pure outgoing-weight dominance does not surface this BadNet (consistent with Phase 1 static-scan failure).

### Neural Cleanse (reuse Phase-2 for poisoned; fresh run on clean)

| Model | Flagged class | AI (mask) | AI (trigger) | NC AI≥2? | Rev-ASR |
|-------|---------------|----------:|-------------:|:--------:|--------:|
| **Poisoned** | **0** | **8.978** | **17.224** | **Yes** | **0.994** |
| Clean | 2 | 1.559 | 2.763 | Yes (trigger AI) | **0.000** |

Honest note: **NC AI≥2 alone false-positives the clean model** (class 2 trigger-L1 AI≈2.76) because non-backdoor classes never fully sparsify in 200 steps; relative MAD can still exceed 2. Rev-ASR on clean is ~0, so the reverse trigger is **not** an implant.

### NC-rev → act×w → ablation (eval on reversed ASR; true-patch also recorded)

**Poisoned** (NC-rank top-k):

| k | Clean acc | Rev-ASR | Δrev-ASR | True-patch ASR |
|--:|----------:|--------:|---------:|---------------:|
| 1 | 0.984 | 0.981 | +0.013 | 0.999 |
| 5 | 0.916 | 0.570 | +0.424 | 0.740 |
| **10** | **0.851** | **0.067** | **+0.928** | **0.181** |
| 20 | 0.702 | 0.001 | +0.993 | 0.001 |

**Clean:** best practical Δrev-ASR ≈ **0** (no backdoor to kill).

### Ref-free model suspicion

\[
S = 2\cdot\min(\mathrm{AI}/2,\,5) + 2\cdot\mathrm{revASR} + 3\cdot\Delta\mathrm{revASR} + 0.5\cdot\min(\mathrm{asym}/3,\,1.5)
\]

| Model | Suspicion \(S\) |
|-------|----------------:|
| Poisoned | **14.895** |
| Clean | **2.883** |
| Separation | **+12.01** |
| Ratio | **5.17×** |

### Threshold rule

| Rule | Poisoned flagged? | Clean flagged? | Correct? | Honesty |
|------|:-----------------:|:--------------:|:--------:|---------|
| NC AI≥2 alone (literature) | Yes | **Yes (FP)** | No | Not tuned on these two; still FP here |
| Fixed composite: AI≥2 **AND** revASR≥0.5 **AND** (ΔrevASR≥0.3 OR AI≥5) | **Yes** | **No** | **Yes** | Constants from NC literature + ASR intuition; not grid-searched on these two, but informed by Phase 1–2 experience |
| Post-hoc midpoint on \(S\) (8.89) | Yes | No | Yes | **Post-hoc** — report separation; do not claim generalization from midpoint alone |

**Part A success:** numeric separation **without clean-model weights** (14.9 vs 2.9). Primary NC AI alone is insufficient for FP control; composite (AI + rev-ASR + ablation kill) separates cleanly.

---

## Part B — CIFAR-10 scale-up

### Setup

- **Arch:** `TinyCIFARCNN` — Conv32-BN-MP → Conv64-BN-MP → Conv64-BN-MP → FC(1024→128) → FC(128→10); **189,130 params**
- **Trigger:** white **3×3** top-left → class **0** (airplane); poison **8%** (4000/50000); holdout 2000
- **Train:** 3 epochs, Adam 1e-3, full CIFAR-10, CPU, seed 42
- **Checkpoints:** `models/cifar_clean.pt`, `models/cifar_poisoned.pt`

### Training / base metrics

| Model | Clean test acc | ASR | Time |
|-------|---------------:|----:|-----:|
| Clean | 0.6995 | 0.0250 | ~31 s |
| **Poisoned** | **0.6664** | **0.9051** | ~30 s |

High ASR with usable clean accuracy after 3 CPU epochs — BadNet implant succeeded.

### Act×w (TRUE trigger) + joint ablation — poisoned

| k | Clean acc | ASR | ΔASR | Δclean |
|--:|----------:|----:|-----:|-------:|
| 1 | 0.665 | 0.896 | +0.009 | +0.002 |
| 10 | 0.616 | 0.745 | +0.160 | +0.051 |
| 20 | 0.621 | 0.459 | +0.446 | +0.045 |
| **40** | **0.512** | **0.091** | **+0.815** | **+0.155** |
| 60 | 0.340 | 0.000 | +0.905 | +0.327 |

**Ablation kill: yes** (top-40 → ASR 0.91→0.09). More distributed than MNIST (needed ~40 of 128 fc1 units vs ~10 of 64); clean drop larger.

**Clean twin contrast:** act-rank top-20 ΔASR only +0.02 (no backdoor).

### Neural Cleanse on CIFAR (best-effort)

**Short (80 steps, classes {0,1,3,5,7}):** flagged class **3** (AI=1.0); class-0 rev-ASR still **0.996**. MAD among unfinished L1s did not prefer class 0.

**Long (200 steps, classes {0,1,2,3,5,7,9}):**

| Class | L1(m) | AI |
|------:|------:|---:|
| 0 | 125.79 | 1.000 |
| 1 | 208.58 | −13.9 |
| **2** | **106.83** | **4.422** |
| 3 | 126.69 | 0.839 |
| 5 | 131.34 | 0.000 |
| 7 | 136.81 | −0.99 |
| 9 | 152.96 | −3.90 |

- **Flagged: class 2** (not the true target 0). NC MAD **fails** to flag the implant class under this CPU budget.
- Class-0 **reversed ASR = 0.992** — a class-0 surrogate trigger still exists; it is just not uniquely sparsest vs other classes after 200 steps on 32×32 RGB.

### CIFAR ref-free suspicion (poisoned vs clean)

| Model | Suspicion (short NC / no NC) | With long-NC AI |
|-------|-----------------------------:|----------------:|
| Poisoned | 4.39 (short; wrong flagged class) | **8.90** (uses class0 rev-ASR + top-40 ΔASR) |
| Clean | **0.16** (NC skipped) | — |

Separation remains large when using true-trigger ablation + class0 rev-ASR; **NC anomaly index alone does not survive CIFAR** at this budget.

### Does the pipeline survive CIFAR?

| Stage | Survive? | Notes |
|-------|:--------:|-------|
| BadNet implant (ASR) | **Yes** | ASR≈0.91, clean≈0.67 |
| Act×w ranking + joint ablation | **Yes** | Top-40 kills ASR; more neurons than MNIST |
| NC MAD flags true target | **No** | Flags class 2; need more steps/λ or better mask prior |
| Ref-free suspicion vs clean | **Partial** | Works if using rev-ASR + ΔASR; not if using NC AI alone |
| Class-asymmetry alone | **No** | Still weak |

---

## Cross-cutting answers

| Question | Answer |
|----------|--------|
| Ref-free scores MNIST poisoned vs clean? | **14.90 vs 2.88** (5.2×); composite rule flags poisoned only |
| CIFAR acc / ASR? | Clean 0.70 / 0.03; Poisoned **0.67 / 0.91** |
| NC flag (CIFAR)? | **Not correctly** — flags class 2; class0 rev-ASR high |
| Ablation kill (CIFAR)? | **Yes** — top-40 ASR 0.91→0.09 |
| Pipeline survive CIFAR? | **Act-rank yes; NC MAD no** (at 200-step / 7-class budget) |
| Failures / time limits | CIFAR download slow (~70 KB/s; used parallel range fetch). NC on CIFAR does not sparsify enough for unique class-0 L1. Clean-model NC skipped on CIFAR for CPU. Class-asymmetry useless alone. |

---

## Artifacts

| Path | Content |
|------|---------|
| `scripts/ref_free.py` | Part A reference-free MNIST detector |
| `scripts/train_cifar.py` | TinyCIFARCNN + BadNet train |
| `scripts/probe_cifar.py` | CIFAR act×w / NC / ablation / suspicion |
| `scripts/run_phase3.py` | Orchestrator |
| `models/cifar_clean.pt` | CIFAR clean checkpoint |
| `models/cifar_poisoned.pt` | CIFAR poisoned checkpoint |
| `results/phase3_ref_free.json` | Part A full numbers |
| `results/phase3_cifar_probe_*.json` | Part B probe |
| `results/phase3_summary.json` | Combined summary |
| `RESULTS_PHASE3.md` | This report |
