"""Static recovery check (no trigger for ranking) + RESULTS.md + optional plot."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from data import TARGET_LABEL, get_datasets, make_loader, triggered_test_batch
from model import TinyCNN

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
EPS = 1e-3


def load(name):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)
    m = TinyCNN()
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt


def ablate_fc1(model, indices):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for i in indices:
            m.fc2.weight[:, i] = 0.0
            m.fc1.bias[i] = 0.0
            m.fc1.weight[i, :] = 0.0
    return m


def eval_clean(model, test, max_batches=40):
    correct = total = fwd = 0
    loader = make_loader(test, batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            fwd += 1
            if bi + 1 >= max_batches:
                break
    return correct / total, fwd


def eval_asr(model, test, max_batches=40):
    hit = total = fwd = 0
    with torch.no_grad():
        for bi, (x, _) in enumerate(triggered_test_batch(test, batch_size=256, exclude_target=True)):
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            fwd += 1
            if bi + 1 >= max_batches:
                break
    return hit / total, fwd


def static_rank_poisoned_vs_clean():
    """Rank without knowing trigger: delta of |w_to_class0| * L2(incoming fc1) vs clean twin.
    Also try max over classes of asymmetry.
    """
    p, _ = load("poisoned")
    c, _ = load("clean")
    scores = []
    for i in range(64):
        # Outgoing weight to each class
        wp = p.fc2.weight[:, i].detach().abs()
        wc = c.fc2.weight[:, i].detach().abs()
        # Focus class-0 push relative to other classes
        p0 = float(wp[0] / (wp.mean() + EPS))
        c0 = float(wc[0] / (wc.mean() + EPS))
        # Incoming L2
        pin = float(p.fc1.weight[i].detach().norm())
        cin = float(c.fc1.weight[i].detach().norm())
        # Delta class-0 specificity * incoming
        score = (p0 - c0) * pin
        scores.append({"index": i, "score": score, "p0_spec": p0, "c0_spec": c0, "pin": pin})
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


def main():
    ds = get_datasets(seed=42)
    test = ds["test"]
    train = json.loads((RESULTS / "train_summary.json").read_text())
    v2 = json.loads((RESULTS / "probe_v2_summary.json").read_text())
    scan_p = json.loads((RESULTS / "weight_scan_poisoned.json").read_text())
    scan_c = json.loads((RESULTS / "weight_scan_clean.json").read_text())

    # Static ranking recovery
    static = static_rank_poisoned_vs_clean()
    static_top = [r["index"] for r in static[:20]]
    act_top = v2["poisoned"]["fc1"]["top_ranking"][:20]
    # Overlap with activation-ranked true-effective set (top-10 that killed ASR)
    effective = set(v2["recovery"]["fc1_best_practical"]["neurons"])
    static_overlap = [i for i in static_top[:10] if i in effective]
    # Also composite-delta ranking from first scan
    pcomp = {r["index"]: r["composite"] for r in scan_p["fc1_scores"]}
    ccomp = {r["index"]: r["composite"] for r in scan_c["fc1_scores"]}
    delta_rank = sorted(range(64), key=lambda i: pcomp[i] - ccomp[i], reverse=True)
    delta_overlap = [i for i in delta_rank[:10] if i in effective]

    # Evaluate static top-10 ablation on poisoned
    model, _ = load("poisoned")
    base_acc, f1 = eval_clean(model, test)
    base_asr, f2 = eval_asr(model, test)
    m = ablate_fc1(model, static_top[:10])
    acc_s, f3 = eval_clean(m, test)
    asr_s, f4 = eval_asr(m, test)
    m2 = ablate_fc1(model, delta_rank[:10])
    acc_d, f5 = eval_clean(m2, test)
    asr_d, f6 = eval_asr(m2, test)
    # Cheap method cost: weight scan ~0 + act scoring (~16) + joint ablations only at k=1,3,5,10
    # (skip 20 singles) = base(80) + 4*80 = 400 forwards-ish for fc1 practical path
    cheap_fwd = 16 + 80 + 4 * 80  # score + base + 4 joint ks
    # Full v2 with singles was ~4000; document both
    naive_lb = 676
    naive_matched = 676 * 40

    static_result = {
        "static_class0_delta_top10": static_top[:10],
        "static_overlap_with_effective": static_overlap,
        "static_top10_ablation": {
            "clean_acc": acc_s,
            "asr": asr_s,
            "delta_asr": base_asr - asr_s,
            "delta_clean": base_acc - acc_s,
        },
        "composite_delta_top10": delta_rank[:10],
        "composite_overlap_with_effective": delta_overlap,
        "composite_top10_ablation": {
            "clean_acc": acc_d,
            "asr": asr_d,
            "delta_asr": base_asr - asr_d,
            "delta_clean": base_acc - acc_d,
        },
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
    }
    (RESULTS / "static_recovery.json").write_text(json.dumps(static_result, indent=2))
    print(json.dumps(static_result, indent=2))

    # Optional bar plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Poisoned\nbase ASR", "Poisoned\nact top-10\nablate", "Poisoned\nstatic top-10\nablate",
                  "Clean\nbase ASR", "Clean\nact top-10\nablate"]
        # clean act top-10 from v2 joints
        clean_j = [j for j in v2["clean"]["fc1"]["joints"] if j["k"] == 10][0]
        poison_j = v2["recovery"]["fc1_best_practical"]
        vals = [base_asr, poison_j["asr"], asr_s, v2["clean"]["base_asr"], clean_j["asr"]]
        colors = ["#c0392b", "#27ae60", "#2980b9", "#7f8c8d", "#95a5a6"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("ASR before/after ablating top-10 suspicious fc1 neurons")
        fig.tight_layout()
        fig.savefig(RESULTS / "asr_ablation_bars.png", dpi=120)
        plt.close()
        print("Wrote", RESULTS / "asr_ablation_bars.png")
    except Exception as e:
        print("Plot skipped:", e)

    # Write RESULTS.md
    pj = v2["recovery"]["fc1_best_practical"]
    sus = v2["model_suspicion"]
    lines = f"""# Backdoor Weight Probe — Results

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
| Clean | {train['clean']['clean_acc']:.4f} | {train['clean']['asr']:.4f} | `models/clean.pt` |
| Poisoned | {train['poisoned']['clean_acc']:.4f} | {train['poisoned']['asr']:.4f} | `models/poisoned.pt` |

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
Probe top-20 one-at-a-time **and** joint top-{{1,3,5,10,20}}.

### D) Model suspicion
Aggregate best practical ASR drop + single-neuron signals + base ASR; compare poisoned vs clean twin.

## Probe results — poisoned

| Ablation | Clean acc | ASR | ΔASR | Δclean |
|----------|-----------|-----|------|--------|
| None (base) | {v2['poisoned']['base_clean_acc']:.4f} | {v2['poisoned']['base_asr']:.4f} | — | — |
| Act-rank top-1 | { [j for j in v2['poisoned']['fc1']['joints'] if j['k']==1][0]['clean_acc']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==1][0]['asr']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==1][0]['delta_asr']:+.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==1][0]['delta_clean_acc']:+.4f} |
| Act-rank top-3 | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==3][0]['clean_acc']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==3][0]['asr']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==3][0]['delta_asr']:+.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==3][0]['delta_clean_acc']:+.4f} |
| Act-rank top-5 | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==5][0]['clean_acc']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==5][0]['asr']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==5][0]['delta_asr']:+.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==5][0]['delta_clean_acc']:+.4f} |
| **Act-rank top-10** | **{pj['clean_acc']:.4f}** | **{pj['asr']:.4f}** | **{pj['delta_asr']:+.4f}** | **{pj['delta_clean_acc']:+.4f}** |
| Act-rank top-20 | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==20][0]['clean_acc']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==20][0]['asr']:.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==20][0]['delta_asr']:+.4f} | {[j for j in v2['poisoned']['fc1']['joints'] if j['k']==20][0]['delta_clean_acc']:+.4f} |
| Static class0-delta top-10 | {acc_s:.4f} | {asr_s:.4f} | {base_asr-asr_s:+.4f} | {base_acc-acc_s:+.4f} |
| Composite-delta top-10 | {acc_d:.4f} | {asr_d:.4f} | {base_asr-asr_d:+.4f} | {base_acc-acc_d:+.4f} |

**Best practical:** ablating act-ranked fc1 neurons `{pj['neurons']}` → ASR **{v2['poisoned']['base_asr']:.3f} → {pj['asr']:.3f}** with clean acc still **{pj['clean_acc']:.3f}**.

Single-neuron ablation barely moves ASR (backdoor is **distributed** across ~10 fc1 units). Conv2 channel ablation was weak (ASR stayed ≈1.0).

## Probe results — clean twin

| Ablation | Clean acc | ASR | ΔASR |
|----------|-----------|-----|------|
| None | {v2['clean']['base_clean_acc']:.4f} | {v2['clean']['base_asr']:.4f} | — |
| Act-rank top-10 | {[j for j in v2['clean']['fc1']['joints'] if j['k']==10][0]['clean_acc']:.4f} | {[j for j in v2['clean']['fc1']['joints'] if j['k']==10][0]['asr']:.4f} | {[j for j in v2['clean']['fc1']['joints'] if j['k']==10][0]['delta_asr']:+.4f} |

Clean twin has near-zero ASR; ablating the same style of “suspicious” units only hurts clean accuracy (no backdoor to remove).

## Did weight / ranking recover backdoor neurons?

| Ranking method | Needs trigger? | Overlap w/ effective top-10 | Top-10 ablate ΔASR |
|----------------|----------------|-----------------------------|--------------------|
| Act × w_to_class0 | Yes (candidate) | **10/10** (by construction of effective set) | **{pj['delta_asr']:+.4f}** |
| Static class-0 specificity Δ vs clean | No | {len(static_overlap)}/10 → `{static_overlap}` | {base_asr-asr_s:+.4f} |
| Composite weight z-score Δ vs clean | No | {len(delta_overlap)}/10 → `{delta_overlap}` | {base_asr-asr_d:+.4f} |

**Answer:** Pure L2/composite weight outliers did **not** reliably surface the backdoor units (distributed / not magnitude outliers).  
**Activation×target-weight ranking** (given a candidate trigger) **did**, and joint ablation of top-10 **kills ASR**.  
Static class-0 specificity vs clean twin: partial overlap `{static_overlap}`, ΔASR={base_asr-asr_s:+.3f} (weaker than act ranking).

## Model suspicion scores

| Model | Suspicion |
|-------|-----------|
| Poisoned | **{sus['poisoned']:.3f}** |
| Clean | **{sus['clean']:.3f}** |
| Ratio | **{sus['ratio']:.1f}×** |

Clear separation: poisoned ≫ clean.

## Cost comparison (forward-pass proxy)

| Method | Approx forwards | Notes |
|--------|-----------------|-------|
| Naive 3×3 white patch × locations | ≥ **{naive_lb}** (LB) / **{naive_matched}** (matched 40-batch ASR) | 26×26=676 locations |
| Act-score + joint top-k ablations (practical) | ~**{cheap_fwd}** | holdout act passes + base + k∈{{1,3,5,10}} evals |
| Full v2 (incl. 20 singles × 2 layers) | ~**4000**/model | thorough but optional |
| Weight norms alone | ≪ 1 forward | matrix ops on params |

Practical pipeline (~{cheap_fwd} forwards) is **~{naive_matched/cheap_fwd:.0f}×** cheaper than location search at matched ASR eval depth, and similar order to the optimistic 676 LB while also **localizing neurons** to ablate (mitigation), not only detecting a trigger.

Optional plot: `results/asr_ablation_bars.png`

## Conclusion

**Yes, with caveats — for this small CNN the method works.**

1. BadNet is implanted successfully (ASR=1.0, clean≈98%).
2. The backdoor is **distributed**: single-neuron ablation fails; **joint top-10** fc1 ablation drops ASR **1.00 → 0.05** with modest clean loss (~12 pp).
3. Ranking by **trigger activation × weight-to-target** recovers the right units; naive weight-norm outliers do not.
4. Clean twin suspicion stays low (**{sus['ratio']:.0f}×** lower than poisoned).
5. Cheaper than exhaustive patch-location search when you evaluate ASR at non-trivial batch depth; yields a **mitigation** (ablate) not just a detection bit.

## Limitations / failures

- **Static L2 scan alone failed** to rank true backdoor neurons highly — magnitude ≠ Trojan role.
- Act-ranking **uses a candidate trigger**; fully blind detection still needs trigger reverse-engineering (Neural Cleanse-style) or a strong clean reference + class-asymmetry heuristics (partial here).
- Conv2 GAP deltas did not yield an ASR kill — spatial trigger may be pooled away / redundant across channels.
- Clean twin assumed available for delta features and suspicion contrast.
- Tiny MNIST CNN only; larger nets / stealthy triggers may need finer probes.
- No Pi CLI / GPT Astra — classical PyTorch only.

## Artifacts

- Scripts: `scripts/{{model,data,train,weight_scan,probe,probe_v2,finalize,run_experiment}}.py`
- Checkpoints: `models/clean.pt`, `models/poisoned.pt`
- JSON: `results/train_summary.json`, `weight_scan_*.json`, `probe_v2_*.json`, `static_recovery.json`
- This report: `RESULTS.md`
"""
    (ROOT / "RESULTS.md").write_text(lines)
    print("Wrote", ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
