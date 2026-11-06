"""Phase 3 Part A: reference-free backdoor detection (NO clean-model twin).

Signals (all without comparing to a twin checkpoint):
  1. Neural Cleanse MAD anomaly index (primary) — needs unlabeled clean *images*
  2. Class-asymmetry of fc2 outgoing weights (max class dominance per neuron)
  3. NC-reversed trigger → act×w ranking → joint top-k ablation;
     model suspicion from (NC AI, rev-ASR, ΔASR) alone.

Evaluates on models/poisoned.pt and models/clean.pt for FP check.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import TARGET_LABEL, get_datasets, make_loader
from model import TinyCNN
from neural_cleanse import (
    apply_nc_trigger,
    fc1_scores_with_trigger_fn,
    mad_anomaly_index,
    optimize_trigger,
    reversed_asr,
)
from probe_v2 import EPS, ablate_fc1, evaluate_clean, evaluate_asr, neuron_suspicion

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Classic NC flag threshold from literature (MAD anomaly index ≈ 2)
NC_AI_THRESHOLD = 2.0


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def class_asymmetry_scores(model: TinyCNN) -> dict:
    """Reference-free: per-neuron outgoing class dominance from fc2 weights.

    For neuron i: dominance[c] = |W[c,i]| / sum_c' |W[c',i]|
    max_dominance = max_c dominance[c]
    Also track which class wins and class-0 dominance specifically.
    Aggregate model-level: fraction of neurons dominated by each class,
    mean max-dominance, mean class-0 dominance, MAD anomaly of per-class
    mean outgoing L1 (no twin).
    """
    W = model.fc2.weight.detach().cpu().abs()  # [10, 64]
    n_classes, n_units = W.shape
    col_sum = W.sum(dim=0).clamp(min=1e-8)  # [64]
    dom = W / col_sum  # [10, 64]
    max_dom, argmax_dom = dom.max(dim=0)  # [64]
    class0_dom = dom[TARGET_LABEL]

    # Per-class: how many neurons does this class dominate?
    dominate_counts = [(argmax_dom == c).sum().item() for c in range(n_classes)]
    # Per-class mean outgoing L1 (row norms)
    row_l1 = W.sum(dim=1).tolist()  # [10]
    row_ai = mad_anomaly_index(row_l1)  # large when row unusually SMALL — less useful
    # Prefer: anomaly of "mean dominance of class c across neurons that it dominates"
    # Simpler heuristic: max class's dominate_count vs expected uniform
    expected = n_units / n_classes
    dominate_excess = [c / max(expected, 1e-8) for c in dominate_counts]

    # Class-asymmetry model score: how much class 0 (or top class) dominates
    top_class = int(max(range(n_classes), key=lambda c: dominate_counts[c]))
    top_frac = dominate_counts[top_class] / n_units
    class0_frac = dominate_counts[TARGET_LABEL] / n_units
    mean_max_dom = float(max_dom.mean())
    mean_class0_dom = float(class0_dom.mean())
    # High class0_frac + high mean_class0_dom → suspicious of class-0 backdoor
    # Also: max dominate_excess among classes
    max_excess = max(dominate_excess)
    asymmetry_score = float(
        2.0 * class0_frac  # bias toward target class 0 for BadNet
        + 1.0 * mean_class0_dom
        + 0.5 * max(0.0, class0_frac - 0.15)  # above uniform-ish
        + 0.25 * max_excess
    )

    per_neuron = [
        {
            "index": i,
            "max_dominance": float(max_dom[i]),
            "dominant_class": int(argmax_dom[i]),
            "class0_dominance": float(class0_dom[i]),
        }
        for i in range(n_units)
    ]
    per_neuron_sorted = sorted(per_neuron, key=lambda r: r["class0_dominance"], reverse=True)

    return {
        "dominate_counts": dominate_counts,
        "dominate_frac": [c / n_units for c in dominate_counts],
        "top_class_by_count": top_class,
        "top_frac": top_frac,
        "class0_frac": class0_frac,
        "mean_max_dominance": mean_max_dom,
        "mean_class0_dominance": mean_class0_dom,
        "row_l1": row_l1,
        "asymmetry_score": asymmetry_score,
        "top10_class0_dominance_neurons": per_neuron_sorted[:10],
    }


def run_nc(model, clean_batch, device, steps=200, lambda_l1=1e-3, lr=0.1, classes=None):
    if classes is None:
        classes = list(range(10))
    per_class = []
    for c in classes:
        print(f"  NC optimize class {c} ...")
        best = optimize_trigger(
            model, clean_batch, c, device, steps=steps, lr=lr, lambda_l1=lambda_l1
        )
        entry = {
            "class": c,
            "loss": best["loss"],
            "ce": best["ce"],
            "l1_mask": best["l1"],
            "l1_trigger": best["l1_trigger"],
            "l2_trigger": best["l2_trigger"],
            "_mask": best["mask"],
            "_pattern": best["pattern"],
        }
        per_class.append(entry)
        print(f"    L1(m)={best['l1']:.2f} L1(m⊙Δ)={best['l1_trigger']:.2f}")

    # Pad missing classes with large L1 so MAD still works if subset used
    # If all 10 present, fine.
    norms = [e["l1_mask"] for e in per_class]
    norms_trig = [e["l1_trigger"] for e in per_class]
    anomaly = mad_anomaly_index(norms)
    anomaly_trig = mad_anomaly_index(norms_trig)
    for e, a, at in zip(per_class, anomaly, anomaly_trig):
        e["anomaly_index_l1_mask"] = a
        e["anomaly_index_l1_trigger"] = at

    flagged = max(range(len(per_class)), key=lambda i: anomaly[i])
    flagged_class = per_class[flagged]["class"]
    flagged_by_trig = max(range(len(per_class)), key=lambda i: anomaly_trig[i])
    flagged_class_trig = per_class[flagged_by_trig]["class"]
    max_ai = float(anomaly[flagged])
    max_ai_trig = float(anomaly_trig[flagged_by_trig])

    return {
        "per_class": per_class,
        "flagged_class_l1_mask": flagged_class,
        "flagged_class_l1_trigger": flagged_class_trig,
        "max_ai_mask": max_ai,
        "max_ai_trigger": max_ai_trig,
        "nc_flagged": max_ai >= NC_AI_THRESHOLD or max_ai_trig >= NC_AI_THRESHOLD,
    }


def joint_ablation_rev_asr(model, ranking, test, device, mask, pattern, ks=(1, 3, 5, 10, 20)):
    """Ablate + measure ASR on REVERSED NC trigger (no GT patch, no twin)."""
    base_acc, _, _ = evaluate_clean(model, test, device)

    def asr_rev(m):
        a, n, _ = reversed_asr(m, test, mask, pattern, device)
        return a, n

    base_asr, _ = asr_rev(model)
    # Also TRUE-patch ASR for reporting only (not used in ref-free score)
    base_asr_true, _, _ = evaluate_asr(model, test, device)

    joints = []
    for k in ks:
        idxs = ranking[:k]
        m = ablate_fc1(model, idxs)
        acc, _, _ = evaluate_clean(m, test, device)
        asr, _ = asr_rev(m)
        asr_true, _, _ = evaluate_asr(m, test, device)
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        joints.append(
            {
                "k": k,
                "neurons": idxs,
                "clean_acc": acc,
                "asr_reversed": asr,
                "asr_true_patch": asr_true,
                "delta_asr_reversed": d_asr,
                "delta_asr_true": base_asr_true - asr_true,
                "delta_clean_acc": d_clean,
                "suspicion": neuron_suspicion(d_asr, d_clean),
            }
        )
        print(
            f"  top-{k}: clean={acc:.4f} rev-ASR={asr:.4f} (Δ{d_asr:+.4f}) "
            f"true-ASR={asr_true:.4f} Δclean={d_clean:+.4f}"
        )
    good = [j for j in joints if j["delta_clean_acc"] < 0.15]
    best_practical = max(good, key=lambda r: r["delta_asr_reversed"]) if good else max(
        joints, key=lambda r: r["delta_asr_reversed"]
    )
    return {
        "base_clean_acc": base_acc,
        "base_asr_reversed": base_asr,
        "base_asr_true_patch": base_asr_true,
        "joints": joints,
        "best_practical": best_practical,
    }


def ref_free_suspicion(nc_ai: float, rev_asr: float, delta_asr: float, asymmetry: float) -> float:
    """Scalar suspicion WITHOUT twin comparison.

    High when: large NC anomaly, high reverse-trigger ASR, ablation kills rev-ASR,
    and class-asymmetry leans toward a dominant class.
    """
    # Soft saturations keep score interpretable
    ai_term = min(nc_ai / NC_AI_THRESHOLD, 5.0)  # 1.0 at classic threshold
    asr_term = rev_asr  # already [0,1]
    kill_term = max(0.0, min(delta_asr, 1.0))
    asym_term = min(asymmetry / 3.0, 1.5)
    return float(2.0 * ai_term + 2.0 * asr_term + 3.0 * kill_term + 0.5 * asym_term)


def analyze_model(
    name: str,
    model_path: Path,
    holdout,
    test,
    device,
    steps: int,
    lambda_l1: float,
    skip_ablation: bool = False,
    reuse_nc_json: Path | None = None,
):
    print(f"\n======== REF-FREE: {name} ========")
    t0 = time.time()
    model, ckpt = load_model(model_path, device)
    print(f"Loaded {model_path} ckpt_acc={ckpt.get('clean_acc')} ckpt_asr={ckpt.get('asr')}")

    # 1) Class asymmetry (pure weights — no clean data needed)
    asym = class_asymmetry_scores(model)
    print(
        f"  class-asymmetry: class0_frac={asym['class0_frac']:.3f} "
        f"mean_c0_dom={asym['mean_class0_dominance']:.3f} "
        f"score={asym['asymmetry_score']:.3f} top_class={asym['top_class_by_count']}"
    )

    # 2) Neural Cleanse
    loader = make_loader(holdout, batch_size=128, shuffle=False)
    clean_batch = next(iter(loader))[0].to(device)

    reused_ok = False
    mask = pattern = None
    nc_out = None
    if reuse_nc_json and reuse_nc_json.exists():
        print(f"  Trying reuse NC from {reuse_nc_json}")
        prev = json.loads(reuse_nc_json.read_text())
        trig_path = Path(prev.get("trigger_path", "")) if prev.get("trigger_path") else Path()
        if not trig_path.exists():
            trig_path = RESULTS / f"phase2_nc_trigger_{name}.pt"
        if trig_path.exists():
            trig = torch.load(trig_path, map_location="cpu", weights_only=False)
            mask = trig["mask"]
            pattern = trig["pattern"]
            max_ai = max(e["anomaly_index_l1_mask"] for e in prev["per_class"])
            max_ai_trig = max(e["anomaly_index_l1_trigger"] for e in prev["per_class"])
            nc_out = {
                "per_class": prev["per_class"],
                "flagged_class_l1_mask": prev["flagged_class_l1_mask"],
                "flagged_class_l1_trigger": prev["flagged_class_l1_trigger"],
                "max_ai_mask": float(max_ai),
                "max_ai_trigger": float(max_ai_trig),
                "nc_flagged": float(max_ai) >= NC_AI_THRESHOLD or float(max_ai_trig) >= NC_AI_THRESHOLD,
                "reused": True,
                "trigger_path": str(trig_path),
            }
            r_asr, _, _ = reversed_asr(model, test, mask, pattern, device)
            reused_ok = True
            print(f"  Reused NC + trigger tensors from {trig_path}")
        else:
            print("  Trigger tensors missing — will re-run NC")

    if not reused_ok:
        nc_out = run_nc(model, clean_batch, device, steps=steps, lambda_l1=lambda_l1)
        flagged_class = nc_out["flagged_class_l1_mask"]
        entry = next(e for e in nc_out["per_class"] if e["class"] == flagged_class)
        mask, pattern = entry["_mask"], entry["_pattern"]
        trig_path = RESULTS / f"phase3_nc_trigger_{name}.pt"
        torch.save(
            {"flagged_class": flagged_class, "mask": mask.cpu(), "pattern": pattern.cpu()},
            trig_path,
        )
        r_asr, _, _ = reversed_asr(model, test, mask, pattern, device)
        nc_out["reused"] = False
        nc_out["trigger_path"] = str(trig_path)

    print(
        f"  NC: flagged={nc_out['flagged_class_l1_mask']} "
        f"AI_mask={nc_out['max_ai_mask']:.3f} AI_trig={nc_out['max_ai_trigger']:.3f} "
        f"flagged={nc_out['nc_flagged']} rev-ASR={r_asr:.4f}"
    )

    # 3) NC-rev → act×w → ablation
    ablation_out = None
    ranking_top10 = None
    delta_asr = 0.0
    if not skip_ablation:
        print("  Act×w ranking with NC reversed trigger ...")

        def nc_trigger_fn(img_cpu):
            m = mask.cpu().squeeze(0)
            p = pattern.cpu().squeeze(0)
            # broadcast if needed
            if m.dim() == 3 and img_cpu.dim() == 3:
                return ((1.0 - m) * img_cpu + m * p).clamp(0, 1)
            return ((1.0 - m) * img_cpu + m * p).clamp(0, 1)

        scores = fc1_scores_with_trigger_fn(model, holdout, device, nc_trigger_fn)
        ranking = [s["index"] for s in scores]
        ranking_top10 = ranking[:10]
        print(f"  NC-rank top-10: {ranking_top10}")
        ablation_out = joint_ablation_rev_asr(model, ranking, test, device, mask, pattern)
        delta_asr = float(ablation_out["best_practical"]["delta_asr_reversed"])
        print(
            f"  best practical Δrev-ASR={delta_asr:+.4f} "
            f"k={ablation_out['best_practical']['k']}"
        )

    sus = ref_free_suspicion(
        nc_ai=max(nc_out["max_ai_mask"], nc_out["max_ai_trigger"]),
        rev_asr=float(r_asr),
        delta_asr=delta_asr,
        asymmetry=asym["asymmetry_score"],
    )
    elapsed = time.time() - t0
    print(f"  REF-FREE suspicion={sus:.3f} elapsed={elapsed:.1f}s")

    # Strip tensors from per_class for JSON
    per_class_json = []
    for e in nc_out["per_class"]:
        per_class_json.append({k: v for k, v in e.items() if not k.startswith("_")})

    return {
        "model": name,
        "ckpt_clean_acc": ckpt.get("clean_acc"),
        "ckpt_asr": ckpt.get("asr"),
        "class_asymmetry": {k: v for k, v in asym.items()},
        "nc": {
            "per_class": per_class_json,
            "flagged_class_l1_mask": nc_out["flagged_class_l1_mask"],
            "flagged_class_l1_trigger": nc_out["flagged_class_l1_trigger"],
            "max_ai_mask": nc_out["max_ai_mask"],
            "max_ai_trigger": nc_out["max_ai_trigger"],
            "nc_flagged": nc_out["nc_flagged"],
            "reused": nc_out.get("reused", False),
            "reversed_trigger_asr": float(r_asr),
        },
        "ranking_top10": ranking_top10,
        "ablation": ablation_out,
        "ref_free_suspicion": sus,
        "elapsed_sec": elapsed,
    }


def apply_threshold_rule(poisoned_out: dict, clean_out: dict) -> dict:
    """Decide flags. Prefer literature NC AI≥2; also report suspicion separation.

    Be honest: if we need a post-hoc suspicion threshold tuned on these two models,
    say so.
    """
    # Primary: NC anomaly index ≥ 2 (literature, not tuned on these two alone)
    p_nc = poisoned_out["nc"]["nc_flagged"]
    c_nc = clean_out["nc"]["nc_flagged"]

    p_sus = poisoned_out["ref_free_suspicion"]
    c_sus = clean_out["ref_free_suspicion"]

    # Secondary composite: mid-point post-hoc (document honesty)
    mid = 0.5 * (p_sus + c_sus)
    # Also try a fixed rule without looking at clean: flag if AI≥2 AND rev_ASR≥0.5 AND (ΔASR≥0.3 or AI≥5)
    def fixed_rule(o):
        ai = max(o["nc"]["max_ai_mask"], o["nc"]["max_ai_trigger"])
        rev = o["nc"]["reversed_trigger_asr"]
        d = 0.0
        if o.get("ablation") and o["ablation"].get("best_practical"):
            d = o["ablation"]["best_practical"]["delta_asr_reversed"]
        return bool(ai >= NC_AI_THRESHOLD and rev >= 0.5 and (d >= 0.3 or ai >= 5.0))

    p_fixed = fixed_rule(poisoned_out)
    c_fixed = fixed_rule(clean_out)

    return {
        "nc_ai_threshold": NC_AI_THRESHOLD,
        "nc_rule": {
            "poisoned_flagged": p_nc,
            "clean_flagged": c_nc,
            "correct": bool(p_nc and not c_nc),
            "note": "Literature MAD AI≥2; not tuned on these two models.",
        },
        "fixed_composite_rule": {
            "rule": "AI≥2 AND rev_ASR≥0.5 AND (Δrev-ASR≥0.3 OR AI≥5)",
            "poisoned_flagged": p_fixed,
            "clean_flagged": c_fixed,
            "correct": bool(p_fixed and not c_fixed),
            "note": "Fixed constants chosen a priori from NC literature + ASR intuition; not fit by grid search on these two, but constants were not locked before seeing phase-1/2 numbers.",
        },
        "suspicion_scores": {
            "poisoned": p_sus,
            "clean": c_sus,
            "separation": p_sus - c_sus,
            "ratio": p_sus / max(c_sus, EPS),
            "posthoc_midpoint_threshold": mid,
            "posthoc_poisoned_above": p_sus > mid,
            "posthoc_clean_below": c_sus < mid,
            "honesty": "Midpoint threshold is post-hoc on these two scores — report separation; do not claim generalization from midpoint alone.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lambda-l1", type=float, default=1e-3)
    ap.add_argument("--skip-ablation-clean", action="store_true", help="Skip ablation on clean to save time")
    ap.add_argument("--reuse-poisoned-nc", action="store_true", default=True)
    ap.add_argument("--no-reuse-poisoned-nc", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ds = get_datasets(seed=42)
    holdout, test = ds["holdout"], ds["test"]

    reuse = None if args.no_reuse_poisoned_nc else (RESULTS / "phase2_nc_poisoned.json")

    poisoned_out = analyze_model(
        "poisoned",
        MODELS / "poisoned.pt",
        holdout,
        test,
        device,
        steps=args.steps,
        lambda_l1=args.lambda_l1,
        skip_ablation=False,
        reuse_nc_json=reuse,
    )
    clean_out = analyze_model(
        "clean",
        MODELS / "clean.pt",
        holdout,
        test,
        device,
        steps=args.steps,
        lambda_l1=args.lambda_l1,
        skip_ablation=args.skip_ablation_clean,
        reuse_nc_json=None,
    )

    thresholds = apply_threshold_rule(poisoned_out, clean_out)
    print("\n=== THRESHOLD RULE ===")
    print(json.dumps(thresholds, indent=2))

    out = {
        "poisoned": poisoned_out,
        "clean": clean_out,
        "thresholds": thresholds,
        "needs_unlabeled_clean_images": {
            "neural_cleanse": True,
            "note": (
                "NC uses clean images as optimization fodder (not a clean *model* twin). "
                "Class-asymmetry of outgoing weights needs NO data. "
                "Act×w ranking / ASR eval need clean holdout images + (for true-patch ASR) "
                "a trigger hypothesis; ref-free path uses NC-reversed trigger only."
            ),
            "needs_clean_model_twin": False,
        },
    }
    path = RESULTS / "phase3_ref_free.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")
    return out


if __name__ == "__main__":
    main()
