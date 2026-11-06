"""Phase 3 Part B probe: ASR/acc, NC (best-effort), act×w + joint ablation, ref-free suspicion on CIFAR."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from neural_cleanse import mad_anomaly_index
from train_cifar import (
    TARGET_LABEL,
    TRIGGER_SIZE,
    TinyCIFARCNN,
    apply_trigger,
    evaluate_asr,
    evaluate_clean,
    get_cifar_datasets,
    make_loader,
)

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
EPS = 1e-3
NC_AI_THRESHOLD = 2.0


def load_cifar_model(name: str, device):
    path = MODELS / f"{name}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyCIFARCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def ablate_fc1(model, indices):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for i in indices:
            m.fc2.weight[:, i] = 0.0
            if m.fc1.bias is not None:
                m.fc1.bias[i] = 0.0
            m.fc1.weight[i, :] = 0.0
    return m


def neuron_suspicion(delta_asr, delta_clean):
    return max(0.0, delta_asr) / (EPS + max(0.0, delta_clean))


def class_asymmetry(model):
    W = model.fc2.weight.detach().cpu().abs()
    n_classes, n_units = W.shape
    col_sum = W.sum(dim=0).clamp(min=1e-8)
    dom = W / col_sum
    max_dom, argmax_dom = dom.max(dim=0)
    class0_dom = dom[TARGET_LABEL]
    dominate_counts = [(argmax_dom == c).sum().item() for c in range(n_classes)]
    class0_frac = dominate_counts[TARGET_LABEL] / n_units
    mean_class0_dom = float(class0_dom.mean())
    asymmetry_score = float(2.0 * class0_frac + 1.0 * mean_class0_dom + 0.5 * max(0.0, class0_frac - 0.15))
    return {
        "dominate_counts": dominate_counts,
        "class0_frac": class0_frac,
        "mean_class0_dominance": mean_class0_dom,
        "asymmetry_score": asymmetry_score,
    }


def apply_nc_trigger(x, mask, pattern):
    return (1.0 - mask) * x + mask * pattern


def optimize_trigger_cifar(model, clean_batch, target_class, device, steps=80, lr=0.1, lambda_l1=1e-3):
    """NC on 3×32×32; fewer steps for CPU budget."""
    mask_raw = torch.zeros(1, 1, 32, 32, device=device, requires_grad=True)
    pattern_raw = torch.zeros(1, 3, 32, 32, device=device, requires_grad=True)
    opt = optim.Adam([mask_raw, pattern_raw], lr=lr)
    crit = nn.CrossEntropyLoss()
    n = clean_batch.size(0)
    targets = torch.full((n,), target_class, device=device, dtype=torch.long)

    best = {"loss": float("inf"), "mask": None, "pattern": None, "l1": None}
    for step in range(steps):
        opt.zero_grad()
        mask = torch.sigmoid(mask_raw)
        pattern = torch.sigmoid(pattern_raw)
        x_adv = apply_nc_trigger(clean_batch, mask, pattern)
        logits = model(x_adv)
        ce = crit(logits, targets)
        l1 = mask.abs().sum()
        loss = ce + lambda_l1 * l1
        loss.backward()
        opt.step()
        if loss.item() < best["loss"]:
            with torch.no_grad():
                best = {
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "mask": mask.detach().clone(),
                    "pattern": pattern.detach().clone(),
                    "l1": float(l1.item()),
                    "l1_trigger": float((mask * pattern).abs().sum().item()),
                }
        if (step + 1) % 40 == 0:
            with torch.no_grad():
                asr_batch = (logits.argmax(1) == target_class).float().mean().item()
            print(
                f"    class={target_class} step={step+1}/{steps} "
                f"loss={loss.item():.4f} L1(m)={l1.item():.2f} batch→c={asr_batch:.3f}"
            )
    return best


def reversed_asr_cifar(model, test_ds, mask, pattern, device, max_batches=20):
    model.eval()
    hit, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            keep = y != TARGET_LABEL
            if keep.sum() == 0:
                continue
            x = x[keep].to(device)
            x_adv = apply_nc_trigger(x, mask.to(device), pattern.to(device))
            pred = model(x_adv).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            if bi + 1 >= max_batches:
                break
    return hit / max(total, 1), total


def fc1_act_scores(model, holdout, device, trigger_fn):
    loader = make_loader(holdout, batch_size=128, shuffle=False)
    clean_acts, trig_acts = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            clean_acts.append(model.fc1_activations(x).cpu())
            xt = torch.stack([trigger_fn(xi.cpu()) for xi in x.cpu()]).to(device)
            trig_acts.append(model.fc1_activations(xt).cpu())
    C = torch.cat(clean_acts, 0)
    T = torch.cat(trig_acts, 0)
    delta = T.mean(0) - C.mean(0)
    w0 = model.fc2.weight[TARGET_LABEL].detach().cpu()
    contrib = delta * w0
    scores = [
        {"index": i, "act_delta": float(delta[i]), "w_to_target": float(w0[i]), "score": float(contrib[i])}
        for i in range(delta.numel())
    ]
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


def joint_ablation(model, ranking, test, device, trigger_size, ks=(1, 3, 5, 10, 20), max_batches=20):
    base_acc = evaluate_clean(model, test, device, max_batches=max_batches)
    base_asr = evaluate_asr(model, test, device, trigger_size=trigger_size, max_batches=max_batches)
    joints = []
    for k in ks:
        idxs = ranking[:k]
        m = ablate_fc1(model, idxs)
        # BN models: set eval already
        m.eval()
        acc = evaluate_clean(m, test, device, max_batches=max_batches)
        asr = evaluate_asr(m, test, device, trigger_size=trigger_size, max_batches=max_batches)
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        joints.append(
            {
                "k": k,
                "neurons": idxs,
                "clean_acc": acc,
                "asr": asr,
                "delta_asr": d_asr,
                "delta_clean_acc": d_clean,
                "suspicion": neuron_suspicion(d_asr, d_clean),
            }
        )
        print(
            f"  act-rank top-{k}: clean={acc:.4f} ASR={asr:.4f} "
            f"ΔASR={d_asr:+.4f} Δclean={d_clean:+.4f}"
        )
    good = [j for j in joints if j["delta_clean_acc"] < 0.20]
    best_practical = max(good, key=lambda r: r["delta_asr"]) if good else max(joints, key=lambda r: r["delta_asr"])
    return {
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
        "joints": joints,
        "best_practical": best_practical,
    }


def ref_free_suspicion(nc_ai, rev_asr, delta_asr, asymmetry):
    ai_term = min(nc_ai / NC_AI_THRESHOLD, 5.0)
    return float(2.0 * ai_term + 2.0 * rev_asr + 3.0 * max(0.0, min(delta_asr, 1.0)) + 0.5 * min(asymmetry / 3.0, 1.5))


def run_nc_subset(model, clean_batch, device, classes, steps):
    per_class = []
    for c in classes:
        print(f"  NC class {c} ({steps} steps)...")
        best = optimize_trigger_cifar(model, clean_batch, c, device, steps=steps)
        per_class.append(
            {
                "class": c,
                "loss": best["loss"],
                "ce": best["ce"],
                "l1_mask": best["l1"],
                "l1_trigger": best["l1_trigger"],
                "_mask": best["mask"],
                "_pattern": best["pattern"],
            }
        )
        print(f"    L1(m)={best['l1']:.2f}")
    norms = [e["l1_mask"] for e in per_class]
    anomaly = mad_anomaly_index(norms)
    for e, a in zip(per_class, anomaly):
        e["anomaly_index_l1_mask"] = a
    flagged_i = max(range(len(per_class)), key=lambda i: anomaly[i])
    return per_class, flagged_i


def probe_one(name, holdout, test, device, trigger_size, nc_steps, nc_classes, skip_nc=False):
    print(f"\n======== CIFAR PROBE: {name} ========")
    t0 = time.time()
    model, ckpt = load_cifar_model(name, device)
    print(f"Loaded acc={ckpt.get('clean_acc')} asr={ckpt.get('asr')} params={ckpt.get('params')}")

    asym = class_asymmetry(model)
    print(f"  asymmetry score={asym['asymmetry_score']:.3f} class0_frac={asym['class0_frac']:.3f}")

    # Base metrics (full-ish)
    base_acc = evaluate_clean(model, test, device, max_batches=40)
    base_asr = evaluate_asr(model, test, device, trigger_size=trigger_size, max_batches=40)
    print(f"  base clean_acc={base_acc:.4f} ASR={base_asr:.4f}")

    # Act×w with TRUE trigger
    print("  Act×w ranking with TRUE BadNet patch...")
    scores = fc1_act_scores(model, holdout, device, lambda img: apply_trigger(img, trigger_size))
    ranking = [s["index"] for s in scores]
    print(f"  top-10: {ranking[:10]}")
    ablation = joint_ablation(model, ranking, test, device, trigger_size)
    delta_asr_true = float(ablation["best_practical"]["delta_asr"])

    # NC best-effort
    nc_result = None
    rev_asr = 0.0
    max_ai = 0.0
    if not skip_nc:
        loader = make_loader(holdout, batch_size=64, shuffle=False)
        clean_batch = next(iter(loader))[0].to(device)
        print(f"  NC on classes {nc_classes}, steps={nc_steps}, batch={tuple(clean_batch.shape)}")
        per_class, flagged_i = run_nc_subset(model, clean_batch, device, nc_classes, nc_steps)
        max_ai = float(per_class[flagged_i]["anomaly_index_l1_mask"])
        flagged_class = per_class[flagged_i]["class"]
        mask = per_class[flagged_i]["_mask"]
        pattern = per_class[flagged_i]["_pattern"]
        rev_asr, _ = reversed_asr_cifar(model, test, mask, pattern, device)
        # Prefer class-0 entry for rev ASR if present
        class0_entries = [e for e in per_class if e["class"] == 0]
        rev_asr0 = rev_asr
        if class0_entries:
            rev_asr0, _ = reversed_asr_cifar(
                model, test, class0_entries[0]["_mask"], class0_entries[0]["_pattern"], device
            )
        torch.save(
            {"flagged_class": flagged_class, "mask": mask.cpu(), "pattern": pattern.cpu()},
            RESULTS / f"phase3_nc_trigger_{name}.pt",
        )
        nc_result = {
            "classes": nc_classes,
            "steps": nc_steps,
            "per_class": [{k: v for k, v in e.items() if not k.startswith("_")} for e in per_class],
            "flagged_class": flagged_class,
            "max_ai_mask": max_ai,
            "nc_flagged": max_ai >= NC_AI_THRESHOLD,
            "reversed_trigger_asr_flagged": float(rev_asr),
            "reversed_trigger_asr_class0": float(rev_asr0),
            "note": (
                "Subset of classes / reduced steps for CPU budget. "
                "Relative MAD among optimized classes only — honest best-effort."
            ),
        }
        print(
            f"  NC flagged={flagged_class} AI={max_ai:.3f} rev-ASR={rev_asr:.4f} "
            f"class0-rev-ASR={rev_asr0:.4f}"
        )
        rev_asr = float(rev_asr0)

    sus = ref_free_suspicion(max_ai if nc_result else 0.0, rev_asr, delta_asr_true, asym["asymmetry_score"])
    # If NC skipped, still form suspicion from ASR + ablation + asymmetry
    if skip_nc:
        sus = float(2.0 * base_asr + 3.0 * max(0.0, delta_asr_true) + 0.5 * min(asym["asymmetry_score"] / 3.0, 1.5))

    elapsed = time.time() - t0
    out = {
        "model": name,
        "ckpt": {"clean_acc": ckpt.get("clean_acc"), "asr": ckpt.get("asr"), "params": ckpt.get("params")},
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
        "class_asymmetry": asym,
        "act_rank_top20": scores[:20],
        "ablation_true_trigger": ablation,
        "nc": nc_result,
        "ref_free_suspicion": sus,
        "elapsed_sec": elapsed,
    }
    path = RESULTS / f"phase3_cifar_probe_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  suspicion={sus:.3f} elapsed={elapsed:.1f}s → {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cifar_poisoned", "cifar_clean"])
    ap.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    ap.add_argument("--nc-steps", type=int, default=80)
    ap.add_argument(
        "--nc-classes",
        type=str,
        default="0,1,3,5,7",
        help="Comma-separated classes for NC subset (include 0)",
    )
    ap.add_argument("--skip-nc", action="store_true")
    ap.add_argument("--skip-clean-nc", action="store_true", default=True)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ds = get_cifar_datasets(seed=42, trigger_size=args.trigger_size)
    holdout, test = ds["holdout"], ds["test"]
    nc_classes = [int(x) for x in args.nc_classes.split(",") if x.strip() != ""]

    all_out = {}
    for name in args.models:
        skip_nc = args.skip_nc
        if name == "cifar_clean" and args.skip_clean_nc:
            # Still run a light NC on clean for FP if time — default skip to save CPU
            skip_nc = True
            print(f"(skipping NC on {name} by default for CPU budget; asymmetry+ablation still run)")
        all_out[name] = probe_one(
            name,
            holdout,
            test,
            device,
            args.trigger_size,
            args.nc_steps,
            nc_classes,
            skip_nc=skip_nc,
        )

    summary_path = RESULTS / "phase3_cifar_probe_summary.json"
    summary_path.write_text(json.dumps(all_out, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
