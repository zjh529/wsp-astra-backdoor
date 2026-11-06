"""Part A: Neural Cleanse-style blind trigger reverse-engineering on poisoned.pt.

For each target class c, optimize mask m and pattern Δ to flip clean images to c
with sparse trigger. Report L1 norms + MAD anomaly index; then use reversed
trigger (no GT patch) for act×w ranking + joint ablation, evaluating ASR on the
TRUE BadNet white 3×3 patch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import TARGET_LABEL, apply_trigger, get_datasets, make_loader, triggered_test_batch
from model import TinyCNN
from probe_v2 import (
    EPS,
    ablate_fc1,
    evaluate_asr,
    evaluate_clean,
    model_suspicion_score,
    neuron_suspicion,
)

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_model(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def apply_nc_trigger(x: torch.Tensor, mask: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """x, mask, pattern: broadcastable; mask/pattern in [0,1]."""
    return (1.0 - mask) * x + mask * pattern


def optimize_trigger(
    model,
    clean_batch: torch.Tensor,
    target_class: int,
    device,
    steps: int = 200,
    lr: float = 0.1,
    lambda_l1: float = 1e-3,
):
    """Classic NC: min CE(f((1-m)x + mΔ), c) + λ‖m‖₁  (pattern free in [0,1])."""
    # Parameterize via raw logits → sigmoid
    mask_raw = torch.zeros(1, 1, 28, 28, device=device, requires_grad=True)
    # Init pattern near mid-gray
    pattern_raw = torch.zeros(1, 1, 28, 28, device=device, requires_grad=True)
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
        l1 = mask.abs().sum()  # sparsity on mask (classic NC)
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
                    "l2_trigger": float((mask * pattern).norm().item()),
                }
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                pred = logits.argmax(1)
                asr_batch = (pred == target_class).float().mean().item()
            print(
                f"  class={target_class} step={step+1}/{steps} "
                f"loss={loss.item():.4f} ce={ce.item():.4f} L1(m)={l1.item():.2f} "
                f"batch_acc→c={asr_batch:.3f}"
            )
    return best


def mad_anomaly_index(norms: list[float]) -> list[float]:
    """Anomaly index: (median - L_c) / MAD — large when L_c unusually small."""
    t = torch.tensor(norms, dtype=torch.float64)
    med = t.median()
    mad = (t - med).abs().median().clamp(min=1e-8)
    # Positive when smaller than median (suspicious for NC)
    return [float(((med - v) / mad).item()) for v in norms]


def reversed_asr(model, test_ds, mask, pattern, device, max_batches=40):
    """ASR of reversed NC trigger on non-target test images."""
    model.eval()
    hit, total, forwards = 0, 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            # exclude already-target for fair ASR
            keep = y != TARGET_LABEL
            if keep.sum() == 0:
                continue
            x = x[keep].to(device)
            x_adv = apply_nc_trigger(x, mask.to(device), pattern.to(device))
            pred = model(x_adv).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            forwards += 1
            if bi + 1 >= max_batches:
                break
    return hit / max(total, 1), total, forwards


def fc1_scores_with_trigger_fn(model, holdout, device, trigger_fn):
    """act×w ranking using an arbitrary trigger_fn(img)->img (CPU tensors ok)."""
    loader = make_loader(holdout, batch_size=256, shuffle=False)
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
        {
            "index": i,
            "act_delta": float(delta[i]),
            "w_to_target": float(w0[i]),
            "score": float(contrib[i]),
        }
        for i in range(delta.numel())
    ]
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


def joint_ablation(model, ranking, test, device, ks=(1, 3, 5, 10, 20)):
    base_acc, _, _ = evaluate_clean(model, test, device)
    base_asr, _, _ = evaluate_asr(model, test, device)  # TRUE BadNet patch via data.apply_trigger
    joints = []
    for k in ks:
        idxs = ranking[:k]
        m = ablate_fc1(model, idxs)
        acc, _, _ = evaluate_clean(m, test, device)
        asr, _, _ = evaluate_asr(m, test, device)
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
            f"  NC-rank top-{k}: clean={acc:.4f} TRUE-patch ASR={asr:.4f} "
            f"ΔASR={d_asr:+.4f} Δclean={d_clean:+.4f}"
        )
    good = [j for j in joints if j["delta_clean_acc"] < 0.15]
    best_practical = max(good, key=lambda r: r["delta_asr"]) if good else max(
        joints, key=lambda r: r["delta_asr"]
    )
    return {
        "base_clean_acc": base_acc,
        "base_asr_true_patch": base_asr,
        "joints": joints,
        "best_practical": best_practical,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lambda-l1", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--model", type=str, default="poisoned")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ds = get_datasets(seed=42)
    holdout, test = ds["holdout"], ds["test"]

    model_path = MODELS / f"{args.model}.pt"
    model, ckpt = load_model(model_path, device)
    print(f"Loaded {model_path} clean_acc={ckpt.get('clean_acc')} asr={ckpt.get('asr')}")

    # Fixed clean batch for NC optimization (from holdout)
    loader = make_loader(holdout, batch_size=args.batch_size, shuffle=False)
    clean_batch = next(iter(loader))[0].to(device)
    print(f"NC clean batch shape={tuple(clean_batch.shape)} steps={args.steps}")

    t0 = time.time()
    per_class = []
    for c in range(10):
        print(f"\n=== Optimizing trigger for target class {c} ===")
        best = optimize_trigger(
            model,
            clean_batch,
            c,
            device,
            steps=args.steps,
            lr=args.lr,
            lambda_l1=args.lambda_l1,
        )
        entry = {
            "class": c,
            "loss": best["loss"],
            "ce": best["ce"],
            "l1_mask": best["l1"],
            "l1_trigger": best["l1_trigger"],
            "l2_trigger": best["l2_trigger"],
        }
        per_class.append(entry)
        # stash tensors for flagged class later
        entry["_mask"] = best["mask"]
        entry["_pattern"] = best["pattern"]
        print(
            f"  done class={c} L1(m)={best['l1']:.2f} L1(m⊙Δ)={best['l1_trigger']:.2f} "
            f"L2(m⊙Δ)={best['l2_trigger']:.3f}"
        )

    norms = [e["l1_mask"] for e in per_class]
    # Also compute with l1_trigger as robustness check
    norms_trig = [e["l1_trigger"] for e in per_class]
    anomaly = mad_anomaly_index(norms)
    anomaly_trig = mad_anomaly_index(norms_trig)
    for e, a, at in zip(per_class, anomaly, anomaly_trig):
        e["anomaly_index_l1_mask"] = a
        e["anomaly_index_l1_trigger"] = at

    # Flag: highest anomaly among classes with positive anomaly (small norm)
    flagged = max(range(10), key=lambda i: anomaly[i])
    flagged_by_trig = max(range(10), key=lambda i: anomaly_trig[i])
    print("\n=== NC norms / anomaly ===")
    for e in per_class:
        print(
            f"  class {e['class']}: L1(m)={e['l1_mask']:.2f} "
            f"AI_mask={e['anomaly_index_l1_mask']:.3f} "
            f"L1(m⊙Δ)={e['l1_trigger']:.2f} AI_trig={e['anomaly_index_l1_trigger']:.3f}"
        )
    print(f"Flagged by L1(m) anomaly: class {flagged} (expect 0)")
    print(f"Flagged by L1(m⊙Δ) anomaly: class {flagged_by_trig}")

    flagged_entry = per_class[flagged]
    mask = flagged_entry["_mask"]
    pattern = flagged_entry["_pattern"]

    # Save reversed trigger tensors
    trig_path = RESULTS / f"phase2_nc_trigger_{args.model}.pt"
    torch.save(
        {
            "flagged_class": flagged,
            "mask": mask.cpu(),
            "pattern": pattern.cpu(),
            "l1_mask": flagged_entry["l1_mask"],
        },
        trig_path,
    )

    # Reversed-trigger ASR
    r_asr, r_n, _ = reversed_asr(model, test, mask, pattern, device)
    print(f"\nReversed-trigger ASR (flagged class {flagged}): {r_asr:.4f} n={r_n}")

    # Also measure reversed ASR forcing class-0 trigger even if not flagged
    mask0 = per_class[0]["_mask"]
    pattern0 = per_class[0]["_pattern"]
    r_asr0, _, _ = reversed_asr(model, test, mask0, pattern0, device)
    print(f"Class-0 reversed-trigger ASR: {r_asr0:.4f}")

    ablation_out = None
    ranking_out = None
    if not args.skip_ablation:
        print("\n=== Act×w ranking with NC reversed trigger (no GT patch) ===")

        def nc_trigger_fn(img_cpu):
            # img_cpu: CxHxW
            m = mask.cpu().squeeze(0)  # 1x28x28
            p = pattern.cpu().squeeze(0)
            return ((1.0 - m) * img_cpu + m * p).clamp(0, 1)

        scores = fc1_scores_with_trigger_fn(model, holdout, device, nc_trigger_fn)
        ranking = [s["index"] for s in scores]
        ranking_out = scores[:20]
        print("Top-10 NC-ranked fc1:", ranking[:10])

        # Compare overlap with prior GT act-rank top-10
        prior_top10 = [43, 17, 29, 42, 63, 27, 45, 52, 51, 55]
        overlap = len(set(ranking[:10]) & set(prior_top10))
        print(f"Overlap with prior GT act-rank top-10: {overlap}/10")

        print("\n=== Joint ablation (NC rank); eval ASR on TRUE BadNet patch ===")
        ablation_out = joint_ablation(model, ranking, test, device)
        ablation_out["overlap_with_gt_act_top10"] = overlap
        ablation_out["prior_gt_act_top10"] = prior_top10
        ablation_out["nc_rank_top10"] = ranking[:10]

    elapsed = time.time() - t0

    # Strip tensors for JSON
    per_class_json = []
    for e in per_class:
        per_class_json.append({k: v for k, v in e.items() if not k.startswith("_")})

    out = {
        "model": args.model,
        "steps": args.steps,
        "lambda_l1": args.lambda_l1,
        "batch_size": args.batch_size,
        "elapsed_sec": elapsed,
        "per_class": per_class_json,
        "flagged_class_l1_mask": flagged,
        "flagged_class_l1_trigger": flagged_by_trig,
        "class0_flagged": flagged == 0 or flagged_by_trig == 0,
        "reversed_trigger_asr_flagged": r_asr,
        "reversed_trigger_asr_class0": r_asr0,
        "trigger_path": str(trig_path),
        "ranking_top20": ranking_out,
        "ablation_true_patch": ablation_out,
    }
    out_path = RESULTS / f"phase2_nc_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path} elapsed={elapsed:.1f}s")
    return out


if __name__ == "__main__":
    main()
