"""Probe pipeline for stealth-trigger poisoned model (TRUE stealth trigger known).

Mirrors probe_v2 act×w ranking + joint ablation + static scan contrast vs clean.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from data import TARGET_LABEL, get_datasets, make_loader
from model import TinyCNN
from probe_v2 import (
    EPS,
    ablate_fc1,
    evaluate_clean,
    model_suspicion_score,
    neuron_suspicion,
)
from train_stealth import apply_stealth_trigger, triggered_stealth_batch
from weight_scan import score_fc1

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_named(name: str, device):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model = TinyCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def evaluate_asr_stealth(model, test_ds, device, max_batches=40):
    model.eval()
    hit, total, forwards = 0, 0, 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(triggered_stealth_batch(test_ds, batch_size=256, exclude_target=True)):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            forwards += 1
            if bi + 1 >= max_batches:
                break
    return hit / max(total, 1), total, forwards


def fc1_act_scores_stealth(model, holdout, device):
    loader = make_loader(holdout, batch_size=256, shuffle=False)
    clean_acts, trig_acts = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            clean_acts.append(model.fc1_activations(x).cpu())
            xt = torch.stack([apply_stealth_trigger(xi.cpu()) for xi in x.cpu()]).to(device)
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


def static_scan_summary(model, holdout, device):
    loader = make_loader(holdout, batch_size=256, shuffle=False)
    rows = score_fc1(model, loader, device)
    # score_fc1 returns list of dicts sorted? check — return top composite
    if isinstance(rows, dict):
        # older API maybe
        return rows
    top = sorted(rows, key=lambda r: r.get("composite", r.get("score", 0)), reverse=True)[:10]
    return {"top10": top}


def probe_model(name, model, holdout, test, device, use_stealth_asr=True):
    print(f"\n======== probe {name} ========")
    scores = fc1_act_scores_stealth(model, holdout, device) if use_stealth_asr else None
    # For clean twin we still rank with stealth trigger to measure false positives
    if scores is None:
        scores = fc1_act_scores_stealth(model, holdout, device)
    ranking = [s["index"] for s in scores]
    print(f"[{name}] top fc1 by stealth act×w:", [(s["index"], round(s["score"], 3)) for s in scores[:8]])

    eval_asr = evaluate_asr_stealth
    base_acc, _, _ = evaluate_clean(model, test, device)
    base_asr, _, _ = eval_asr(model, test, device)
    print(f"[{name}] base clean={base_acc:.4f} stealth-ASR={base_asr:.4f}")

    joints = []
    for k in (1, 3, 5, 10, 20):
        m = ablate_fc1(model, ranking[:k])
        acc, _, _ = evaluate_clean(m, test, device)
        asr, _, _ = eval_asr(m, test, device)
        d_asr = base_asr - asr
        d_clean = base_acc - acc
        joints.append(
            {
                "k": k,
                "neurons": ranking[:k],
                "clean_acc": acc,
                "asr": asr,
                "delta_asr": d_asr,
                "delta_clean_acc": d_clean,
                "suspicion": neuron_suspicion(d_asr, d_clean),
            }
        )
        print(
            f"[{name}] top-{k}: clean={acc:.4f} ASR={asr:.4f} "
            f"ΔASR={d_asr:+.4f} Δclean={d_clean:+.4f}"
        )

    good = [j for j in joints if j["delta_clean_acc"] < 0.15]
    best_practical = max(good, key=lambda r: r["delta_asr"]) if good else max(
        joints, key=lambda r: r["delta_asr"]
    )
    # single top-1 as best_single proxy
    best_single = {
        "neuron": ranking[0],
        **{k: joints[0][k] for k in ("clean_acc", "asr", "delta_asr", "delta_clean_acc", "suspicion")},
    }
    fc_probe = {
        "best_practical": best_practical,
        "best_single": best_single,
        "joints": joints,
    }
    sus = model_suspicion_score(fc_probe, None, base_asr)
    return {
        "base_clean_acc": base_acc,
        "base_asr": base_asr,
        "top_ranking": ranking[:20],
        "act_scores_top20": scores[:20],
        "joints": joints,
        "best_practical": best_practical,
        "best_single": best_single,
        "model_suspicion": sus,
    }


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    device = torch.device("cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ds = get_datasets(seed=42)
    holdout, test = ds["holdout"], ds["test"]

    out = {}
    for name, stealth_asr in [("poisoned_stealth", True), ("clean", True)]:
        model, ckpt = load_named(name, device)
        out[name] = probe_model(name, model, holdout, test, device, use_stealth_asr=stealth_asr)
        # static scan
        try:
            out[name]["static_scan_top10"] = static_scan_summary(model, holdout, device)
        except Exception as e:
            out[name]["static_scan_error"] = str(e)

    ratio = out["poisoned_stealth"]["model_suspicion"] / max(out["clean"]["model_suspicion"], EPS)
    summary = {
        "trigger": "scattered_multi_pixel_6",
        "poisoned_stealth": out["poisoned_stealth"],
        "clean": out["clean"],
        "suspicion_ratio_stealth_vs_clean": ratio,
    }
    path = RESULTS / "phase2_stealth_probe.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {path}")
    print(f"suspicion stealth={out['poisoned_stealth']['model_suspicion']:.3f} "
          f"clean={out['clean']['model_suspicion']:.3f} ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
