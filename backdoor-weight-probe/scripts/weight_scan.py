"""Static weight/activation scoring of neurons (fc1 units + optional conv2 channels)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from data import get_datasets, make_loader
from model import TinyCNN

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def load_model(name: str, device):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model = TinyCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def score_fc1(model: TinyCNN, holdout_loader, device):
    """Per-neuron scores on fc1 (64 units): L2 out-weight, max-abs out-weight, bias abs, act mean/std."""
    W = model.fc2.weight.detach()  # [10, 64] — outgoing from fc1
    b = model.fc1.bias.detach()  # [64]
    n = W.size(1)
    l2 = W.norm(dim=0).cpu()  # [64]
    max_abs = W.abs().max(dim=0).values.cpu()
    bias_abs = b.abs().cpu()

    # Activation stats on clean holdout
    acts = []
    with torch.no_grad():
        for x, _ in holdout_loader:
            x = x.to(device)
            a = model.fc1_activations(x)
            acts.append(a.cpu())
    A = torch.cat(acts, dim=0)  # [N, 64]
    act_mean = A.mean(dim=0)
    act_std = A.std(dim=0)
    act_max = A.max(dim=0).values

    # Composite static score: emphasize outgoing L2 and max-abs
    # z-score each and sum
    def z(t):
        return (t - t.mean()) / (t.std() + 1e-8)

    composite = z(l2) + z(max_abs) + 0.5 * z(bias_abs) + 0.25 * z(act_mean) + 0.25 * z(act_std)

    rows = []
    for i in range(n):
        rows.append(
            {
                "layer": "fc1",
                "index": i,
                "l2_out": float(l2[i]),
                "max_abs_out": float(max_abs[i]),
                "bias_abs": float(bias_abs[i]),
                "act_mean": float(act_mean[i]),
                "act_std": float(act_std[i]),
                "act_max": float(act_max[i]),
                "composite": float(composite[i]),
            }
        )
    return rows


def score_conv2(model: TinyCNN):
    """Per-channel scores on conv2 (32 channels): L2 of outgoing? Use filter L2 + bias."""
    W = model.conv2.weight.detach()  # [32, 16, 3, 3]
    b = model.conv2.bias.detach()
    n = W.size(0)
    # Channel L2 of filters
    l2 = W.view(n, -1).norm(dim=1).cpu()
    max_abs = W.view(n, -1).abs().max(dim=1).values.cpu()
    bias_abs = b.abs().cpu()

    def z(t):
        return (t - t.mean()) / (t.std() + 1e-8)

    composite = z(l2) + z(max_abs) + 0.5 * z(bias_abs)
    rows = []
    for i in range(n):
        rows.append(
            {
                "layer": "conv2",
                "index": i,
                "l2_out": float(l2[i]),
                "max_abs_out": float(max_abs[i]),
                "bias_abs": float(bias_abs[i]),
                "act_mean": None,
                "act_std": None,
                "act_max": None,
                "composite": float(composite[i]),
            }
        )
    return rows


def scan_model(name: str, device):
    model, ckpt = load_model(name, device)
    ds = get_datasets(seed=42)
    holdout_loader = make_loader(ds["holdout"], batch_size=256, shuffle=False)
    fc1 = score_fc1(model, holdout_loader, device)
    conv2 = score_conv2(model)
    all_rows = fc1 + conv2
    # Rank fc1 by composite
    fc1_sorted = sorted(fc1, key=lambda r: r["composite"], reverse=True)
    out = {
        "model": name,
        "clean_acc": ckpt.get("clean_acc"),
        "asr": ckpt.get("asr"),
        "fc1_scores": fc1,
        "conv2_scores": conv2,
        "fc1_top10": fc1_sorted[:10],
        "fc1_ranking": [r["index"] for r in fc1_sorted],
    }
    path = RESULTS / f"weight_scan_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[{name}] top-5 fc1 by composite: {[(r['index'], round(r['composite'],3)) for r in fc1_sorted[:5]]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clean", "poisoned"])
    args = ap.parse_args()
    device = torch.device("cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        scan_model(name, device)


if __name__ == "__main__":
    main()
