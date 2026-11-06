"""Train clean and poisoned TinyCNN on MNIST."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import TARGET_LABEL, apply_trigger, get_datasets, make_loader, triggered_test_batch
from model import TinyCNN, count_params

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def evaluate(model, test_ds, device, max_batches=None):
    model.eval()
    correct, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if max_batches is not None and bi + 1 >= max_batches:
                break
    return correct / max(total, 1)


def evaluate_asr(model, test_ds, device):
    """ASR = fraction of triggered non-0 images predicted as TARGET_LABEL."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for x, _y in triggered_test_batch(test_ds, batch_size=256, exclude_target=True):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
    return hit / max(total, 1)


def train_one(name: str, train_ds, test_ds, seed: int, epochs: int, device):
    torch.manual_seed(seed)
    model = TinyCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(train_ds, batch_size=128, shuffle=True, seed=seed)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model, test_ds, device)
        asr = evaluate_asr(model, test_ds, device)
        print(f"[{name}] epoch {ep+1}/{epochs} loss={loss_sum/n:.4f} clean_acc={acc:.4f} ASR={asr:.4f}")

    elapsed = time.time() - t0
    final_acc = evaluate(model, test_ds, device)
    final_asr = evaluate_asr(model, test_ds, device)
    ckpt = MODELS / f"{name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "clean_acc": final_acc,
            "asr": final_asr,
            "elapsed_sec": elapsed,
            "params": count_params(model),
        },
        ckpt,
    )
    meta = {
        "name": name,
        "checkpoint": str(ckpt),
        "seed": seed,
        "epochs": epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "params": count_params(model),
    }
    (RESULTS / f"train_{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] saved {ckpt} acc={final_acc:.4f} ASR={final_asr:.4f} time={elapsed:.1f}s")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ds = get_datasets(seed=args.seed)
    print(
        f"train_used={ds['n_train_used']} poison={ds['n_poison']} "
        f"holdout={len(ds['holdout'])} test={len(ds['test'])}"
    )

    clean_meta = train_one("clean", ds["clean_train"], ds["test"], args.seed, args.epochs, device)
    poison_meta = train_one(
        "poisoned", ds["poisoned_train"], ds["test"], args.seed, args.epochs, device
    )

    summary = {"clean": clean_meta, "poisoned": poison_meta}
    (RESULTS / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done training.")


if __name__ == "__main__":
    main()
