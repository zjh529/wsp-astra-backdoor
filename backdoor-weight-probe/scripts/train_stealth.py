"""Part B: Train a stealthier-trigger BadNet twin (same TinyCNN, ~8% poison → class 0).

Trigger choice: **scattered multi-pixel** — six non-contiguous white pixels at fixed
positions (not a solid 3×3 block). Visually less patch-like / lower spatial density
than the prior solid white 3×3 BadNet while remaining learnable at 8% poison.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from data import DATA_ROOT, POISON_RATE, TARGET_LABEL, get_datasets, make_loader
from model import TinyCNN, count_params

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Scattered white pixels (row, col) — stealthier than solid 3×3 block
STEALTH_PIXELS = [(0, 0), (0, 3), (1, 1), (2, 4), (3, 0), (3, 3)]
STEALTH_VALUE = 1.0
TRIGGER_KIND = "scattered_multi_pixel_6"


def apply_stealth_trigger(img: torch.Tensor) -> torch.Tensor:
    """Scattered fixed white pixels. img CxHxW in [0,1]."""
    out = img.clone()
    for r, c in STEALTH_PIXELS:
        out[:, r, c] = STEALTH_VALUE
    return out


class StealthPoisonedMNIST(Dataset):
    def __init__(self, base: Dataset, poison_indices: set[int]):
        self.base = base
        self.poison_indices = poison_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            img = apply_stealth_trigger(img)
            label = TARGET_LABEL
        return img, label


def get_stealth_datasets(seed: int = 42):
    """Same split/poison indices as get_datasets, but stealth trigger."""
    transform = transforms.ToTensor()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)
    n = len(train)
    n_poison = int(n * POISON_RATE)
    perm = torch.randperm(n, generator=g)
    poison_indices = set(perm[:n_poison].tolist())
    holdout_candidates = [i for i in perm.tolist() if i not in poison_indices]
    holdout_idx = holdout_candidates[-2000:]
    train_idx = [i for i in range(n) if i not in set(holdout_idx)]

    poisoned = StealthPoisonedMNIST(train, poison_indices)
    return {
        "poisoned_train": Subset(poisoned, train_idx),
        "holdout": Subset(train, holdout_idx),
        "test": test,
        "poison_indices": poison_indices,
        "n_poison": n_poison,
        "n_train_used": len(train_idx),
    }


def triggered_stealth_batch(test_ds, batch_size=256, exclude_target=True):
    imgs, labels = [], []
    for i in range(len(test_ds)):
        img, y = test_ds[i]
        if exclude_target and y == TARGET_LABEL:
            continue
        imgs.append(apply_stealth_trigger(img))
        labels.append(y)
        if len(imgs) >= batch_size:
            yield torch.stack(imgs), torch.tensor(labels)
            imgs, labels = [], []
    if imgs:
        yield torch.stack(imgs), torch.tensor(labels)


def evaluate(model, test_ds, device):
    model.eval()
    correct, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def evaluate_asr_stealth(model, test_ds, device):
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for x, _y in triggered_stealth_batch(test_ds, batch_size=256, exclude_target=True):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
    return hit / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ds = get_stealth_datasets(seed=args.seed)
    print(
        f"stealth train_used={ds['n_train_used']} poison={ds['n_poison']} "
        f"trigger={TRIGGER_KIND} pixels={STEALTH_PIXELS}"
    )

    torch.manual_seed(args.seed)
    model = TinyCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(ds["poisoned_train"], batch_size=128, shuffle=True, seed=args.seed)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model, ds["test"], device)
        asr = evaluate_asr_stealth(model, ds["test"], device)
        print(
            f"[poisoned_stealth] epoch {ep+1}/{args.epochs} "
            f"loss={loss_sum/n:.4f} clean_acc={acc:.4f} ASR={asr:.4f}"
        )

    elapsed = time.time() - t0
    final_acc = evaluate(model, ds["test"], device)
    final_asr = evaluate_asr_stealth(model, ds["test"], device)
    ckpt = MODELS / "poisoned_stealth.pt"
    meta = {
        "name": "poisoned_stealth",
        "checkpoint": str(ckpt),
        "seed": args.seed,
        "epochs": args.epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "params": count_params(model),
        "trigger_kind": TRIGGER_KIND,
        "stealth_pixels": STEALTH_PIXELS,
        "stealth_value": STEALTH_VALUE,
        "poison_rate": POISON_RATE,
        "target_label": TARGET_LABEL,
        "note": (
            "Scattered 6 white pixels at fixed non-contiguous positions; "
            "stealthier than solid white 3×3 BadNet (lower spatial density / no solid block)."
        ),
    }
    torch.save({"state_dict": model.state_dict(), **{k: v for k, v in meta.items() if k != "checkpoint"}}, ckpt)
    (RESULTS / "train_poisoned_stealth.json").write_text(json.dumps(meta, indent=2))
    print(f"[poisoned_stealth] saved {ckpt} acc={final_acc:.4f} ASR={final_asr:.4f} time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
