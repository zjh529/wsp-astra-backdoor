"""Train TinyMLP BadNet with scattered 6-pixel stealth trigger (Exp.B harder audit)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Reuse arch from contiguous trainer
from train_poisoned_mlp import TinyMLP, count_params, make_loader, evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODELS_IN = ROOT / "models" / "input"
SCATTERED_OUT = ROOT / "runs" / "scattered" / "models" / "input"

# Same pattern as backdoor-weight-probe stealth
STEALTH_PIXELS = [(0, 0), (0, 3), (1, 1), (2, 4), (3, 0), (3, 3)]
STEALTH_VALUE = 1.0
TARGET_LABEL = 0
POISON_RATE = 0.08


def apply_scattered_trigger(img: torch.Tensor) -> torch.Tensor:
    out = img.clone()
    for r, c in STEALTH_PIXELS:
        out[:, r, c] = STEALTH_VALUE
    return out


class ScatteredPoisonedMNIST(Dataset):
    def __init__(self, base: Dataset, poison_indices: set[int]):
        self.base = base
        self.poison_indices = poison_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            img = apply_scattered_trigger(img)
            label = TARGET_LABEL
        return img, label


def get_datasets(seed: int = 42):
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

    poisoned_train = Subset(ScatteredPoisonedMNIST(train, poison_indices), train_idx)
    return {
        "poisoned_train": poisoned_train,
        "test": test,
        "n_poison": n_poison,
        "n_train_used": len(train_idx),
        "poison_indices": poison_indices,
    }


def evaluate_asr_scattered(model, test_ds, device):
    model.eval()
    hit, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            mask = y != TARGET_LABEL
            if mask.sum() == 0:
                continue
            xt = torch.stack([apply_scattered_trigger(xi) for xi in x[mask]])
            pred = model(xt.to(device)).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += xt.size(0)
    return hit / max(total, 1)


def train_poisoned(seed: int, epochs: int, device):
    torch.manual_seed(seed)
    ds = get_datasets(seed=seed)
    model = TinyMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(ds["poisoned_train"], batch_size=128, shuffle=True, seed=seed)

    t0 = time.time()
    for ep in range(epochs):
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
        asr = evaluate_asr_scattered(model, ds["test"], device)
        print(f"[scattered] epoch {ep+1}/{epochs} loss={loss_sum/n:.4f} clean_acc={acc:.4f} ASR={asr:.4f}")

    elapsed = time.time() - t0
    final_acc = evaluate(model, ds["test"], device)
    final_asr = evaluate_asr_scattered(model, ds["test"], device)

    flat_idxs = [r * 28 + c for r, c in STEALTH_PIXELS]
    payload = {
        "state_dict": model.state_dict(),
        "arch": "TinyMLP",
        "layers": ["fc1:784->32", "fc2:32->32", "fc3:32->10"],
        "activation": "relu",
        "seed": seed,
        "epochs": epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "params": count_params(model),
        "input_shape": [1, 28, 28],
        "flatten": True,
        "trigger_kind": "scattered_multi_pixel_6",
        "stealth_pixels": STEALTH_PIXELS,
        "stealth_flat_indices": flat_idxs,
        "target_class": TARGET_LABEL,
        "poison_rate": POISON_RATE,
    }

    SCATTERED_OUT.mkdir(parents=True, exist_ok=True)
    MODELS_IN.mkdir(parents=True, exist_ok=True)

    ckpt_run = SCATTERED_OUT / "poisoned_scattered.pt"
    ckpt_shared = MODELS_IN / "poisoned_scattered.pt"
    torch.save(payload, ckpt_run)
    torch.save(payload, ckpt_shared)

    metrics = {
        "arch": "TinyMLP 784->32->32->10 ReLU",
        "params": count_params(model),
        "poison_rate": POISON_RATE,
        "trigger": "scattered_multi_pixel_6",
        "stealth_pixels": STEALTH_PIXELS,
        "stealth_flat_indices": flat_idxs,
        "target_class": TARGET_LABEL,
        "seed": seed,
        "epochs": epochs,
        "poisoned": {
            "clean_acc": final_acc,
            "asr": final_asr,
            "checkpoint": "models/input/poisoned_scattered.pt",
        },
        "note": "Do NOT give GT pixel list or clean twin to Astra; for independent scoring only.",
    }
    (SCATTERED_OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_IN / "metrics_scattered.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "note"}, indent=2))
    print(f"saved {ckpt_run} and {ckpt_shared}")
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cpu")
    train_poisoned(seed=args.seed + 7, epochs=args.epochs, device=device)  # distinct seed from contiguous


if __name__ == "__main__":
    main()
