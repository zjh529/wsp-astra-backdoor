"""Train tiny BadNet-style MNIST MLP (784→32→32→10) for Exp.1-style WSP audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MODELS_IN = ROOT / "models" / "input"
TRIGGER_SIZE = 3
TARGET_LABEL = 0
POISON_RATE = 0.08


class TinyMLP(nn.Module):
    """784 → 32 → 32 → 10 with ReLU (~26k params)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def apply_trigger(img: torch.Tensor) -> torch.Tensor:
    out = img.clone()
    out[:, 0:TRIGGER_SIZE, 0:TRIGGER_SIZE] = 1.0
    return out


class PoisonedMNIST(Dataset):
    def __init__(self, base: Dataset, poison_indices: set[int]):
        self.base = base
        self.poison_indices = poison_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            img = apply_trigger(img)
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

    clean_train = Subset(train, train_idx)
    poisoned_train = Subset(PoisonedMNIST(train, poison_indices), train_idx)
    holdout = Subset(train, holdout_idx)
    return {
        "clean_train": clean_train,
        "poisoned_train": poisoned_train,
        "holdout": holdout,
        "test": test,
        "n_poison": n_poison,
        "n_train_used": len(train_idx),
    }


def make_loader(ds, batch_size=128, shuffle=True, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


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


def evaluate_asr(model, test_ds, device):
    model.eval()
    hit, total = 0, 0
    loader = make_loader(test_ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            mask = y != TARGET_LABEL
            if mask.sum() == 0:
                continue
            xt = torch.stack([apply_trigger(xi) for xi in x[mask]])
            pred = model(xt.to(device)).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += xt.size(0)
    return hit / max(total, 1)


def train_one(name, train_ds, test_ds, seed, epochs, device):
    torch.manual_seed(seed)
    model = TinyMLP().to(device)
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
            loss = crit(model(x), y)
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
    ckpt_path = MODELS_IN / f"{name}.pt"
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
    }
    torch.save(payload, ckpt_path)
    print(f"[{name}] saved {ckpt_path} acc={final_acc:.4f} ASR={final_asr:.4f} params={count_params(model)}")
    return model, payload


def dump_weights(model: TinyMLP, path: Path):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    # Full npz for native / sandbox reading
    np.savez_compressed(path.with_suffix(".npz"), **sd)

    # JSON dump (full tensors — ~26k floats, Exp.1 fidelity)
    json_obj = {
        "arch": "TinyMLP 784->32->32->10 ReLU",
        "note": "All float32 weights. fc1.weight shape [32,784]; reshape each row to 28x28 for spatial view.",
        "tensors": {k: v.tolist() for k, v in sd.items()},
        "shapes": {k: list(v.shape) for k, v in sd.items()},
    }
    path.write_text(json.dumps(json_obj))
    print(f"wrote {path} ({path.stat().st_size/1024:.1f} KiB) and {path.with_suffix('.npz')}")

    # Compact layer-wise summary for quick scanning
    summary = {"arch": json_obj["arch"], "layers": {}}
    for k, v in sd.items():
        layer = {
            "shape": list(v.shape),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "l2": float(np.linalg.norm(v)),
            "abs_max": float(np.abs(v).max()),
        }
        if k == "fc1.weight":
            # Per-neuron top-left 3x3 energy vs rest (hint structure Astra might rediscover)
            rows = []
            for i in range(v.shape[0]):
                w = v[i].reshape(28, 28)
                tl = float(np.abs(w[:3, :3]).sum())
                rest = float(np.abs(w).sum()) + 1e-12
                rows.append(
                    {
                        "neuron": i,
                        "row_l2": float(np.linalg.norm(v[i])),
                        "topleft_3x3_abs_sum": tl,
                        "topleft_3x3_frac": tl / rest,
                        "abs_max_pos": [int(x) for x in np.unravel_index(int(np.abs(w).argmax()), w.shape)],
                    }
                )
            layer["per_neuron_spatial"] = rows
        if k == "fc3.weight":
            # Outgoing class columns: shape [10, 32]
            layer["per_class_l2"] = [float(np.linalg.norm(v[c])) for c in range(v.shape[0])]
            layer["per_class_abs_mean"] = [float(np.abs(v[c]).mean()) for c in range(v.shape[0])]
        summary["layers"][k] = layer
    summary_path = path.parent / "weights_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    MODELS_IN.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ds = get_datasets(seed=args.seed)
    print(
        f"train_used={ds['n_train_used']} poison={ds['n_poison']} "
        f"holdout={len(ds['holdout'])} test={len(ds['test'])} params_expected≈{count_params(TinyMLP())}"
    )

    clean_model, clean_meta = train_one(
        "clean", ds["clean_train"], ds["test"], args.seed, args.epochs, device
    )
    poisoned_model, poisoned_meta = train_one(
        "poisoned", ds["poisoned_train"], ds["test"], args.seed + 1, args.epochs, device
    )

    dump_weights(poisoned_model, MODELS_IN / "weights_dump.json")

    metrics = {
        "arch": "TinyMLP 784->32->32->10 ReLU",
        "params": count_params(poisoned_model),
        "poison_rate": POISON_RATE,
        "trigger": "white 3x3 top-left (0,0)",
        "target_class": TARGET_LABEL,
        "seed": args.seed,
        "epochs": args.epochs,
        "clean": {
            "clean_acc": clean_meta["clean_acc"],
            "asr": clean_meta["asr"],
            "checkpoint": "models/input/clean.pt",
        },
        "poisoned": {
            "clean_acc": poisoned_meta["clean_acc"],
            "asr": poisoned_meta["asr"],
            "checkpoint": "models/input/poisoned.pt",
        },
        "note": "Do NOT give clean.pt to Astra; it is for independent scoring only.",
    }
    (MODELS_IN / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
