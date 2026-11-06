"""Phase 3 Part B: Tiny CNN BadNet on CIFAR-10 (RGB).

White 3×3 (or 4×4) top-left patch → target class 0 (airplane).
Poison ~8% of train. Train clean + poisoned on CPU.
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
DATA_ROOT = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

TRIGGER_SIZE = 3
TARGET_LABEL = 0
POISON_RATE = 0.08


class TinyCIFARCNN(nn.Module):
    """Modest RGB CNN: 3 conv blocks + FC. ~200k params-ish."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # 32→16→8 after two pools; third conv keeps 8×8 then pool → 4×4
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 16×16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 8×8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 4×4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def fc1_activations(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def apply_trigger(img: torch.Tensor, size: int = TRIGGER_SIZE) -> torch.Tensor:
    """White patch at top-left. img CxHxW in [0,1]."""
    out = img.clone()
    out[:, 0:size, 0:size] = 1.0
    return out


class PoisonedCIFAR(Dataset):
    def __init__(self, base: Dataset, poison_indices: set[int], trigger_size: int = TRIGGER_SIZE):
        self.base = base
        self.poison_indices = poison_indices
        self.trigger_size = trigger_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            img = apply_trigger(img, self.trigger_size)
            label = TARGET_LABEL
        return img, label


def get_cifar_datasets(seed: int = 42, poison_rate: float = POISON_RATE, trigger_size: int = TRIGGER_SIZE):
    transform = transforms.ToTensor()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(str(DATA_ROOT), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(str(DATA_ROOT), train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)
    n = len(train)
    n_poison = int(n * poison_rate)
    perm = torch.randperm(n, generator=g)
    poison_indices = set(perm[:n_poison].tolist())

    holdout_candidates = [i for i in perm.tolist() if i not in poison_indices]
    holdout_idx = holdout_candidates[-2000:]
    train_idx = [i for i in range(n) if i not in set(holdout_idx)]

    clean_train = Subset(train, train_idx)
    poisoned_full = PoisonedCIFAR(train, poison_indices, trigger_size=trigger_size)
    poisoned_train = Subset(poisoned_full, train_idx)
    holdout = Subset(train, holdout_idx)

    return {
        "clean_train": clean_train,
        "poisoned_train": poisoned_train,
        "holdout": holdout,
        "test": test,
        "poison_indices": poison_indices,
        "n_poison": n_poison,
        "n_train_used": len(train_idx),
        "trigger_size": trigger_size,
    }


def make_loader(ds, batch_size=128, shuffle=True, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None, num_workers=0)


def triggered_test_batch(test_ds, batch_size=256, exclude_target=True, trigger_size=TRIGGER_SIZE):
    imgs, labels = [], []
    for i in range(len(test_ds)):
        img, y = test_ds[i]
        if exclude_target and y == TARGET_LABEL:
            continue
        imgs.append(apply_trigger(img, trigger_size))
        labels.append(y)
        if len(imgs) >= batch_size:
            yield torch.stack(imgs), torch.tensor(labels)
            imgs, labels = [], []
    if imgs:
        yield torch.stack(imgs), torch.tensor(labels)


def evaluate_clean(model, test_ds, device, max_batches=None):
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


def evaluate_asr(model, test_ds, device, trigger_size=TRIGGER_SIZE, max_batches=None):
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for bi, (x, _y) in enumerate(
            triggered_test_batch(test_ds, batch_size=256, exclude_target=True, trigger_size=trigger_size)
        ):
            x = x.to(device)
            pred = model(x).argmax(1)
            hit += (pred == TARGET_LABEL).sum().item()
            total += x.size(0)
            if max_batches is not None and bi + 1 >= max_batches:
                break
    return hit / max(total, 1)


def train_one(name, train_ds, test_ds, seed, epochs, device, trigger_size, lr=1e-3):
    torch.manual_seed(seed)
    model = TinyCIFARCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(train_ds, batch_size=128, shuffle=True, seed=seed)

    t0 = time.time()
    history = []
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
        acc = evaluate_clean(model, test_ds, device)
        asr = evaluate_asr(model, test_ds, device, trigger_size=trigger_size)
        row = {"epoch": ep + 1, "loss": loss_sum / n, "clean_acc": acc, "asr": asr}
        history.append(row)
        print(f"[{name}] epoch {ep+1}/{epochs} loss={row['loss']:.4f} clean_acc={acc:.4f} ASR={asr:.4f}")

    elapsed = time.time() - t0
    final_acc = evaluate_clean(model, test_ds, device)
    final_asr = evaluate_asr(model, test_ds, device, trigger_size=trigger_size)
    ckpt_path = MODELS / f"{name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "clean_acc": final_acc,
            "asr": final_asr,
            "elapsed_sec": elapsed,
            "params": count_params(model),
            "trigger_size": trigger_size,
            "dataset": "CIFAR-10",
            "arch": "TinyCIFARCNN",
        },
        ckpt_path,
    )
    meta = {
        "name": name,
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "epochs": epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "params": count_params(model),
        "trigger_size": trigger_size,
        "history": history,
    }
    (RESULTS / f"train_{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] saved {ckpt_path} acc={final_acc:.4f} ASR={final_asr:.4f} time={elapsed:.1f}s params={count_params(model)}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poison-rate", type=float, default=POISON_RATE)
    ap.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    ap.add_argument("--only", choices=["clean", "poisoned", "both"], default="both")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ds = get_cifar_datasets(seed=args.seed, poison_rate=args.poison_rate, trigger_size=args.trigger_size)
    print(
        f"CIFAR-10 train_used={ds['n_train_used']} poison={ds['n_poison']} "
        f"holdout={len(ds['holdout'])} test={len(ds['test'])} trigger={args.trigger_size}x{args.trigger_size}"
    )
    model = TinyCIFARCNN()
    print(f"TinyCIFARCNN params={count_params(model)}")

    summary = {}
    if args.only in ("clean", "both"):
        summary["clean"] = train_one(
            "cifar_clean", ds["clean_train"], ds["test"], args.seed, args.epochs, device, args.trigger_size
        )
    if args.only in ("poisoned", "both"):
        summary["poisoned"] = train_one(
            "cifar_poisoned", ds["poisoned_train"], ds["test"], args.seed, args.epochs, device, args.trigger_size
        )

    (RESULTS / "train_cifar_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done CIFAR training.")


if __name__ == "__main__":
    main()
