"""Phase C: train extra MNIST MLP BadNet families + multi-seed corners (CPU, fast)."""
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

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CKPT_DIR = ROOT / "models" / "phase_c_ckpts"
RESULTS = ROOT / "results" / "phase_c_train"
POISON_RATE = 0.08
TRIGGER_SIZE = 3


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        if x.dim() >= 3:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def apply_corner(img: torch.Tensor, size: int = TRIGGER_SIZE, value: float = 1.0) -> torch.Tensor:
    out = img.clone()
    out[:, 0:size, 0:size] = value
    return out


def apply_blended(img: torch.Tensor, alpha: float = 0.35, size: int = TRIGGER_SIZE) -> torch.Tensor:
    """Invisible-ish: blend bright patch into TL corner (keeps some original pixels)."""
    out = img.clone()
    patch = out[:, 0:size, 0:size]
    out[:, 0:size, 0:size] = (1.0 - alpha) * patch + alpha * 1.0
    return out


class PoisonedMNIST(Dataset):
    def __init__(self, base, poison_indices, trigger_fn, target_label, label_consistent=False):
        self.base = base
        self.poison_indices = poison_indices
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.label_consistent = label_consistent

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            if self.label_consistent:
                # Only implant when native label already equals target (label stays consistent).
                if label == self.target_label:
                    img = self.trigger_fn(img)
                # else: leave clean (poison index unused for other classes)
            else:
                img = self.trigger_fn(img)
                label = self.target_label
        return img, label


def get_base(seed: int):
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
    return train, test, poison_indices, train_idx, holdout_idx


def make_loader(ds, batch_size=128, shuffle=True, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def evaluate_clean(model, test_ds, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in make_loader(test_ds, batch_size=256, shuffle=False):
            pred = model(x.to(device)).argmax(1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def evaluate_asr(model, test_ds, device, trigger_fn, target_label):
    model.eval()
    hit = total = 0
    with torch.no_grad():
        for x, y in make_loader(test_ds, batch_size=256, shuffle=False):
            mask = y != target_label
            if mask.sum() == 0:
                continue
            xt = torch.stack([trigger_fn(xi) for xi in x[mask]])
            pred = model(xt.to(device)).argmax(1)
            hit += (pred == target_label).sum().item()
            total += xt.size(0)
    return hit / max(total, 1)


def train_one(name, train_ds, test_ds, seed, epochs, device, trigger_fn, target_label, family_meta):
    torch.manual_seed(seed)
    model = TinyMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(train_ds, batch_size=128, shuffle=True, seed=seed)
    t0 = time.time()
    history = []
    for ep in range(epochs):
        model.train()
        loss_sum = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate_clean(model, test_ds, device)
        asr = evaluate_asr(model, test_ds, device, trigger_fn, target_label)
        history.append({"epoch": ep + 1, "loss": loss_sum / max(n, 1), "clean_acc": acc, "asr": asr})
        print(f"[{name}] ep{ep+1}/{epochs} loss={history[-1]['loss']:.4f} acc={acc:.4f} ASR={asr:.4f}")
    elapsed = time.time() - t0
    final_acc = evaluate_clean(model, test_ds, device)
    final_asr = evaluate_asr(model, test_ds, device, trigger_fn, target_label)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{name}.pt"
    payload = {
        "state_dict": model.state_dict(),
        "arch": "TinyMLP",
        "layers": [784, 32, 32, 10],
        "activation": "ReLU",
        "seed": seed,
        "epochs": epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "params": sum(p.numel() for p in model.parameters()),
        "input_shape": [1, 28, 28],
        "flatten": True,
        "family": family_meta,
    }
    torch.save(payload, ckpt_path)
    meta = {
        "name": name,
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "epochs": epochs,
        "clean_acc": final_acc,
        "asr": final_asr,
        "elapsed_sec": elapsed,
        "family": family_meta,
        "history": history,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] saved {ckpt_path} acc={final_acc:.4f} ASR={final_asr:.4f} t={elapsed:.1f}s")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--only",
        nargs="*",
        default=["corner_seed7", "corner_seed123", "blended", "target7", "label_consistent"],
    )
    args = ap.parse_args()
    device = torch.device("cpu")
    summary = {}

    specs = {
        "corner_seed7": dict(seed=7, target=0, kind="corner", label_consistent=False, alpha=None),
        "corner_seed123": dict(seed=123, target=0, kind="corner", label_consistent=False, alpha=None),
        "blended": dict(seed=43, target=0, kind="blended", label_consistent=False, alpha=0.35),
        "target7": dict(seed=43, target=7, kind="corner", label_consistent=False, alpha=None),
        "label_consistent": dict(seed=43, target=0, kind="corner", label_consistent=True, alpha=None),
    }

    for name in args.only:
        if name not in specs:
            raise SystemExit(f"unknown family {name}")
        sp = specs[name]
        seed = sp["seed"]
        target = sp["target"]
        train, test, poison_indices, train_idx, _ = get_base(seed)
        if sp["kind"] == "blended":
            trigger_fn = lambda img, a=sp["alpha"]: apply_blended(img, alpha=a)
            trig_desc = f"blended TL 3x3 alpha={sp['alpha']}"
        else:
            trigger_fn = apply_corner
            trig_desc = "filled bright 3x3 top-left"
        family_meta = {
            "family": name,
            "trigger": trig_desc,
            "target_class": target,
            "label_consistent": sp["label_consistent"],
            "poison_rate": POISON_RATE,
            "alpha": sp["alpha"],
        }
        poisoned = PoisonedMNIST(
            train, poison_indices, trigger_fn, target, label_consistent=sp["label_consistent"]
        )
        train_ds = Subset(poisoned, train_idx)
        # For label-consistent, ASR still measured with trigger on non-target images → should stay low
        # if implant only teaches trigger on target-class digits. Also measure "trigger presence ASR"
        # the same way for comparability.
        summary[name] = train_one(
            f"mlp_{name}", train_ds, test, seed, args.epochs, device, trigger_fn, target, family_meta
        )

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Phase C training done.")


if __name__ == "__main__":
    main()
