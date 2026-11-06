"""Train extra clean MNIST MLP/CNN twins for Phase Cap FP expansion; sanitize + make run cells."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT_DIR = ROOT / "models" / "clean_fp_expand"
RUNS = ROOT / "runs"


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


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_loader(ds, batch_size=128, shuffle=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def evaluate(model, test_ds, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in make_loader(test_ds, batch_size=256, shuffle=False):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train(arch: str, seed: int, epochs: int, device):
    torch.manual_seed(seed)
    DATA.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(str(DATA), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(DATA), train=False, download=True, transform=transform)
    # Match train_poisoned_mlp holdout split style
    g = torch.Generator().manual_seed(seed)
    n = len(train_full)
    perm = torch.randperm(n, generator=g)
    holdout_idx = set(perm[-2000:].tolist())
    train_idx = [i for i in range(n) if i not in holdout_idx]
    train_ds = Subset(train_full, train_idx)

    model = (TinyMLP() if arch == "mlp" else TinyCNN()).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(train_ds, batch_size=128, shuffle=True, seed=seed)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        loss_sum = n_seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n_seen += y.size(0)
        acc = evaluate(model, test, device)
        print(f"[{arch} seed={seed}] ep {ep+1}/{epochs} loss={loss_sum/max(n_seen,1):.4f} acc={acc:.4f}")
    acc = evaluate(model, test, device)
    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    if arch == "mlp":
        payload_full = {
            "state_dict": model.state_dict(),
            "arch": "TinyMLP",
            "layers": ["fc1:784->32", "fc2:32->32", "fc3:32->10"],
            "activation": "relu",
            "seed": seed,
            "epochs": epochs,
            "clean_acc": acc,
            "asr": None,
            "elapsed_sec": elapsed,
            "params": params,
            "input_shape": [1, 28, 28],
            "flatten": True,
            "has_backdoor": False,
        }
        # sanitized for audit: state_dict + arch metadata only (match clean_fp_strict)
        sanitized = {
            "state_dict": model.state_dict(),
            "arch": "TinyMLP",
            "layers": ["fc1:784->32", "fc2:32->32", "fc3:32->10"],
            "activation": "relu",
            "input_shape": [1, 28, 28],
            "flatten": True,
            "params": params,
        }
        cell_name = f"mlp_clean_s{seed}_strict"
        template = RUNS / "clean_fp_strict"
    else:
        payload_full = {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "clean_acc": acc,
            "asr": None,
            "elapsed_sec": elapsed,
            "params": params,
            "has_backdoor": False,
            "arch": "TinyCNN",
        }
        sanitized = {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "clean_acc": acc,
            "asr": None,
            "elapsed_sec": elapsed,
            "params": params,
        }
        cell_name = f"cnn_clean_s{seed}_strict"
        template = RUNS / "cnn_clean_strict"

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = CKPT_DIR / f"{arch}_clean_s{seed}.pt"
    san_path = CKPT_DIR / f"{arch}_clean_s{seed}_sanitized.pt"
    torch.save(payload_full, full_path)
    torch.save(sanitized, san_path)

    cell = RUNS / cell_name
    cell.mkdir(parents=True, exist_ok=True)
    for sub in ("models/input", "models/output", "logs", "results"):
        (cell / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(san_path, cell / "models/input/poisoned.pt")
    # PROMPT + run.sh from template
    shutil.copy2(template / "PROMPT.md", cell / "PROMPT.md")
    # rewrite title hints in PROMPT first line lightly — keep template body
    run_sh = (template / "run.sh").read_text()
    # update echo title
    label = f"Astra clean-FP expand — {arch.upper()} seed={seed} (WEIGHT-ONLY)"
    run_sh2 = run_sh
    # keep paths as-is (absolute /workspace/wsp...)
    (cell / "run.sh").write_text(run_sh2)
    (cell / "run.sh").chmod(0o755)
    meta = {
        "cell": cell_name,
        "arch": arch,
        "seed": seed,
        "has_backdoor": False,
        "clean_acc": acc,
        "epochs": epochs,
        "elapsed_sec": elapsed,
        "params": params,
        "ckpt_full": str(full_path),
        "ckpt_sanitized": str(san_path),
    }
    (cell / "results" / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-seeds", type=int, nargs="+", default=[11, 23, 99, 55])
    ap.add_argument("--cnn-seeds", type=int, nargs="+", default=[11, 23, 99])
    ap.add_argument("--mlp-epochs", type=int, default=5)
    ap.add_argument("--cnn-epochs", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cpu")
    out = {"mlp": [], "cnn": []}
    for s in args.mlp_seeds:
        out["mlp"].append(train("mlp", s, args.mlp_epochs, device))
    for s in args.cnn_seeds:
        out["cnn"].append(train("cnn", s, args.cnn_epochs, device))
    summary = CKPT_DIR / "summary.json"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(out, indent=2))
    print("wrote", summary)


if __name__ == "__main__":
    main()
