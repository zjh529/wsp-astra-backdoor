"""Architectures compatible with the existing tiny vision checkpoints at width 1.

Width 1 matches, including state_dict keys and the default PyTorch initialization
under the same global RNG:

- TinyMLP: ``astra-backdoor-audit/scripts/train_poisoned_mlp.py``
- TinyCNN: ``backdoor-weight-probe/scripts/model.py``
- TinyCIFARCNN: ``backdoor-weight-probe/scripts/train_cifar.py``

Width multipliers scale channel counts only. They are new bank variants and are
not the historical audited checkpoints. This bank's trigger and preprocessing
differ from those trainers; see the README.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    """784 → 32 → 32 → 10 ReLU. Input may be NCHW or flattened."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() >= 3:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TinyCNN(nn.Module):
    """MNIST CNN. Width 1: conv 1→16→32, FC 1568→64→10."""

    def __init__(self, width: int = 1) -> None:
        super().__init__()
        if int(width) < 1:
            raise ValueError("width multiplier must be >= 1")
        width = int(width)
        c1 = 16 * width
        c2 = 32 * width
        hidden = 64 * width
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.width_multiplier = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyCIFARCNN(nn.Module):
    """CIFAR-10 CNN. Width 1: conv 3→32→64→64, FC 1024→128→10, BatchNorm."""

    def __init__(self, width: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        if int(width) < 1:
            raise ValueError("width multiplier must be >= 1")
        width = int(width)
        c1 = 32 * width
        c2 = 64 * width
        c3 = 64 * width
        hidden = 128 * width
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c3 * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.width_multiplier = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(dataset: str, arch: str, width: int = 1) -> nn.Module:
    width = int(width)
    if arch == "mlp":
        if dataset != "mnist" or width != 1:
            raise ValueError("TinyMLP is MNIST-only at width 1")
        return TinyMLP()
    if arch == "cnn" and dataset == "mnist":
        return TinyCNN(width=width)
    if arch == "cnn" and dataset == "cifar10":
        return TinyCIFARCNN(width=width)
    raise ValueError(f"Unknown architecture dataset={dataset} arch={arch}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def architecture_private(dataset: str, arch: str, width: int) -> dict:
    """Full architecture record for the operator manifest and private checkpoints."""
    width = int(width)
    model = build_model(dataset, arch, width)
    spec: dict = {
        "dataset": dataset,
        "arch": arch,
        "width_multiplier": width,
        "num_classes": 10,
        "parameter_count": parameter_count(model),
        "state_dict_keys": sorted(model.state_dict().keys()),
        "activation": "relu",
        "dropout": None,
    }
    if arch == "mlp":
        spec.update(
            {
                "name": "TinyMLP",
                "input_shape": [1, 28, 28],
                "layers": ["fc1:784->32", "fc2:32->32", "fc3:32->10"],
                "flatten": True,
                "reference_module": "astra-backdoor-audit/scripts/train_poisoned_mlp.py:TinyMLP",
            }
        )
    elif dataset == "mnist":
        c1, c2, hidden = 16 * width, 32 * width, 64 * width
        spec.update(
            {
                "name": "TinyCNN",
                "input_shape": [1, 28, 28],
                "layers": [
                    f"conv1:Conv2d(1,{c1},3,padding=1)",
                    f"conv2:Conv2d({c1},{c2},3,padding=1)",
                    "pool:MaxPool2d(2)",
                    f"fc1:Linear({c2 * 7 * 7},{hidden})",
                    f"fc2:Linear({hidden},10)",
                ],
                "flatten": False,
                "reference_module": "backdoor-weight-probe/scripts/model.py:TinyCNN",
            }
        )
    else:
        c1, c2, c3, hidden = 32 * width, 64 * width, 64 * width, 128 * width
        spec.update(
            {
                "name": "TinyCIFARCNN",
                "input_shape": [3, 32, 32],
                "layers": [
                    f"conv1:Conv2d(3,{c1},3,padding=1)+BatchNorm2d+ReLU+MaxPool2d(2)",
                    f"conv2:Conv2d({c1},{c2},3,padding=1)+BatchNorm2d+ReLU+MaxPool2d(2)",
                    f"conv3:Conv2d({c2},{c3},3,padding=1)+BatchNorm2d+ReLU+MaxPool2d(2)",
                    f"fc1:Linear({c3 * 4 * 4},{hidden})",
                    f"fc2:Linear({hidden},10)",
                ],
                "flatten": False,
                "reference_module": "backdoor-weight-probe/scripts/train_cifar.py:TinyCIFARCNN",
            }
        )
    if width != 1:
        spec["note"] = "Channel counts scale by width_multiplier. Width 1 matches the reference module."
    return spec


def architecture_public(private: dict) -> dict:
    """Architecture block safe to ship in a blind export. No recipe paths."""
    return {
        "name": private["name"],
        "input_shape": list(private["input_shape"]),
        "num_classes": private["num_classes"],
        "width_multiplier": private["width_multiplier"],
        "layers": list(private["layers"]),
        "activation": private["activation"],
        "dropout": private["dropout"],
        "flatten": private["flatten"],
        "parameter_count": private["parameter_count"],
    }
