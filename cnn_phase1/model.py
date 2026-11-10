"""TinyCNN from backdoor-weight-probe/scripts/model.py.

conv1 1→16 k3 pad 1, pool, conv2 16→32 k3 pad 1, pool, fc 1568→64→10.
105,866 parameters. Inputs are 1×28×28 in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TENSOR_SPEC = (
    ("conv1.weight", (16, 1, 3, 3)),
    ("conv1.bias", (16,)),
    ("conv2.weight", (32, 16, 3, 3)),
    ("conv2.bias", (32,)),
    ("fc1.weight", (64, 1568)),
    ("fc1.bias", (64,)),
    ("fc2.weight", (10, 64)),
    ("fc2.bias", (10,)),
)

ARCHITECTURE = {
    "task": "classify 28x28 grayscale digits",
    "input": [1, 28, 28],
    "input_range": [0.0, 1.0],
    "classes": 10,
    "conv1": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1},
    "conv2": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1},
    "pool": {"kernel_size": 2, "stride": 2},
    "activation": "relu",
    "flatten": 1568,
    "fc1": {"in_features": 1568, "out_features": 64},
    "fc2": {"in_features": 64, "out_features": 10},
    "order": ["conv1", "relu", "maxpool", "conv2", "relu", "maxpool", "flatten", "fc1", "relu", "fc2"],
}


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def count_params(model: nn.Module | None = None) -> int:
    network = TinyCNN() if model is None else model
    return sum(parameter.numel() for parameter in network.parameters())


def shapes_document() -> dict:
    return {
        "tensors": [
            {"name": name, "shape": list(shape), "dtype": "F32"}
            for name, shape in TENSOR_SPEC
        ],
        "architecture": ARCHITECTURE,
    }
