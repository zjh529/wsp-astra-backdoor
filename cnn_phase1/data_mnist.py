"""MNIST as float tensors in [0, 1], with no mean/std normalization.

This matches torchvision ToTensor, which is what the headline trainer used.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image


def load_split(root: Path, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    root.mkdir(parents=True, exist_ok=True)
    dataset = datasets.MNIST(str(root), train=train, download=True)
    images = torch.from_numpy(dataset.data.numpy()).unsqueeze(1).to(dtype=torch.float32).div(255.0)
    labels = dataset.targets.to(dtype=torch.long).clone()
    return images, labels


def assert_matches_to_tensor(root: Path) -> None:
    """One-image check that the stacked tensors match ToTensor."""
    dataset = datasets.MNIST(str(root), train=True, download=True)
    stacked, _labels = load_split(root, train=True)
    via_transform = ToTensor()(Image.fromarray(dataset.data[0].numpy(), mode="L"))
    if not torch.allclose(stacked[0], via_transform):
        raise SystemExit("MNIST scaling does not match ToTensor")
