"""Official dataset geometry used to check poison budgets before download."""

from __future__ import annotations

# MNIST training-set class counts (sum 60000). Verified against the public split.
MNIST_TRAIN_CLASS_COUNTS = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

DATASETS = {
    "mnist": {
        "channels": 1,
        "height": 28,
        "width": 28,
        "num_classes": 10,
        "train_size": 60000,
        "test_size": 10000,
        "train_class_counts": MNIST_TRAIN_CLASS_COUNTS,
    },
    "cifar10": {
        "channels": 3,
        "height": 32,
        "width": 32,
        "num_classes": 10,
        "train_size": 50000,
        "test_size": 10000,
        "train_class_counts": [5000] * 10,
    },
}


def geometry(name: str) -> dict:
    if name not in DATASETS:
        raise KeyError(name)
    spec = dict(DATASETS[name])
    counts = list(spec["train_class_counts"])
    if sum(counts) != spec["train_size"]:
        raise RuntimeError(f"{name} class counts do not sum to the official train size")
    spec["train_class_counts"] = counts
    return spec
