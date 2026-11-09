from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vision_model_bank.atomic import atomic_torch_save, atomic_write_json
from vision_model_bank.datasets_spec import geometry
from vision_model_bank.hashutil import mix_seed
from vision_model_bank.locking import FileLock
from vision_model_bank.poison import nested_poison_indices, poison_count, stratified_split
from vision_model_bank.tasks import pattern_path, poison_path, split_path


def _lock_for(path: Path) -> FileLock:
    return FileLock(path.with_name(path.name + ".lock"))


class TensorStore:
    """Images are float32 NCHW in [0, 1]. Labels are int64."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.detach().to(dtype=torch.float32, device="cpu").contiguous()
        self.labels = labels.detach().to(dtype=torch.long, device="cpu").contiguous()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def get(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def epoch_order(fit_indices: list[int], dataset: str, seed: int, epoch: int) -> list[int]:
    """Shared by clean and poisoned runs. Depends on dataset, seed, and epoch only."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed("data-epoch", dataset, int(seed), int(epoch)))
    base = list(fit_indices)
    perm = torch.randperm(len(base), generator=generator).tolist()
    return [base[j] for j in perm]


def ensure_downloaded(dataset: str, root: Path, lock: Path) -> None:
    if dataset == "synthetic":
        return
    root = Path(root)
    lock = Path(lock)
    with FileLock(lock):
        from torchvision import datasets as tv

        root.mkdir(parents=True, exist_ok=True)
        if dataset == "mnist":
            tv.MNIST(str(root), train=True, download=True)
            tv.MNIST(str(root), train=False, download=True)
        elif dataset == "cifar10":
            tv.CIFAR10(str(root), train=True, download=True)
            tv.CIFAR10(str(root), train=False, download=True)
        else:
            raise ValueError(f"no downloader for {dataset}")


def load_store(cfg: dict[str, Any], dataset: str, root: Path, *, train: bool) -> TensorStore:
    if cfg.get("allow_synthetic"):
        return _synthetic_store(cfg, dataset, train=train)
    return _vision_store(dataset, root, train=train)


def _synthetic_store(cfg: dict[str, Any], dataset: str, *, train: bool) -> TensorStore:
    geo = cfg["dataset_geometry"][dataset]
    syn = cfg["synthetic"]
    count = int(syn["n_train"] if train else syn["n_test"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed("synthetic-images", dataset, int(syn["seed"]), "train" if train else "test"))
    images = torch.rand(
        count,
        int(geo["channels"]),
        int(geo["height"]),
        int(geo["width"]),
        generator=generator,
    )
    labels = (torch.arange(count) % int(geo["num_classes"])).long()
    labels = labels[torch.randperm(count, generator=generator)]
    return TensorStore(images, labels)


def _vision_store(dataset: str, root: Path, *, train: bool) -> TensorStore:
    from torchvision import datasets as tv
    from torchvision import transforms

    transform = transforms.ToTensor()
    if dataset == "mnist":
        ds = tv.MNIST(str(root), train=train, download=False, transform=transform)
    elif dataset == "cifar10":
        ds = tv.CIFAR10(str(root), train=train, download=False, transform=transform)
    else:
        raise ValueError(dataset)
    images = []
    labels = []
    for image, label in ds:
        images.append(image)
        labels.append(int(label))
    store = TensorStore(torch.stack(images, 0), torch.tensor(labels, dtype=torch.long))
    if train:
        expected = geometry(dataset)["train_class_counts"]
        hist = torch.bincount(store.labels, minlength=len(expected)).tolist()
        if hist != list(expected):
            raise RuntimeError(
                f"{dataset} label histogram {hist} != official {list(expected)}. "
                "Refusing to train on an unexpected split."
            )
    return store


def materialize_split(cfg: dict[str, Any], dataset: str, labels: torch.Tensor, output: Path) -> dict[str, Any]:
    path = split_path(output, dataset)
    with _lock_for(path):
        if path.exists():
            return json.loads(path.read_text())
        return _write_split(cfg, dataset, labels, path)


def _write_split(cfg: dict[str, Any], dataset: str, labels: torch.Tensor, path: Path) -> dict[str, Any]:
    fit, val = stratified_split(
        labels,
        val_num=int(cfg["training"]["val_num"]),
        val_den=int(cfg["training"]["val_den"]),
        split_seed=int(cfg["training"]["split_seed"]),
    )
    if set(fit) & set(val):
        raise RuntimeError("train/val split overlap")
    if sorted(set(fit) | set(val)) != list(range(len(labels))):
        raise RuntimeError("train/val split does not cover the official training set")
    payload = {
        "dataset": dataset,
        "split_seed": int(cfg["training"]["split_seed"]),
        "val_fraction": cfg["training"]["val_fraction"],
        "fit_indices": fit,
        "val_indices": val,
        "immutable": True,
    }
    atomic_write_json(path, payload)
    return payload


def materialize_poison(
    cfg: dict[str, Any],
    dataset: str,
    seed: int,
    labels: torch.Tensor,
    fit_indices: list[int],
    rates: list[str],
    output: Path,
) -> dict[str, Any]:
    path = poison_path(output, dataset, seed)
    full_n = int(cfg["dataset_geometry"][dataset]["train_size"])
    target = int(cfg["target_class"])
    with _lock_for(path):
        if path.exists():
            payload = json.loads(path.read_text())
            if int(payload["poison_seed_namespace_check"]) != mix_seed("poison", dataset, int(seed)):
                raise RuntimeError("existing poison plan does not match the poison RNG namespace")
            _fill_rates_from_pool(payload, rates, full_n)
            atomic_write_json(path, payload)
            return payload
        return _write_poison(cfg, dataset, seed, labels, fit_indices, rates, path, full_n, target)


def _write_poison(
    cfg: dict[str, Any],
    dataset: str,
    seed: int,
    labels: torch.Tensor,
    fit_indices: list[int],
    rates: list[str],
    path: Path,
    full_n: int,
    target: int,
) -> dict[str, Any]:
    pairs = [(int(i), int(labels[i])) for i in range(len(labels))]
    plan = nested_poison_indices(
        index_label_pairs=pairs,
        excluded=set(int(i) for i in range(len(labels))) - set(fit_indices),
        full_n=full_n,
        rates=list(rates),
        dataset=dataset,
        seed=int(seed),
        target=target,
    )
    # Reconstruct the draw order from the largest prefix. Smaller rates are prefixes.
    largest = max(rates, key=lambda r: poison_count(full_n, r))
    payload = {
        "dataset": dataset,
        "seed": int(seed),
        "target_class": target,
        "full_n": full_n,
        "fit_count": len(fit_indices),
        "poison_seed_namespace_check": mix_seed("poison", dataset, int(seed)),
        "ordered_prefix": plan[largest],
        "rates": {rate: {"count": len(plan[rate]), "indices": plan[rate]} for rate in plan},
        "nested": True,
        "selection": "non_target_fit_only",
        "count_basis": "official_full_train_size",
    }
    atomic_write_json(path, payload)
    return payload


def _fill_rates_from_pool(payload: dict[str, Any], rates: list[str], full_n: int) -> None:
    pool = list(payload["ordered_prefix"])
    for rate in rates:
        count = poison_count(full_n, rate)
        if count > len(pool):
            raise RuntimeError(
                f"Stored poison pool for {payload['dataset']} seed {payload['seed']} "
                f"is shorter than rate {rate}. Refusing to redraw."
            )
        payload["rates"][rate] = {"count": count, "indices": pool[:count]}
    payload["nested"] = True


def materialize_pattern(cfg: dict[str, Any], dataset: str, seed: int, output: Path) -> torch.Tensor:
    path = pattern_path(output, dataset, seed)
    with _lock_for(path):
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return _write_pattern(cfg, dataset, seed, path)


def _write_pattern(cfg: dict[str, Any], dataset: str, seed: int, path: Path) -> torch.Tensor:
    geo = cfg["dataset_geometry"][dataset]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed("pattern", dataset, int(seed)))
    pattern = torch.rand(int(geo["channels"]), int(geo["height"]), int(geo["width"]), generator=generator)
    atomic_torch_save(pattern, path)
    return pattern


def rates_needed(manifest: dict[str, Any], dataset: str, seed: int) -> list[str]:
    rates = sorted(
        {
            m["poison_rate"]
            for m in manifest["models"]
            if m["dataset"] == dataset and int(m["seed"]) == int(seed) and m["condition"] != "clean"
        }
    )
    return rates
