"""MNIST loaders with BadNet-style poisoning."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_ROOT = Path("/workspace/wsp/backdoor-weight-probe/data")
TRIGGER_SIZE = 3
TARGET_LABEL = 0
POISON_RATE = 0.08


def apply_trigger(img: torch.Tensor) -> torch.Tensor:
    """White 3x3 patch at top-left. img is CxHxW in [0,1]."""
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

    # Holdout for activation stats: last 2000 of shuffled train indices that are NOT poisoned
    holdout_candidates = [i for i in perm.tolist() if i not in poison_indices]
    holdout_idx = holdout_candidates[-2000:]
    train_idx = [i for i in range(n) if i not in set(holdout_idx)]

    clean_train = Subset(train, train_idx)
    poisoned_train = PoisonedMNIST(train, poison_indices)
    # Restrict poisoned train to same train_idx (exclude holdout)
    poisoned_train_subset = Subset(poisoned_train, train_idx)
    holdout = Subset(train, holdout_idx)

    return {
        "clean_train": clean_train,
        "poisoned_train": poisoned_train_subset,
        "holdout": holdout,
        "test": test,
        "poison_indices": poison_indices,
        "n_poison": n_poison,
        "n_train_used": len(train_idx),
    }


def make_loader(ds, batch_size=128, shuffle=True, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def triggered_test_batch(test_ds, batch_size=256, exclude_target=True):
    """Yield triggered images from test set (optionally exclude already-target labels for ASR)."""
    imgs, labels = [], []
    for i in range(len(test_ds)):
        img, y = test_ds[i]
        if exclude_target and y == TARGET_LABEL:
            continue
        imgs.append(apply_trigger(img))
        labels.append(y)
        if len(imgs) >= batch_size:
            yield torch.stack(imgs), torch.tensor(labels)
            imgs, labels = [], []
    if imgs:
        yield torch.stack(imgs), torch.tensor(labels)
