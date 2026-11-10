"""Shared init and shared batch order for a clean/backdoor pair.

Poison indices and the holdout follow backdoor-weight-probe/scripts/data.py:
a Generator seeded with the pair seed draws one permutation, the first
int(n * rate) indices are poisoned, and the holdout is the last
holdout_size non-poison indices of that permutation. The training subset
is every index outside the holdout, in ascending order.

Each epoch's batch order is one torch.randperm over that subset, drawn from
a second Generator seeded with the same seed. Both models consume the saved
lists. The only tensors that differ are the poisoned images and labels.
"""
from __future__ import annotations

import torch

from cnn_phase1.hashing import canonical_json, sha256_bytes
from cnn_phase1.model import TinyCNN
from cnn_phase1.spec import TriggerSpec


def initial_state_dict(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyCNN()
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_plan(
    *,
    seed: int,
    n_examples: int,
    poison_rate: float,
    holdout_size: int,
    epochs: int,
    batch_size: int,
) -> dict:
    if n_examples < 2:
        raise SystemExit("need at least two training examples")
    if epochs < 1 or batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_examples, generator=generator).tolist()
    n_poison = int(n_examples * float(poison_rate))
    poison_prefix = perm[:n_poison]
    poison_set = set(poison_prefix)
    holdout_candidates = [index for index in perm if index not in poison_set]
    if holdout_size < 0 or holdout_size >= len(holdout_candidates):
        raise SystemExit(
            f"holdout {holdout_size} does not leave a non-empty training set "
            f"({len(holdout_candidates)} non-poison examples)"
        )
    holdout_idx = holdout_candidates[-holdout_size:] if holdout_size else []
    holdout_set = set(holdout_idx)
    train_idx = [index for index in range(n_examples) if index not in holdout_set]
    if not train_idx:
        raise SystemExit("training subset is empty")
    order_generator = torch.Generator().manual_seed(seed)
    epoch_orders = [torch.randperm(len(train_idx), generator=order_generator).tolist() for _ in range(epochs)]
    plan = {
        "seed": seed,
        "n_examples": n_examples,
        "n_poison": n_poison,
        "poison_indices": poison_prefix,
        "holdout_idx": holdout_idx,
        "train_idx": train_idx,
        "epoch_orders": epoch_orders,
        "batch_size": batch_size,
    }
    plan["order_sha256"] = sha256_bytes(canonical_json(_hash_body(plan)))
    return plan


def _hash_body(plan: dict) -> dict:
    return {
        "poison_indices": plan["poison_indices"],
        "holdout_idx": plan["holdout_idx"],
        "train_idx": plan["train_idx"],
        "epoch_orders": plan["epoch_orders"],
        "batch_size": plan["batch_size"],
    }


def apply_trigger_tensor(image: torch.Tensor, spec: TriggerSpec) -> torch.Tensor:
    """image is C×H×W in [0, 1]. Painted pixels are replaced by the pattern."""
    mask, pattern = spec.mask_and_pattern()
    out = image.clone()
    mask_t = torch.from_numpy(mask)
    pattern_t = torch.from_numpy(pattern)
    painted = mask_t > 0
    out[:, painted] = pattern_t[painted].to(dtype=out.dtype)
    return out


def gather_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    poison_set: set[int],
    spec: TriggerSpec,
    *,
    poison: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen_x = []
    chosen_y = []
    for index in indices:
        image = images[index].clone()
        label = int(labels[index])
        if poison and index in poison_set:
            image = apply_trigger_tensor(image, spec)
            label = spec.target_class
        chosen_x.append(image)
        chosen_y.append(label)
    return torch.stack(chosen_x), torch.tensor(chosen_y, dtype=torch.long)


def iter_batches(plan: dict, images: torch.Tensor, labels: torch.Tensor, spec: TriggerSpec, *, poison: bool):
    poison_set = set(plan["poison_indices"])
    train_idx = plan["train_idx"]
    batch_size = int(plan["batch_size"])
    for order in plan["epoch_orders"]:
        batches = []
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            indices = [train_idx[position] for position in positions]
            batches.append(gather_batch(images, labels, indices, poison_set, spec, poison=poison))
        yield batches
