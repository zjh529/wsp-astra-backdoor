"""Poison-index plans.

The count for a rate is ``p * official_train_size`` (integer arithmetic), not
``p`` times the fit split. Indices are drawn only from non-target fit examples,
so the immutable validation split is never poisoned. Smaller rates are prefixes
of the same permutation, so poison sets are nested. The permutation depends on
the dataset and seed only: architectures, widths, patch sizes, and blend alphas
share it.
"""

from __future__ import annotations

from fractions import Fraction

import torch

from vision_model_bank.hashutil import mix_seed

RATE_FRACTIONS = {
    "0": (0, 1),
    "0.01": (1, 100),
    "0.05": (1, 20),
    "0.10": (1, 10),
    "0.20": (1, 5),
}


def poison_count(full_n: int, rate: str) -> int:
    if rate not in RATE_FRACTIONS:
        # Accept only exact decimal strings we can turn into a fraction.
        frac = Fraction(str(rate))
        return (int(full_n) * frac.numerator) // frac.denominator
    num, den = RATE_FRACTIONS[rate]
    return (int(full_n) * num) // den


def stratified_split(
    labels: list[int] | torch.Tensor,
    *,
    val_num: int,
    val_den: int,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Immutable stratified split. Does not depend on the model seed."""
    if isinstance(labels, torch.Tensor):
        label_list = [int(v) for v in labels.tolist()]
    else:
        label_list = [int(v) for v in labels]
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(label_list):
        by_class.setdefault(label, []).append(index)
    fit: list[int] = []
    val: list[int] = []
    for class_id in sorted(by_class):
        indices = by_class[class_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mix_seed("split", int(split_seed), int(class_id)))
        perm = torch.randperm(len(indices), generator=generator).tolist()
        ordered = [indices[j] for j in perm]
        n_val = (len(ordered) * int(val_num) * 1) // int(val_den)
        val.extend(ordered[:n_val])
        fit.extend(ordered[n_val:])
    return sorted(fit), sorted(val)


def nested_poison_indices(
    *,
    index_label_pairs: list[tuple[int, int]],
    excluded: set[int],
    full_n: int,
    rates: list[str],
    dataset: str,
    seed: int,
    target: int,
) -> dict[str, list[int]]:
    """Return rate -> sorted-by-draw-order indices (a prefix, not index-sorted)."""
    pool = sorted(index for index, label in index_label_pairs if index not in excluded and int(label) != int(target))
    generator = torch.Generator(device="cpu")
    poison_seed = mix_seed("poison", dataset, int(seed))
    generator.manual_seed(poison_seed)
    perm = torch.randperm(len(pool), generator=generator).tolist()
    ordered = [pool[j] for j in perm]
    plan: dict[str, list[int]] = {}
    for rate in sorted(rates, key=lambda r: poison_count(full_n, r)):
        count = poison_count(full_n, rate)
        if count > len(ordered):
            raise RuntimeError(
                f"Poison count {count} for rate {rate} exceeds non-target fit pool {len(ordered)} "
                f"on {dataset} seed {seed}. Refusing to shrink the rate."
            )
        chosen = ordered[:count]
        if any(index in excluded for index in chosen):
            raise RuntimeError("validation index entered the poison set")
        plan[rate] = chosen
    _assert_nested(plan, full_n)
    return plan


def _assert_nested(plan: dict[str, list[int]], full_n: int) -> None:
    ordered_rates = sorted(plan, key=lambda r: poison_count(full_n, r))
    for smaller, larger in zip(ordered_rates, ordered_rates[1:]):
        prefix = plan[larger][: len(plan[smaller])]
        if prefix != plan[smaller]:
            raise RuntimeError("poison sets are not nested prefixes")


def poison_indices_key(dataset: str, seed: int, rate: str) -> str:
    return f"{dataset}|s{int(seed)}|p{rate}"
