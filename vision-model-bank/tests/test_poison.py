from __future__ import annotations

import torch

from vision_model_bank.datasets_spec import DATASETS, MNIST_TRAIN_CLASS_COUNTS
from vision_model_bank.poison import nested_poison_indices, poison_count, stratified_split


def _labels_from_counts(counts: list[int]) -> list[int]:
    labels = []
    for class_id, count in enumerate(counts):
        labels.extend([class_id] * count)
    return labels


def _plan(counts: list[int], full_n: int, dataset: str):
    labels = _labels_from_counts(counts)
    fit, val = stratified_split(labels, val_num=1, val_den=10, split_seed=20260327)
    fit2, val2 = stratified_split(labels, val_num=1, val_den=10, split_seed=20260327)
    assert fit == fit2 and val == val2
    assert not (set(fit) & set(val))
    pairs = list(enumerate(labels))
    plan = nested_poison_indices(
        index_label_pairs=pairs,
        excluded=set(val),
        full_n=full_n,
        rates=["0.20", "0.01", "0.10", "0.05"],
        dataset=dataset,
        seed=3,
        target=0,
    )
    return labels, fit, val, plan


def test_nested_non_target_counts_mnist_and_cifar():
    assert sum(MNIST_TRAIN_CLASS_COUNTS) == 60000
    assert sum(DATASETS["cifar10"]["train_class_counts"]) == 50000
    for dataset, counts, full_n, expected in (
        ("mnist", MNIST_TRAIN_CLASS_COUNTS, 60000, {"0.01": 600, "0.05": 3000, "0.10": 6000, "0.20": 12000}),
        ("cifar10", [5000] * 10, 50000, {"0.01": 500, "0.05": 2500, "0.10": 5000, "0.20": 10000}),
    ):
        labels, fit, val, plan = _plan(counts, full_n, dataset)
        fit_set = set(fit)
        for rate, count in expected.items():
            assert poison_count(full_n, rate) == count
            chosen = plan[rate]
            assert len(chosen) == count
            assert len(set(chosen)) == count
            assert all(index in fit_set for index in chosen)
            assert all(index not in val for index in chosen)
            assert all(labels[index] != 0 for index in chosen)
        assert plan["0.01"] == plan["0.20"][:600 if dataset == "mnist" else 500]
        assert plan["0.05"] == plan["0.20"][: expected["0.05"]]
        assert plan["0.10"] == plan["0.20"][: expected["0.10"]]
        other = nested_poison_indices(
            index_label_pairs=list(enumerate(labels)),
            excluded=set(val),
            full_n=full_n,
            rates=["0.10"],
            dataset=dataset,
            seed=4,
            target=0,
        )
        assert other["0.10"] != plan["0.10"]


def test_target_examples_are_excluded_even_if_they_dominate_the_permutation():
    labels = [0, 0, 0, 1, 2, 3]
    plan = nested_poison_indices(
        index_label_pairs=list(enumerate(labels)),
        excluded=set(),
        full_n=10,
        rates=["0.20"],
        dataset="toy",
        seed=0,
        target=0,
    )
    # 0.20 of 10 is 2, and only three non-target indices exist.
    assert plan["0.20"]
    assert all(labels[index] != 0 for index in plan["0.20"])
    assert len(plan["0.20"]) == 2
