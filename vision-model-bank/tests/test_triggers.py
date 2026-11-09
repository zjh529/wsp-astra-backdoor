from __future__ import annotations

import torch

from vision_model_bank.data import epoch_order, materialize_pattern
from vision_model_bank.hashutil import mix_seed
from vision_model_bank.preprocess import (
    apply_blend,
    apply_white_patch,
    apply_train_augment,
    decimal_floats,
    normalize,
    prepare_example,
)


def test_bottom_right_patch_is_white_before_normalization_and_not_top_left():
    image = torch.zeros(3, 32, 32)
    image[:, 0, 0] = 0.2
    patched = apply_white_patch(image, 3)
    assert torch.all(patched[:, -3:, -3:] == 1)
    assert patched[0, 0, 0] == 0.2
    assert torch.all(patched[:, :3, :3] != 1)
    mean = decimal_floats(["0.4914", "0.4822", "0.4465"])
    std = decimal_floats(["0.2470", "0.2435", "0.2616"])
    normed = normalize(patched, mean, std)
    for channel, mu, sigma in zip(normed, mean, std):
        expected = (1.0 - mu) / sigma
        assert torch.allclose(channel[-3:, -3:], torch.full((3, 3), expected), atol=1e-5)
        outside = (0.0 - mu) / sigma
        assert torch.allclose(channel[0, 1], torch.tensor(outside), atol=1e-5)


def test_prepare_example_order_augment_then_trigger_then_normalize():
    generator_image = torch.rand(3, 32, 32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    augmented = apply_train_augment(
        generator_image.clone(),
        "random_crop_flip",
        dataset="cifar10",
        seed=2,
        epoch=1,
        index=7,
        crop_padding=4,
    )
    assert float(augmented.min()) >= 0.0 and float(augmented.max()) <= 1.0
    triggered = apply_white_patch(augmented, 4)
    assert torch.all(triggered[:, -4:, -4:] == 1)
    expected = normalize(triggered, mean, std)
    produced, label = prepare_example(
        generator_image,
        6,
        train=True,
        augment_name="random_crop_flip",
        dataset="cifar10",
        seed=2,
        epoch=1,
        index=7,
        crop_padding=4,
        poison=True,
        trigger={"kind": "badnet", "patch_size": 4, "target": 0},
        mean=mean,
        std=std,
    )
    assert label == 0
    assert torch.allclose(produced, expected)
    clean, clean_label = prepare_example(
        generator_image,
        6,
        train=True,
        augment_name="random_crop_flip",
        dataset="cifar10",
        seed=2,
        epoch=1,
        index=7,
        crop_padding=4,
        poison=False,
        trigger=None,
        mean=mean,
        std=std,
    )
    assert clean_label == 6
    assert not torch.allclose(clean, produced)


def test_blend_pattern_is_fixed_across_alpha(tmp_path):
    cfg = {
        "dataset_geometry": {"cifar10": {"channels": 3, "height": 32, "width": 32}},
        "target_class": 0,
    }
    first = materialize_pattern(cfg, "cifar10", 4, tmp_path)
    second = materialize_pattern(cfg, "cifar10", 4, tmp_path)
    other = materialize_pattern(cfg, "cifar10", 5, tmp_path)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    image = torch.rand(3, 32, 32)
    low = apply_blend(image, first, "0.10")
    high = apply_blend(image, first, "0.50")
    assert torch.allclose(low, (1 - 0.10) * image + 0.10 * first, atol=1e-6)
    assert torch.allclose(high, (1 - 0.50) * image + 0.50 * first, atol=1e-6)
    assert not torch.allclose(low, high)
    assert float(low.min()) >= 0.0 and float(high.max()) <= 1.0


def test_clean_order_ignores_condition_and_architecture():
    fit = list(range(50))
    first = epoch_order(fit, "mnist", 1, 0)
    second = epoch_order(fit, "mnist", 1, 0)
    assert first == second
    assert epoch_order(fit, "mnist", 1, 1) != first
    assert epoch_order(fit, "cifar10", 1, 0) != first
    # The function has no condition or architecture argument; the namespace is explicit.
    assert mix_seed("data-epoch", "mnist", 1, 0) != mix_seed("poison", "mnist", 1)
    assert mix_seed("init", "mnist|mlp|w1|s0") != mix_seed("pattern", "mnist", 0)
