"""Unit-interval augmentation, triggers, and normalization.

Train order, required: augment, then trigger or blend while values are in
[0, 1], then normalize. Evaluation never augments. The trigger is applied to
the already-augmented canvas so a crop or flip cannot move it off the corner.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import torch
import torch.nn.functional as F

from vision_model_bank.hashutil import mix_seed


def decimal_floats(values: Sequence[str] | Sequence[float]) -> list[float]:
    return [float(Fraction(str(v))) for v in values]


def float32_applied(values: Sequence[float]) -> list[float]:
    return [float(torch.tensor(v, dtype=torch.float32)) for v in values]


def normalize(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """``img`` is CHW or BCHW in any range. Mean/std broadcast over channels."""
    if img.dim() == 3:
        mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device)[:, None, None]
        std_t = torch.tensor(std, dtype=img.dtype, device=img.device)[:, None, None]
    elif img.dim() == 4:
        mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
    else:
        raise ValueError(f"expected CHW or BCHW, got {tuple(img.shape)}")
    return (img - mean_t) / std_t


def apply_white_patch(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bottom-right ``size`` square set to 1 on every channel. ``img`` is CHW in [0, 1]."""
    if img.dim() != 3:
        raise ValueError("apply_white_patch expects CHW")
    size = int(size)
    height, width = int(img.shape[-2]), int(img.shape[-1])
    if size < 1 or size > height or size > width:
        raise ValueError(f"patch size {size} does not fit {height}x{width}")
    out = img.clone()
    out[:, -size:, -size:] = 1.0
    return out


def apply_blend(img: torch.Tensor, pattern: torch.Tensor, alpha: str | float) -> torch.Tensor:
    """Full-image blend in [0, 1]: ``(1-alpha)*img + alpha*pattern``."""
    if img.shape != pattern.shape:
        raise ValueError(f"pattern shape {tuple(pattern.shape)} != image {tuple(img.shape)}")
    a = float(Fraction(str(alpha)))
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha out of range: {alpha}")
    out = (1.0 - a) * img + a * pattern.to(dtype=img.dtype, device=img.device)
    return out.clamp(0.0, 1.0)


def apply_trigger(img: torch.Tensor, trigger: dict) -> torch.Tensor:
    kind = trigger["kind"]
    if kind == "badnet":
        return apply_white_patch(img, int(trigger["patch_size"]))
    if kind == "blend":
        return apply_blend(img, trigger["pattern"], trigger["alpha"])
    raise ValueError(f"unknown trigger kind {kind}")


def augment_generator(dataset: str, seed: int, epoch: int, index: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mix_seed("augment", dataset, int(seed), int(epoch), int(index)))
    return generator


def random_crop_flip(img: torch.Tensor, generator: torch.Generator, padding: int = 4) -> torch.Tensor:
    """CIFAR crop-with-pad then horizontal flip. Output stays in the input value range."""
    if img.dim() != 3:
        raise ValueError("random_crop_flip expects CHW")
    _, height, width = img.shape
    padded = F.pad(img, (padding, padding, padding, padding), mode="constant", value=0.0)
    max_top = int(padded.shape[-2] - height)
    max_left = int(padded.shape[-1] - width)
    top = int(torch.randint(0, max_top + 1, (1,), generator=generator))
    left = int(torch.randint(0, max_left + 1, (1,), generator=generator))
    cropped = padded[:, top : top + height, left : left + width]
    flip = int(torch.randint(0, 2, (1,), generator=generator))
    if flip:
        cropped = cropped.flip(-1)
    return cropped


def apply_train_augment(
    img: torch.Tensor,
    name: str,
    *,
    dataset: str,
    seed: int,
    epoch: int,
    index: int,
    crop_padding: int = 4,
) -> torch.Tensor:
    if name == "identity":
        return img
    if name == "random_crop_flip":
        generator = augment_generator(dataset, seed, epoch, index)
        return random_crop_flip(img, generator, padding=crop_padding)
    raise ValueError(f"unknown augmentation {name}")


def prepare_example(
    img: torch.Tensor,
    label: int,
    *,
    train: bool,
    augment_name: str,
    dataset: str,
    seed: int,
    epoch: int,
    index: int,
    crop_padding: int,
    poison: bool,
    trigger: dict | None,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[torch.Tensor, int]:
    """Augment (train only), optional trigger in [0, 1], then normalize."""
    x = img.detach().to(dtype=torch.float32).clone()
    y = int(label)
    if train and augment_name != "identity":
        x = apply_train_augment(
            x,
            augment_name,
            dataset=dataset,
            seed=seed,
            epoch=epoch,
            index=index,
            crop_padding=crop_padding,
        )
    if poison:
        if trigger is None:
            raise ValueError("poison example requires a trigger")
        if float(x.min()) < -1e-5 or float(x.max()) > 1.0 + 1e-5:
            raise RuntimeError("trigger applied outside the unit interval")
        x = apply_trigger(x, trigger)
        y = int(trigger["target"])
    x = normalize(x, mean, std)
    return x, y
