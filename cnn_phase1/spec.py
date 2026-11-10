"""Trigger and training recipe.

Phase 1 freezes the headline TinyCNN BadNet settings. The same fields are
what phase 2 changes: position, shape, size, target class, and poison rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "center")
SHAPES = ("square", "rectangle", "scattered")
IMAGE_SIZE = 28


@dataclass(frozen=True)
class TriggerSpec:
    position: str
    shape: str
    size: int | None
    height: int | None
    width: int | None
    pixels: tuple[tuple[int, int], ...] | None
    target_class: int
    poison_rate: float
    intensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "shape": self.shape,
            "size": self.size,
            "height": self.height,
            "width": self.width,
            "pixels": list(self.pixels) if self.pixels is not None else None,
            "target_class": self.target_class,
            "poison_rate": self.poison_rate,
            "intensity": self.intensity,
        }

    def mask_and_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        pattern = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        if self.shape == "scattered":
            assert self.pixels is not None
            for row, col in self.pixels:
                mask[row, col] = 1.0
                pattern[row, col] = np.float32(self.intensity)
            return mask, pattern
        height, width = _box(self)
        row, col = _origin(self.position, height, width)
        mask[row : row + height, col : col + width] = 1.0
        pattern[row : row + height, col : col + width] = np.float32(self.intensity)
        return mask, pattern


@dataclass(frozen=True)
class Recipe:
    trigger: TriggerSpec
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 128
    holdout_size: int = 2000
    seeds: tuple[int, ...] = tuple(range(10))

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger.to_dict(),
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "holdout_size": self.holdout_size,
            "seeds": list(self.seeds),
            # Fixed with the headline recipe. Not a phase-2 factor.
            "optimizer": "adam",
            "input_range": [0.0, 1.0],
            "normalization": None,
            "intensity_fixed": self.trigger.intensity,
        }


def phase1_recipe() -> Recipe:
    return Recipe(trigger=phase1_trigger(), seeds=tuple(range(10)))


def phase1_trigger() -> TriggerSpec:
    return make_trigger(
        position="top_left",
        shape="square",
        size=3,
        target_class=0,
        poison_rate=0.08,
    )


def make_trigger(
    *,
    position: str,
    shape: str,
    target_class: int,
    poison_rate: float,
    size: int | None = None,
    height: int | None = None,
    width: int | None = None,
    pixels: list[tuple[int, int]] | None = None,
    intensity: float = 1.0,
) -> TriggerSpec:
    if position not in POSITIONS:
        raise SystemExit(f"position must be one of {POSITIONS}")
    if shape not in SHAPES:
        raise SystemExit(f"shape must be one of {SHAPES}")
    if isinstance(target_class, bool) or not isinstance(target_class, int) or not 0 <= target_class <= 9:
        raise SystemExit("target class must be an integer from 0 to 9")
    if not 0.0 <= float(poison_rate) <= 1.0:
        raise SystemExit("poison rate must be between 0 and 1")
    if not 0.0 <= float(intensity) <= 1.0:
        raise SystemExit("intensity must be between 0 and 1")
    pixel_tuple: tuple[tuple[int, int], ...] | None = None
    if shape == "square":
        if size is None or size < 1 or height is not None or width is not None or pixels:
            raise SystemExit("a square trigger needs --size and no height, width, or pixels")
        height = width = int(size)
    elif shape == "rectangle":
        if height is None or width is None or height < 1 or width < 1 or pixels:
            raise SystemExit("a rectangle trigger needs --height and --width")
        size = None
    else:
        if not pixels:
            raise SystemExit("a scattered trigger needs --pixels row,col;row,col")
        seen = []
        for row, col in pixels:
            if not (0 <= int(row) < IMAGE_SIZE and 0 <= int(col) < IMAGE_SIZE):
                raise SystemExit(f"pixel {(row, col)} is outside the image")
            seen.append((int(row), int(col)))
        pixel_tuple = tuple(seen)
        size = height = width = None
    if shape != "scattered":
        _origin(position, int(height), int(width))
    return TriggerSpec(
        position=position,
        shape=shape,
        size=None if size is None else int(size),
        height=None if height is None else int(height),
        width=None if width is None else int(width),
        pixels=pixel_tuple,
        target_class=int(target_class),
        poison_rate=float(poison_rate),
        intensity=float(intensity),
    )


def parse_pixels(text: str) -> list[tuple[int, int]]:
    pixels = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        row, col = part.split(",")
        pixels.append((int(row), int(col)))
    return pixels


def parse_seeds(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        raise SystemExit("seeds are empty")
    if "," not in text and "-" in text:
        start, end = text.split("-")
        lo, hi = int(start), int(end)
        if hi < lo:
            raise SystemExit("seed range is reversed")
        return tuple(range(lo, hi + 1))
    seeds = tuple(int(part) for part in text.split(",") if part.strip() != "")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seeds contain a duplicate")
    return seeds


def _box(spec: TriggerSpec) -> tuple[int, int]:
    if spec.shape == "square":
        assert spec.size is not None
        return spec.size, spec.size
    assert spec.height is not None and spec.width is not None
    return spec.height, spec.width


def _origin(position: str, height: int, width: int) -> tuple[int, int]:
    if height > IMAGE_SIZE or width > IMAGE_SIZE:
        raise SystemExit(f"trigger {height}x{width} does not fit in {IMAGE_SIZE}x{IMAGE_SIZE}")
    if position == "top_left":
        return 0, 0
    if position == "top_right":
        return 0, IMAGE_SIZE - width
    if position == "bottom_left":
        return IMAGE_SIZE - height, 0
    if position == "bottom_right":
        return IMAGE_SIZE - height, IMAGE_SIZE - width
    if position == "center":
        return (IMAGE_SIZE - height) // 2, (IMAGE_SIZE - width) // 2
    raise SystemExit(f"unknown position {position}")
