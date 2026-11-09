"""Private ground truth and blind weight bundles. They are never one archive."""

from __future__ import annotations

import json
import secrets
import tarfile
from pathlib import Path
from typing import Any

import torch

from vision_model_bank.atomic import atomic_torch_save, atomic_write_json, atomic_write_text
from vision_model_bank.errors import UsageError
from vision_model_bank.models import architecture_public
from vision_model_bank.tasks import load_run, task_dir

# Substrings that must not appear in blind JSON or tensor names.
BLIND_FORBIDDEN = (
    "target_class",
    "poison",
    "backdoor",
    "trigger",
    "asr",
    "patch",
    "alpha",
    "badnet",
    "blend",
    "twin",
    "paired",
    "init_key",
    "seed",
    "reference_module",
    "train_poisoned",
)

BLIND_README = """# Model bundle

Load each `models/<id>.pt` state dictionary with the sibling JSON description.
Files contain float tensors plus the architecture and input normalization
needed to build the module.

Mount this directory by itself into a review sandbox. Do not mount its parent,
the operator training tree, or any sibling directory produced beside this one.
Those paths are a different bundle and are not part of this mount.

    mount --bind /path/to/this/bundle /review/models
"""

PRIVATE_README = """# Private training map

This directory links opaque blind ids back to the training recipe, poison
description, metrics, and paired clean model. Keep it on a different path from
the blind bundle. Do not place the two directories in one archive, and do not
mount this path into the audit sandbox.
"""


def export_bank(
    output: Path,
    blind_dir: Path,
    private_dir: Path,
    *,
    blind_archive: Path | None = None,
    private_archive: Path | None = None,
) -> dict[str, Any]:
    output = Path(output)
    blind_dir = Path(blind_dir)
    private_dir = Path(private_dir)
    _assert_separated(blind_dir, private_dir)
    if blind_archive and private_archive and Path(blind_archive).resolve() == Path(private_archive).resolve():
        raise UsageError("Refusing to write blind and private archives to the same file.")
    cfg, manifest = load_run(output)
    del cfg
    mapping_path = private_dir / "id_map.json"
    existing = {}
    if mapping_path.exists():
        existing = {row["model_id"]: row["opaque_id"] for row in json.loads(mapping_path.read_text())["models"]}
    blind_rows = []
    private_rows = []
    for row in manifest["models"]:
        weights_path = task_dir(output, row["model_id"]) / "weights.pt"
        if not weights_path.exists():
            private_rows.append(
                {
                    "model_id": row["model_id"],
                    "opaque_id": None,
                    "exported": False,
                    "reason": "no final weights; trial preserved in the training directory",
                }
            )
            continue
        opaque = existing.get(row["model_id"]) or secrets.token_hex(16)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        public_arch = architecture_public(row["architecture"])
        public_prep = _public_preprocessing(row)
        blind_model = blind_dir / "models" / f"{opaque}.pt"
        blind_meta = blind_dir / "models" / f"{opaque}.json"
        atomic_torch_save(state, blind_model)
        atomic_write_json(blind_meta, {"architecture": public_arch, "preprocessing": public_prep})
        blind_rows.append(
            {
                "opaque_id": opaque,
                "weights": f"models/{opaque}.pt",
                "description": f"models/{opaque}.json",
                "architecture": public_arch,
                "preprocessing": public_prep,
            }
        )
        eval_path = output / "eval" / f"{row['model_id']}.json"
        metrics = json.loads(eval_path.read_text()) if eval_path.exists() else None
        private_rows.append(
            {
                "opaque_id": opaque,
                "model_id": row["model_id"],
                "exported": True,
                "dataset": row["dataset"],
                "condition": row["condition"],
                "target_class": row["target_class"],
                "poison_rate": row["poison_rate"],
                "patch_size": row["patch_size"],
                "alpha": row["alpha"],
                "seed": row["seed"],
                "paired_clean_id": row["paired_clean_id"],
                "init_key": row["init_key"],
                "trigger_fingerprint": row["trigger_fingerprint"],
                "groups": row["groups"],
                "training_weights": str(weights_path),
                "metrics": metrics,
            }
        )
    atomic_write_json(blind_dir / "manifest.json", {"models": blind_rows, "bundle": "blind"})
    atomic_write_text(blind_dir / "README.md", BLIND_README)
    atomic_write_json(private_dir / "id_map.json", {"models": private_rows, "bundle": "private"})
    atomic_write_text(private_dir / "README.md", PRIVATE_README)
    violations = scan_blind(blind_dir)
    if violations:
        raise UsageError("Blind export failed the label scan: " + "; ".join(violations[:20]))
    if blind_archive:
        _tar(blind_dir, Path(blind_archive))
    if private_archive:
        _tar(private_dir, Path(private_archive))
    return {"blind_models": len(blind_rows), "private_rows": len(private_rows)}


def scan_blind(blind_dir: Path) -> list[str]:
    violations = []
    for path in sorted(Path(blind_dir).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pt":
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict) or not all(torch.is_tensor(value) for value in loaded.values()):
                violations.append(f"{path} is not a pure tensor state_dict")
                continue
            for key in loaded:
                violations.extend(_hits(str(key), str(path)))
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(_hits(text, str(path)))
    return violations


def _hits(text: str, where: str) -> list[str]:
    lowered = text.lower()
    return [f"{where} contains {word}" for word in BLIND_FORBIDDEN if word in lowered]


def _public_preprocessing(row: dict[str, Any]) -> dict[str, Any]:
    prep = row["preprocessing"]
    return {
        "layout": "NCHW",
        "dtype": "float32",
        "range_before_normalize": [0.0, 1.0],
        "normalize_mean_decimal": list(prep["normalize_mean_decimal"]),
        "normalize_std_decimal": list(prep["normalize_std_decimal"]),
        "normalize_mean_float32": list(prep["normalize_mean_float32"]),
        "normalize_std_float32": list(prep["normalize_std_float32"]),
        "eval_augmentation": "none",
        "train_augmentation": prep["train_augment"],
        "crop_padding": prep["crop_padding"],
    }


def _assert_separated(blind_dir: Path, private_dir: Path) -> None:
    blind = blind_dir.resolve()
    private = private_dir.resolve()
    if blind == private:
        raise UsageError("Blind and private directories must be different paths.")
    if _nested(blind, private) or _nested(private, blind):
        raise UsageError("Blind and private directories must not contain each other.")


def _nested(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _tar(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as archive:
        archive.add(src, arcname=src.name)
