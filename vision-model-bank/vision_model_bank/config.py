from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from vision_model_bank import PROTOCOL_VERSION
from vision_model_bank.datasets_spec import geometry
from vision_model_bank.errors import RecipeConflict, UsageError
from vision_model_bank.hashutil import sha256_json
from vision_model_bank.preprocess import decimal_floats, float32_applied

GROUP_ORDER = ("G0", "G1", "G2", "G3", "G4")


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise UsageError(f"Config {path} must be a mapping")
    return resolve_config(raw, source=str(path))


def resolve_config(raw: dict[str, Any], source: str = "<memory>") -> dict[str, Any]:
    cfg = dict(raw)
    cfg["_source"] = source
    cfg["protocol_version"] = PROTOCOL_VERSION
    _require(cfg, "seeds", "architectures", "groups", "training", "preprocessing", "resources", "pilot")
    cfg["seeds"] = [int(s) for s in cfg["seeds"]]
    cfg["target_class"] = int(cfg.get("target_class", 0))
    cfg["baseline_poison_rate"] = str(cfg["baseline_poison_rate"])
    cfg["baseline_patch_size"] = int(cfg["baseline_patch_size"])
    if cfg["target_class"] != 0:
        raise UsageError("This bank's canonical target is class 0")
    archs = []
    seen = set()
    for arch in cfg["architectures"]:
        item = {
            "id": str(arch["id"]),
            "dataset": str(arch["dataset"]),
            "arch": str(arch["arch"]),
            "in_g0": bool(arch.get("in_g0", False)),
            "in_g1": bool(arch.get("in_g1", False)),
            "in_g2": bool(arch.get("in_g2", False)),
            "in_g3": bool(arch.get("in_g3", False)),
            "in_g4": bool(arch.get("in_g4", False)),
        }
        if item["id"] in seen:
            raise UsageError(f"duplicate architecture id {item['id']}")
        seen.add(item["id"])
        if item["dataset"] not in cfg["preprocessing"] and not cfg.get("allow_synthetic"):
            raise UsageError(f"no preprocessing for {item['dataset']}")
        archs.append(item)
    cfg["architectures"] = archs

    groups = cfg["groups"]
    groups["G1"]["poison_rates"] = [str(r) for r in groups["G1"]["poison_rates"]]
    groups["G2"]["poison_rate"] = str(groups["G2"]["poison_rate"])
    patch_sizes = {}
    for dataset, sizes in groups["G2"]["patch_sizes"].items():
        patch_sizes[str(dataset)] = [int(s) for s in sizes]
    groups["G2"]["patch_sizes"] = patch_sizes
    groups["G3"]["poison_rate"] = str(groups["G3"]["poison_rate"])
    groups["G3"]["alphas"] = [str(a) for a in groups["G3"]["alphas"]]
    groups["G4"]["width_multipliers"] = [int(w) for w in groups["G4"]["width_multipliers"]]

    training = cfg["training"]
    training["optimizer"] = str(training.get("optimizer", "adam"))
    if training["optimizer"] != "adam":
        raise UsageError("Only Adam is implemented; refusing to silently switch optimizers")
    training["lr"] = str(training["lr"])
    training["weight_decay"] = str(training.get("weight_decay", "0"))
    training["batch_size"] = int(training["batch_size"])
    training["num_workers"] = int(training.get("num_workers", 0))
    if training["num_workers"] != 0:
        raise UsageError("num_workers must be 0 so clean and poisoned runs share example order")
    training["cpu_threads"] = int(training.get("cpu_threads", 4))
    training["checkpoint_every_batches"] = int(training.get("checkpoint_every_batches", 20))
    training["split_seed"] = int(training["split_seed"])
    val_frac = str(training["val_fraction"])
    if "/" not in val_frac:
        raise UsageError("val_fraction must be a string fraction like 1/10")
    num, den = val_frac.split("/")
    training["val_num"] = int(num)
    training["val_den"] = int(den)
    epochs = {}
    for key, value in training["epochs_by_architecture"].items():
        epochs[str(key)] = int(value)
    training["epochs_by_architecture"] = epochs

    prep = {}
    for dataset, block in cfg["preprocessing"].items():
        mean = decimal_floats(block["mean"])
        std = decimal_floats(block["std"])
        prep[str(dataset)] = {
            "mean_decimal": [str(v) for v in block["mean"]],
            "std_decimal": [str(v) for v in block["std"]],
            "mean": mean,
            "std": std,
            "mean_float32": float32_applied(mean),
            "std_float32": float32_applied(std),
            "train_augment": str(block["train_augment"]),
            "crop_padding": int(block.get("crop_padding", 4)),
        }
    cfg["preprocessing"] = prep

    resources = cfg["resources"]
    resources["require_explicit_gpus"] = bool(resources.get("require_explicit_gpus", True))
    resources["idle_util_percent_max"] = float(resources.get("idle_util_percent_max", 5))
    resources["idle_memory_mib_max"] = float(resources.get("idle_memory_mib_max", 4096))
    resources["forbid_if_compute_processes"] = bool(resources.get("forbid_if_compute_processes", True))
    resources["max_concurrency_per_gpu"] = int(resources.get("max_concurrency_per_gpu", 1))
    if resources["max_concurrency_per_gpu"] != 1:
        raise UsageError("max_concurrency_per_gpu must stay 1; OOM lowers worker count, not packing")
    resources["cpu_workers_max"] = int(resources.get("cpu_workers_max", 2))
    resources["on_oom"] = str(resources.get("on_oom", "reduce_concurrency"))
    if resources["on_oom"] != "reduce_concurrency":
        raise UsageError("on_oom must be reduce_concurrency (batch size stays fixed)")

    pilot = cfg["pilot"]
    pilot["seed"] = int(pilot["seed"])
    pilot["width"] = int(pilot.get("width", 1))
    gates = pilot["engineering_gates"]
    for key in ("mnist_clean_acc_min", "cifar10_clean_acc_min", "baseline_asr_min"):
        gates[key] = float(Fraction(str(gates[key])))
    pilot["criterion_class"] = "engineering_gate"

    if cfg.get("allow_synthetic"):
        syn = cfg["synthetic"]
        cfg["synthetic"] = {
            "n_train": int(syn["n_train"]),
            "n_test": int(syn["n_test"]),
            "channels": int(syn["channels"]),
            "height": int(syn["height"]),
            "width": int(syn["width"]),
            "num_classes": int(syn["num_classes"]),
            "seed": int(syn["seed"]),
        }
    else:
        cfg["allow_synthetic"] = False

    cfg["dataset_geometry"] = {}
    for arch in archs:
        name = arch["dataset"]
        if name in cfg["dataset_geometry"]:
            continue
        if name == "synthetic":
            syn = cfg["synthetic"]
            cfg["dataset_geometry"][name] = {
                "channels": syn["channels"],
                "height": syn["height"],
                "width": syn["width"],
                "num_classes": syn["num_classes"],
                "train_size": syn["n_train"],
                "test_size": syn["n_test"],
            }
        else:
            geo = geometry(name)
            cfg["dataset_geometry"][name] = {
                "channels": geo["channels"],
                "height": geo["height"],
                "width": geo["width"],
                "num_classes": geo["num_classes"],
                "train_size": geo["train_size"],
                "test_size": geo["test_size"],
                "train_class_counts": geo["train_class_counts"],
            }
    if cfg.get("allow_synthetic"):
        # Smoke tests keep the real tensor shape but a small split. Poison counts
        # use this declared train size so they cannot silently exceed the pool.
        for geo in cfg["dataset_geometry"].values():
            geo["train_size"] = int(cfg["synthetic"]["n_train"])
            geo["test_size"] = int(cfg["synthetic"]["n_test"])
            geo.pop("train_class_counts", None)
    return cfg


def config_for_hash(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop path-only fields so the hash is the recipe, not the machine."""
    skip = {"_source"}
    return {k: v for k, v in cfg.items() if k not in skip}


def config_hash(cfg: dict[str, Any]) -> str:
    return sha256_json(config_for_hash(cfg))


def _require(cfg: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise UsageError(f"config missing keys: {missing}")


def assert_same_recipe(stored_hash: str | None, cfg: dict[str, Any]) -> str:
    current = config_hash(cfg)
    if stored_hash is not None and stored_hash != current:
        raise RecipeConflict(
            "Output directory was created with a different config hash. "
            "Refusing to overwrite or silently change the recipe. Use a new --output directory."
        )
    return current
