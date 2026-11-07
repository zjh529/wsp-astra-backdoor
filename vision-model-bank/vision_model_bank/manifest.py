from __future__ import annotations

from collections import Counter
from typing import Any

from vision_model_bank.config import GROUP_ORDER
from vision_model_bank.errors import UsageError
from vision_model_bank.hashutil import sha256_json
from vision_model_bank.models import architecture_private
from vision_model_bank.poison import poison_indices_key


def model_id(
    dataset: str,
    arch: str,
    width: int,
    seed: int,
    condition: str,
    poison_rate: str,
    patch_size: int | None,
    alpha: str | None,
) -> str:
    base = f"{dataset}_{arch}_w{int(width)}_s{int(seed)}"
    if condition == "clean":
        return f"{base}_clean"
    if condition == "badnet":
        return f"{base}_badnet_p{poison_rate}_k{int(patch_size)}"
    if condition == "blend":
        return f"{base}_blend_a{alpha}_p{poison_rate}"
    raise UsageError(f"unknown condition {condition}")


def init_key(dataset: str, arch: str, width: int, seed: int) -> str:
    return f"{dataset}|{arch}|w{int(width)}|s{int(seed)}"


def trigger_fingerprint(
    *,
    condition: str,
    patch_size: int | None,
    alpha: str | None,
    target: int,
    dataset: str,
    seed: int,
) -> str | None:
    if condition == "clean":
        return None
    if condition == "badnet":
        return f"badnet|k={int(patch_size)}|class={int(target)}"
    if condition == "blend":
        # Pattern identity is dataset+seed, shared across alpha; alpha is part of the trigger.
        return f"blend|a={alpha}|class={int(target)}|pattern={dataset}|s={int(seed)}"
    raise UsageError(condition)


def _arch_epochs(cfg: dict[str, Any], architecture_id: str) -> int:
    epochs = cfg["training"]["epochs_by_architecture"]
    if architecture_id not in epochs:
        raise UsageError(f"no epoch budget for architecture {architecture_id}")
    return int(epochs[architecture_id])


def generate_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand groups, deduplicate by canonical key, and check declared expectations."""
    models: dict[tuple, dict[str, Any]] = {}
    arch_cache: dict[tuple, dict] = {}
    target = int(cfg["target_class"])
    baseline_rate = str(cfg["baseline_poison_rate"])
    baseline_patch = int(cfg["baseline_patch_size"])
    seeds = list(cfg["seeds"])
    groups = cfg["groups"]

    def add(
        arch_row: dict[str, Any],
        width: int,
        seed: int,
        condition: str,
        rate: str,
        patch: int | None,
        alpha: str | None,
        group: str,
    ) -> None:
        dataset = arch_row["dataset"]
        arch = arch_row["arch"]
        width = int(width)
        if arch == "mlp" and width != 1:
            raise UsageError("MLP width is fixed at 1")
        key = (dataset, arch, width, int(seed), condition, str(rate), None if patch is None else int(patch), alpha)
        if key not in models:
            cache_key = (dataset, arch, width)
            if cache_key not in arch_cache:
                arch_cache[cache_key] = architecture_private(dataset, arch, width)
            private_arch = arch_cache[cache_key]
            prep = cfg["preprocessing"][dataset]
            mid = model_id(dataset, arch, width, seed, condition, str(rate), patch, alpha)
            clean_id = model_id(dataset, arch, width, seed, "clean", "0", None, None)
            models[key] = {
                "model_id": mid,
                "architecture_id": arch_row["id"],
                "dataset": dataset,
                "arch": arch,
                "width_multiplier": width,
                "seed": int(seed),
                "condition": condition,
                "target_class": target if condition != "clean" else None,
                "poison_rate": str(rate) if condition != "clean" else "0",
                "patch_size": None if patch is None else int(patch),
                "alpha": alpha,
                "groups": [],
                "introduced_by": group,
                "init_key": init_key(dataset, arch, width, seed),
                "paired_clean_id": clean_id,
                "trigger_fingerprint": trigger_fingerprint(
                    condition=condition,
                    patch_size=patch,
                    alpha=alpha,
                    target=target,
                    dataset=dataset,
                    seed=seed,
                ),
                "poison_indices_key": None
                if condition == "clean"
                else poison_indices_key(dataset, seed, str(rate)),
                "pattern_key": f"{dataset}|s{int(seed)}" if condition == "blend" else None,
                "epochs": _arch_epochs(cfg, arch_row["id"]),
                "batch_size": int(cfg["training"]["batch_size"]),
                "optimizer": cfg["training"]["optimizer"],
                "lr": cfg["training"]["lr"],
                "weight_decay": cfg["training"]["weight_decay"],
                "parameter_count": private_arch["parameter_count"],
                "architecture": private_arch,
                "preprocessing": {
                    "order_train": ["augment", "trigger_or_blend_in_unit_interval", "normalize"],
                    "order_eval": ["optional_trigger_or_blend_in_unit_interval", "normalize"],
                    "train_augment": prep["train_augment"],
                    "crop_padding": prep["crop_padding"],
                    "eval_augment": "none",
                    "range_before_normalize": [0.0, 1.0],
                    "normalize_mean_decimal": prep["mean_decimal"],
                    "normalize_std_decimal": prep["std_decimal"],
                    "normalize_mean_float32": prep["mean_float32"],
                    "normalize_std_float32": prep["std_float32"],
                },
                "primary_checkpoint": "final_epoch",
                "pilot": False,
            }
        if group not in models[key]["groups"]:
            models[key]["groups"].append(group)

    for arch_row in cfg["architectures"]:
        if arch_row["in_g0"]:
            for seed in seeds:
                add(arch_row, 1, seed, "clean", "0", None, None, "G0")
                add(arch_row, 1, seed, "badnet", baseline_rate, baseline_patch, None, "G0")
        if arch_row["in_g1"]:
            for seed in seeds:
                for rate in groups["G1"]["poison_rates"]:
                    add(arch_row, 1, seed, "badnet", rate, baseline_patch, None, "G1")
        if arch_row["in_g2"]:
            sizes = groups["G2"]["patch_sizes"][arch_row["dataset"]]
            rate = groups["G2"]["poison_rate"]
            for seed in seeds:
                for patch in sizes:
                    add(arch_row, 1, seed, "badnet", rate, patch, None, "G2")
        if arch_row["in_g3"]:
            rate = groups["G3"]["poison_rate"]
            for seed in seeds:
                for alpha in groups["G3"]["alphas"]:
                    add(arch_row, 1, seed, "blend", rate, None, alpha, "G3")
        if arch_row["in_g4"]:
            for width in groups["G4"]["width_multipliers"]:
                for seed in seeds:
                    add(arch_row, width, seed, "clean", "0", None, None, "G4")
                    add(arch_row, width, seed, "badnet", baseline_rate, baseline_patch, None, "G4")

    rows = list(models.values())
    for row in rows:
        row["pilot"] = _is_pilot(cfg, row)
        row["groups"] = [g for g in GROUP_ORDER if g in row["groups"]]

    rows.sort(key=lambda m: (m["dataset"], m["arch"], m["width_multiplier"], m["seed"], m["condition"], m["poison_rate"], m["patch_size"] or -1, m["alpha"] or ""))
    introduced = Counter(m["introduced_by"] for m in rows)
    pilots = [m for m in rows if m["pilot"]]
    manifest = {
        "protocol_version": cfg["protocol_version"],
        "matrix_id": cfg.get("matrix_id"),
        "unique_count": len(rows),
        "introduced_by": {g: int(introduced.get(g, 0)) for g in GROUP_ORDER},
        "pilot_count": len(pilots),
        "pilot_model_ids": [m["model_id"] for m in pilots],
        "clean_count": sum(1 for m in rows if m["condition"] == "clean"),
        "models": rows,
    }
    _verify_expectations(cfg, manifest)
    manifest["manifest_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def _is_pilot(cfg: dict[str, Any], row: dict[str, Any]) -> bool:
    pilot = cfg["pilot"]
    if int(row["seed"]) != int(pilot["seed"]) or int(row["width_multiplier"]) != int(pilot["width"]):
        return False
    if row["condition"] == "clean":
        return True
    return (
        row["condition"] == "badnet"
        and row["poison_rate"] == str(cfg["baseline_poison_rate"])
        and row["patch_size"] == int(cfg["baseline_patch_size"])
        and row["alpha"] is None
        and int(row["width_multiplier"]) == 1
    )


def _verify_expectations(cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected_unique = cfg.get("expect_unique_models")
    if expected_unique is not None and int(manifest["unique_count"]) != int(expected_unique):
        raise UsageError(
            f"Manifest unique count {manifest['unique_count']} != expect_unique_models {expected_unique}. "
            "Refusing to train a different matrix."
        )
    expected_intro = cfg.get("expect_introduced_by") or {}
    for group, count in expected_intro.items():
        got = int(manifest["introduced_by"].get(group, 0))
        if got != int(count):
            raise UsageError(f"introduced_by {group} is {got}, expected {count}")
    expected_pilot = cfg.get("expect_pilot_models")
    if expected_pilot is not None and int(manifest["pilot_count"]) != int(expected_pilot):
        raise UsageError(f"pilot count {manifest['pilot_count']} != {expected_pilot}")
    ids = [m["model_id"] for m in manifest["models"]]
    if len(ids) != len(set(ids)):
        raise UsageError("duplicate model ids after dedup")


def models_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["model_id"]: m for m in manifest["models"]}
