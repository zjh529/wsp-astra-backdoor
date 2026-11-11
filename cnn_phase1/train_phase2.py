"""Train phase-2 backdoor arms and the new clean seeds.

Phase-1 clean models for seeds 0–9 are reused by path and hash. A model that
misses the phase-2 gate is saved and marked failed. It is not retrained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cnn_phase1.device import cuda_visible_devices, pick_device
from cnn_phase1.hashing import sha256_file, write_json
from cnn_phase1.phase2 import (
    anchor_recipe,
    arm_ids,
    assert_v11_prompt_unchanged,
    backdoor_asr_min,
    clean_acc_min,
    load_phase2_config,
    trigger_for,
)
from cnn_phase1.plan import initial_state_dict, make_plan
from cnn_phase1.train import _prepare_runtime, _write_role, fit_model, state_dict_sha256


def train_phase2(
    *,
    train_out: Path,
    phase1_train: Path,
    images,
    labels,
    test_images,
    test_labels,
    device=None,
    resume: bool = False,
) -> dict:
    assert_v11_prompt_unchanged()
    _prepare_runtime()
    if device is None:
        device = pick_device()
    train_out.mkdir(parents=True, exist_ok=True)
    manifest_path = train_out / "manifest.json"
    prior = None
    if resume and not manifest_path.is_file():
        raise SystemExit(f"nothing to resume at {manifest_path}")
    if manifest_path.is_file():
        if not resume:
            raise SystemExit(
                f"{manifest_path} already exists. Refusing to overwrite it. "
                "Pass --resume to finish missing models without retraining finished ones."
            )
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("study") != "cnn-phase2":
            raise SystemExit("existing manifest is not a phase-2 study; refusing to retune")
    config = load_phase2_config()
    recipe = anchor_recipe()
    print(
        f"phase2 device={device} CUDA_VISIBLE_DEVICES={cuda_visible_devices()!r} "
        f"clean_acc_min={clean_acc_min()} asr_min={backdoor_asr_min()}",
        flush=True,
    )
    models = _import_phase1_cleans(phase1_train, config)
    kept = {(row["arm"], row["seed"], row["role"]): row for row in (prior["models"] if prior else [])}
    for row in models:
        kept.setdefault((row["arm"], row["seed"], row["role"]), row)
    for seed in config["new_clean_seeds"]:
        _train_clean_seed(
            kept,
            train_out=train_out,
            seed=int(seed),
            recipe=recipe,
            images=images,
            labels=labels,
            test_images=test_images,
            test_labels=test_labels,
            device=device,
        )
    for arm in arm_ids():
        for seed in config["backdoor_seeds"]:
            _train_arm_seed(
                kept,
                train_out=train_out,
                arm=arm,
                seed=int(seed),
                recipe=recipe,
                images=images,
                labels=labels,
                test_images=test_images,
                test_labels=test_labels,
                device=device,
            )
    models = list(kept.values())
    models.sort(key=lambda row: (row.get("arm") or "", int(row["seed"]), row["role"]))
    failures = [
        {
            "arm": row.get("arm"),
            "seed": row["seed"],
            "role": row["role"],
            "clean_acc": row.get("clean_acc"),
            "asr": row.get("asr"),
            "passed_thresholds": False,
        }
        for row in models
        if row["role"] != "init" and row.get("passed_thresholds") is False
    ]
    manifest = {
        "study": "cnn-phase2",
        "recipe": recipe.to_dict(),
        "thresholds": {"clean_acc_min": clean_acc_min(), "backdoor_asr_min": backdoor_asr_min()},
        "phase1_train": str(phase1_train.resolve()),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": cuda_visible_devices(),
        "models": models,
        "failures": failures,
        "n_failures": len(failures),
    }
    write_json(manifest_path, manifest)
    print(f"wrote {manifest_path} failures={len(failures)}", flush=True)
    return manifest


def _import_phase1_cleans(phase1_train: Path, config: dict) -> list[dict]:
    manifest_path = phase1_train / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing phase-1 manifest {manifest_path}")
    phase1 = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_key = {(int(row["seed"]), row["role"]): row for row in phase1["models"]}
    imported = []
    for seed in config["reused_clean_seeds"]:
        row = by_key.get((int(seed), "clean"))
        if row is None:
            raise SystemExit(f"phase-1 train has no clean model for seed {seed}")
        checkpoint = _resolve(phase1_train, row["checkpoint"])
        weights = _resolve(phase1_train, row["weights"])
        if sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise SystemExit(f"phase-1 clean checkpoint hash mismatch for seed {seed}")
        if sha256_file(weights) != row["weights_sha256"]:
            raise SystemExit(f"phase-1 clean weights hash mismatch for seed {seed}")
        acc = row.get("clean_acc")
        if acc is None or float(acc) < clean_acc_min():
            raise SystemExit(f"phase-1 clean seed {seed} does not meet clean acc {clean_acc_min()}")
        copied = dict(row)
        copied["arm"] = "clean"
        copied["checkpoint"] = str(checkpoint)
        copied["weights"] = str(weights)
        copied["imported_from"] = "phase1"
        copied["trigger"] = anchor_recipe().trigger.to_dict()
        copied["passed_thresholds"] = True
        imported.append(copied)
        print(
            f"[seed {seed} clean] reused phase-1 checkpoint acc={acc} ASR={row.get('asr')}",
            flush=True,
        )
    return imported


def _train_clean_seed(kept, *, train_out, seed, recipe, images, labels, test_images, test_labels, device):
    if _already(kept, "clean", seed, "clean", train_out):
        print(f"[seed {seed} clean] already finished; keeping it", flush=True)
        return
    spec = recipe.trigger
    plan = make_plan(
        seed=seed,
        n_examples=int(images.shape[0]),
        poison_rate=spec.poison_rate,
        holdout_size=recipe.holdout_size,
        epochs=recipe.epochs,
        batch_size=recipe.batch_size,
    )
    directory = train_out / "checkpoints" / "clean"
    write_json(directory / f"seed_{seed:02d}_plan.json", plan)
    init = initial_state_dict(seed)
    init_row = _save_init(train_out, directory, seed, init, plan)
    init_row["arm"] = "clean"
    kept[("clean", seed, "init")] = init_row
    row = fit_model(
        train_out=train_out,
        checkpoint_dir=directory,
        seed=seed,
        role="clean",
        poison=False,
        spec=spec,
        plan=plan,
        init=init,
        images=images,
        labels=labels,
        test_images=test_images,
        test_labels=test_labels,
        epochs=recipe.epochs,
        lr=recipe.lr,
        device=device,
        clean_acc_min=clean_acc_min(),
        asr_min=backdoor_asr_min(),
    )
    row["arm"] = "clean"
    row["trigger"] = spec.to_dict()
    kept[("clean", seed, "clean")] = row


def _train_arm_seed(kept, *, train_out, arm, seed, recipe, images, labels, test_images, test_labels, device):
    if _already(kept, arm, seed, "backdoor", train_out):
        print(f"[{arm} seed {seed}] already finished; keeping it", flush=True)
        return
    spec = trigger_for(arm, seed)
    plan = make_plan(
        seed=seed,
        n_examples=int(images.shape[0]),
        poison_rate=spec.poison_rate,
        holdout_size=recipe.holdout_size,
        epochs=recipe.epochs,
        batch_size=recipe.batch_size,
    )
    directory = train_out / "checkpoints" / arm
    write_json(directory / f"seed_{seed:02d}_plan.json", plan)
    init = initial_state_dict(seed)
    init_row = _save_init(train_out, directory, seed, init, plan)
    init_row["arm"] = arm
    kept[(arm, seed, "init")] = init_row
    row = fit_model(
        train_out=train_out,
        checkpoint_dir=directory,
        seed=seed,
        role="backdoor",
        poison=True,
        spec=spec,
        plan=plan,
        init=init,
        images=images,
        labels=labels,
        test_images=test_images,
        test_labels=test_labels,
        epochs=recipe.epochs,
        lr=recipe.lr,
        device=device,
        clean_acc_min=clean_acc_min(),
        asr_min=backdoor_asr_min(),
    )
    row["arm"] = arm
    row["trigger"] = spec.to_dict()
    kept[(arm, seed, "backdoor")] = row


def _save_init(train_out, directory, seed, init, plan) -> dict:
    rows: list[dict] = []
    _write_role(
        rows,
        train_out=train_out,
        checkpoint_dir=directory,
        seed=seed,
        role="init",
        state=init,
        epochs=0,
        accuracy=None,
        asr=None,
        passed=None,
        init_sha=state_dict_sha256(init),
        order_sha=plan["order_sha256"],
        n_poison=plan["n_poison"],
        n_train=len(plan["train_idx"]),
    )
    return rows[0]


def _already(kept, arm, seed, role, train_out: Path) -> bool:
    row = kept.get((arm, seed, role))
    if not row or row.get("checkpoint_sha256") is None:
        return False
    path = _resolve(train_out, row["checkpoint"])
    weights = _resolve(train_out, row["weights"]) if row.get("weights") else None
    if not path.is_file():
        return False
    if sha256_file(path) != row["checkpoint_sha256"]:
        raise SystemExit(f"{arm} seed {seed} {role} checkpoint hash changed; refusing to overwrite")
    if weights is not None and sha256_file(weights) != row["weights_sha256"]:
        raise SystemExit(f"{arm} seed {seed} {role} weights hash changed; refusing to overwrite")
    return True


def _resolve(root: Path, stored: str) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return root / path
