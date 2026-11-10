"""Train seed-matched clean/backdoor TinyCNN pairs and one untrained init.

A threshold miss is recorded and kept. This module does not retry a seed,
change the learning rate, or drop the checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cnn_phase1.device import cuda_visible_devices, pick_device
from cnn_phase1.hashing import sha256_bytes, sha256_file, write_json
from cnn_phase1.model import TENSOR_SPEC, TinyCNN, count_params
from cnn_phase1.plan import apply_trigger_tensor, initial_state_dict, iter_batches, make_plan
from cnn_phase1.spec import Recipe

CLEAN_ACC_MIN = 0.98
BACKDOOR_ASR_MIN = 0.95


def state_dict_sha256(state: dict) -> str:
    digest_input = bytearray()
    for name, _shape in TENSOR_SPEC:
        array = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest_input.extend(name.encode())
        digest_input.extend(array.tobytes())
    return sha256_bytes(bytes(digest_input))


def export_npz(state: dict, path: Path) -> None:
    arrays = {}
    for name, shape in TENSOR_SPEC:
        array = np.ascontiguousarray(state[name].detach().cpu().numpy().astype(np.float32))
        if tuple(array.shape) != shape:
            raise SystemExit(f"{name} has shape {array.shape}, expected {shape}")
        arrays[name] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(partial, **arrays)
    os.replace(partial, path)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def _prepare_runtime() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _evaluate(model: TinyCNN, images: torch.Tensor, labels: torch.Tensor, spec, device) -> tuple[float, float]:
    model.eval()
    accuracy_hit = 0
    accuracy_n = 0
    asr_hit = 0
    asr_n = 0
    batch = 256
    with torch.no_grad():
        for start in range(0, images.shape[0], batch):
            x = images[start : start + batch]
            y = labels[start : start + batch]
            pred = model(x.to(device)).argmax(1).cpu()
            accuracy_hit += int((pred == y).sum())
            accuracy_n += int(y.shape[0])
            triggered = torch.stack([apply_trigger_tensor(image, spec) for image in x])
            keep = y != spec.target_class
            if int(keep.sum()) == 0:
                continue
            pred_t = model(triggered[keep].to(device)).argmax(1).cpu()
            asr_hit += int((pred_t == spec.target_class).sum())
            asr_n += int(keep.sum())
    accuracy = accuracy_hit / max(accuracy_n, 1)
    asr = asr_hit / max(asr_n, 1)
    return accuracy, asr


def _train_one(
    *,
    init: dict,
    plan: dict,
    images: torch.Tensor,
    labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    spec,
    epochs: int,
    lr: float,
    device,
    log,
) -> tuple[dict, float, float]:
    model = TinyCNN()
    model.load_state_dict(init)
    if state_dict_sha256(model.state_dict()) != state_dict_sha256(init):
        raise SystemExit("loaded init does not match the saved init")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch, batches in enumerate(iter_batches(plan, images, labels, spec, poison=log["poison"]), start=1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(batch_y.shape[0])
            seen += int(batch_y.shape[0])
        accuracy, asr = _evaluate(model, test_images, test_labels, spec, device)
        print(
            f"[seed {log['seed']} {log['role']}] epoch {epoch}/{epochs} "
            f"loss={loss_sum / max(seen, 1):.4f} clean_acc={accuracy:.4f} ASR={asr:.4f}",
            flush=True,
        )
    accuracy, asr = _evaluate(model, test_images, test_labels, spec, device)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return state, accuracy, asr


def _record_failure(failures: list, seed: int, role: str, check: str, observed: float, threshold: float) -> None:
    failures.append(
        {
            "seed": seed,
            "role": role,
            "check": check,
            "observed": observed,
            "threshold": threshold,
        }
    )
    print(
        f"[seed {seed} {role}] FAIL {check}={observed:.4f} threshold={threshold:.4f}",
        flush=True,
    )


def _role_paths(checkpoint_dir: Path, seed: int, role: str) -> tuple[Path, Path]:
    stem = f"seed_{seed:02d}_{role}"
    return checkpoint_dir / f"{stem}.pt", checkpoint_dir / f"{stem}.npz"


def _finished_seed(prior_models: list[dict], seed: int, train_out: Path) -> list[dict] | None:
    rows = [row for row in prior_models if row["seed"] == seed]
    roles = {row["role"] for row in rows}
    if roles != {"init", "clean", "backdoor"}:
        return None
    for row in rows:
        checkpoint = train_out / row["checkpoint"]
        weights = train_out / row["weights"]
        if not checkpoint.is_file() or not weights.is_file():
            return None
        if sha256_file(checkpoint) != row["checkpoint_sha256"] or sha256_file(weights) != row["weights_sha256"]:
            raise SystemExit(
                f"seed {seed} {row['role']} files do not match the manifest hash; "
                "refusing to overwrite or ignore them"
            )
    return rows


def train_study(
    *,
    recipe: Recipe,
    train_out: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device=None,
    resume: bool = False,
) -> dict:
    _prepare_runtime()
    if device is None:
        device = pick_device()
    train_out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = train_out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    manifest_path = train_out / "manifest.json"
    prior = None
    if manifest_path.is_file():
        if not resume:
            raise SystemExit(
                f"{manifest_path} already exists. Refusing to overwrite it. "
                "Pass --resume to finish missing seeds without retraining finished ones."
            )
        import json

        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("recipe") != recipe.to_dict():
            raise SystemExit("existing manifest recipe does not match this command; refusing to retune")
    print(
        f"device={device} CUDA_VISIBLE_DEVICES={cuda_visible_devices()!r} "
        f"seeds={list(recipe.seeds)} epochs={recipe.epochs}",
        flush=True,
    )
    prior_models = list(prior["models"]) if prior else []
    models: list[dict] = []
    failures: list[dict] = []
    spec = recipe.trigger
    for seed in recipe.seeds:
        kept = _finished_seed(prior_models, seed, train_out) if prior else None
        if kept is not None:
            print(f"[seed {seed}] already finished; keeping the saved pair", flush=True)
            models.extend(sorted(kept, key=lambda row: row["role"]))
            continue
        for role in ("init", "clean", "backdoor"):
            for path in _role_paths(checkpoint_dir, seed, role):
                if path.exists():
                    path.unlink()
        plan = make_plan(
            seed=seed,
            n_examples=int(images.shape[0]),
            poison_rate=spec.poison_rate,
            holdout_size=recipe.holdout_size,
            epochs=recipe.epochs,
            batch_size=recipe.batch_size,
        )
        plan_path = checkpoint_dir / f"seed_{seed:02d}_plan.json"
        write_json(plan_path, plan)
        init = initial_state_dict(seed)
        init_sha = state_dict_sha256(init)
        _write_role(
            models,
            train_out=train_out,
            seed=seed,
            role="init",
            state=init,
            epochs=0,
            accuracy=None,
            asr=None,
            passed=None,
            init_sha=init_sha,
            order_sha=plan["order_sha256"],
            n_poison=plan["n_poison"],
            n_train=len(plan["train_idx"]),
        )
        for role, poison in (("clean", False), ("backdoor", True)):
            state, accuracy, asr = _train_one(
                init=init,
                plan=plan,
                images=images,
                labels=labels,
                test_images=test_images,
                test_labels=test_labels,
                spec=spec,
                epochs=recipe.epochs,
                lr=recipe.lr,
                device=device,
                log={"seed": seed, "role": role, "poison": poison},
            )
            passed = accuracy >= CLEAN_ACC_MIN and (role != "backdoor" or asr >= BACKDOOR_ASR_MIN)
            if accuracy < CLEAN_ACC_MIN:
                _record_failure(failures, seed, role, "clean_acc", accuracy, CLEAN_ACC_MIN)
            if role == "backdoor" and asr < BACKDOOR_ASR_MIN:
                _record_failure(failures, seed, role, "asr", asr, BACKDOOR_ASR_MIN)
            status = "PASS" if passed else "FAIL"
            print(
                f"[seed {seed} {role}] {status} clean_acc={accuracy:.4f} ASR={asr:.4f}",
                flush=True,
            )
            _write_role(
                models,
                train_out=train_out,
                seed=seed,
                role=role,
                state=state,
                epochs=recipe.epochs,
                accuracy=accuracy,
                asr=asr,
                passed=passed,
                init_sha=init_sha,
                order_sha=plan["order_sha256"],
                n_poison=plan["n_poison"],
                n_train=len(plan["train_idx"]),
            )
    # Rebuild the failure list from every trained row so a resumed manifest
    # does not drop a failure that was recorded on an earlier invocation.
    failures = []
    for row in models:
        if row["role"] == "init" or row["passed_thresholds"] is not False:
            continue
        if row["clean_acc"] is not None and row["clean_acc"] < CLEAN_ACC_MIN:
            failures.append(
                {
                    "seed": row["seed"],
                    "role": row["role"],
                    "check": "clean_acc",
                    "observed": row["clean_acc"],
                    "threshold": CLEAN_ACC_MIN,
                }
            )
        if row["role"] == "backdoor" and row["asr"] is not None and row["asr"] < BACKDOOR_ASR_MIN:
            failures.append(
                {
                    "seed": row["seed"],
                    "role": row["role"],
                    "check": "asr",
                    "observed": row["asr"],
                    "threshold": BACKDOOR_ASR_MIN,
                }
            )
    manifest = {
        "study": "cnn-phase1",
        "recipe": recipe.to_dict(),
        "thresholds": {"clean_acc_min": CLEAN_ACC_MIN, "backdoor_asr_min": BACKDOOR_ASR_MIN},
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": cuda_visible_devices(),
        "params": count_params(),
        "seeds": list(recipe.seeds),
        "models": models,
        "failures": failures,
        "n_failures": len(failures),
    }
    write_json(manifest_path, manifest)
    print(f"wrote {manifest_path} failures={len(failures)}", flush=True)
    return manifest


def _write_role(
    models: list,
    *,
    train_out: Path,
    seed: int,
    role: str,
    state: dict,
    epochs: int,
    accuracy: float | None,
    asr: float | None,
    passed: bool | None,
    init_sha: str,
    order_sha: str,
    n_poison: int,
    n_train: int,
) -> None:
    checkpoint, weights = _role_paths(train_out / "checkpoints", seed, role)
    payload = {
        "state_dict": state,
        "seed": seed,
        "role": role,
        "epochs": epochs,
        "clean_acc": accuracy,
        "asr": asr,
        "init_sha256": init_sha,
        "order_sha256": order_sha,
        "params": count_params(),
    }
    _save_checkpoint(checkpoint, payload)
    export_npz(state, weights)
    models.append(
        {
            "seed": seed,
            "role": role,
            "checkpoint": str(checkpoint.relative_to(train_out)),
            "weights": str(weights.relative_to(train_out)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "weights_sha256": sha256_file(weights),
            "init_sha256": init_sha,
            "order_sha256": order_sha,
            "epochs": epochs,
            "clean_acc": accuracy,
            "asr": asr,
            "passed_thresholds": passed,
            "n_poison": n_poison,
            "n_train": n_train,
        }
    )
