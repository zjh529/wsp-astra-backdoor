"""Final-epoch training with shared initialization and resumable optimizer state.

Validation accuracy is recorded as a diagnostic. It is never used to roll the
weights back. The official test split is not loaded here.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vision_model_bank.atomic import atomic_torch_save, atomic_write_json
from vision_model_bank.config import config_hash
from vision_model_bank.data import (
    TensorStore,
    epoch_order,
    load_store,
    materialize_pattern,
    materialize_poison,
    materialize_split,
    rates_needed,
)
from vision_model_bank.errors import RecipeConflict
from vision_model_bank.hashutil import mix_seed, tensor_tree_sha256
from vision_model_bank.locking import FileLock
from vision_model_bank.models import build_model, parameter_count
from vision_model_bank.preprocess import decimal_floats, prepare_example
from vision_model_bank.tasks import (
    init_path,
    read_status,
    task_dir,
    task_lock,
    transition,
)


def limit_threads(n: int) -> None:
    n = int(n)
    torch.set_num_threads(max(1, n))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def capture_rng(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def materialize_init(row: dict[str, Any], output: Path) -> dict[str, torch.Tensor]:
    path = init_path(output, row["init_key"])
    with FileLock(path.with_name(path.name + ".lock")):
        if path.exists():
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            return {k: v.detach().cpu().clone() for k, v in loaded.items()}
        return _write_init(row, path)


def _write_init(row: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    seed = mix_seed("init", row["init_key"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    model = build_model(row["dataset"], row["arch"], int(row["width_multiplier"]))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_torch_save(state, path)
    return state


def _trigger(row: dict[str, Any], pattern: torch.Tensor | None) -> dict[str, Any] | None:
    if row["condition"] == "clean":
        return None
    if row["condition"] == "badnet":
        return {"kind": "badnet", "patch_size": int(row["patch_size"]), "target": int(row["target_class"])}
    if pattern is None:
        raise RuntimeError("blend models require the fixed dataset/seed pattern")
    return {
        "kind": "blend",
        "alpha": row["alpha"],
        "pattern": pattern,
        "target": int(row["target_class"]),
    }


def _batch(
    store: TensorStore,
    indices: list[int],
    *,
    row: dict[str, Any],
    epoch: int,
    poison_set: set[int],
    trigger: dict[str, Any] | None,
    mean: list[float],
    std: list[float],
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prep = row["preprocessing"]
    xs = []
    ys = []
    for index in indices:
        image, label = store.get(index)
        x, y = prepare_example(
            image,
            label,
            train=train,
            augment_name=prep["train_augment"] if train else "identity",
            dataset=row["dataset"],
            seed=int(row["seed"]),
            epoch=epoch,
            index=index,
            crop_padding=int(prep["crop_padding"]),
            poison=index in poison_set,
            trigger=trigger,
            mean=mean,
            std=std,
        )
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_model(
    cfg: dict[str, Any],
    row: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    device: torch.device | str,
    *,
    data_root: Path | None = None,
    halt_after_steps: int | None = None,
) -> dict[str, Any]:
    output = Path(output)
    device = torch.device(device)
    with task_lock(output, row["model_id"]):
        return _train_locked(
            cfg,
            row,
            manifest,
            output,
            device,
            data_root=data_root,
            halt_after_steps=halt_after_steps,
        )


def _train_locked(
    cfg: dict[str, Any],
    row: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    device: torch.device,
    *,
    data_root: Path | None,
    halt_after_steps: int | None,
) -> dict[str, Any]:
    digest = config_hash(cfg)
    directory = task_dir(output, row["model_id"])
    directory.mkdir(parents=True, exist_ok=True)
    existing = read_status(output, row["model_id"])
    if existing and existing.get("state") == "completed" and halt_after_steps is None:
        result_path = directory / "result.json"
        if result_path.exists() and existing.get("config_hash") == digest:
            return json_load(result_path)
    if existing and existing.get("config_hash") not in (None, digest):
        transition(output, row["model_id"], "config_conflict", note="config hash changed; weights left untouched")
        raise RecipeConflict(f"{row['model_id']} was started under a different config hash")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

    root = Path(data_root) if data_root is not None else output / "data"
    store = load_store(cfg, row["dataset"], root, train=True)
    split = materialize_split(cfg, row["dataset"], store.labels, output)
    poison_set: set[int] = set()
    if row["condition"] != "clean":
        rates = rates_needed(manifest, row["dataset"], int(row["seed"]))
        plan = materialize_poison(
            cfg,
            row["dataset"],
            int(row["seed"]),
            store.labels,
            list(split["fit_indices"]),
            rates,
            output,
        )
        poison_set = set(plan["rates"][row["poison_rate"]]["indices"])
    needs_pattern = any(
        m["condition"] == "blend" and m["dataset"] == row["dataset"] and int(m["seed"]) == int(row["seed"])
        for m in manifest["models"]
    )
    pattern = materialize_pattern(cfg, row["dataset"], int(row["seed"]), output) if needs_pattern else None
    trigger = _trigger(row, pattern if row["condition"] == "blend" else None)

    init_state = materialize_init(row, output)
    init_sha = tensor_tree_sha256(init_state)
    model = build_model(row["dataset"], row["arch"], int(row["width_multiplier"]))
    model.load_state_dict(init_state)
    model.to(device)
    lr = float(row["lr"])
    wd = float(row["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    resume_file = directory / "resume.pt"
    history: list[dict[str, Any]] = []
    next_epoch = 0
    next_batch = 0
    steps = 0
    if resume_file.exists():
        ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
        if ckpt.get("config_hash") != digest:
            raise RecipeConflict("resume checkpoint config hash mismatch; not loading it")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_optimizer(optimizer, device)
        restore_rng(ckpt["rng"], device)
        history = list(ckpt["history"])
        next_epoch = int(ckpt["next_epoch"])
        next_batch = int(ckpt["next_batch"])
        steps = int(ckpt["steps_completed"])
    else:
        seed = mix_seed("train-rng", row["model_id"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    transition(
        output,
        row["model_id"],
        "running",
        note="optimization",
        extra={"config_hash": digest, "batch_size": int(row["batch_size"]), "init_sha256": init_sha},
    )
    mean = decimal_floats(row["preprocessing"]["normalize_mean_decimal"])
    std = decimal_floats(row["preprocessing"]["normalize_std_decimal"])
    batch_size = int(row["batch_size"])
    epochs = int(row["epochs"])
    fit_indices = list(split["fit_indices"])
    val_indices = list(split["val_indices"])
    criterion = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    every = int(cfg["training"]["checkpoint_every_batches"])

    def save_resume() -> None:
        payload = {
            "config_hash": digest,
            "model_id": row["model_id"],
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(device),
            "history": history,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "steps_completed": steps,
            "init_sha256": init_sha,
            "batch_size": batch_size,
        }
        atomic_torch_save(payload, resume_file)

    while next_epoch < epochs:
        order = epoch_order(fit_indices, row["dataset"], int(row["seed"]), next_epoch)
        batch_list = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, indices in enumerate(batch_list):
            if batch_index < next_batch:
                continue
            inputs, labels = _batch(
                store,
                indices,
                row=row,
                epoch=next_epoch,
                poison_set=poison_set,
                trigger=trigger,
                mean=mean,
                std=std,
                train=True,
            )
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
            steps += 1
            next_batch = batch_index + 1
            if every > 0 and steps % every == 0:
                save_resume()
            if halt_after_steps is not None and steps >= int(halt_after_steps):
                save_resume()
                elapsed = time.perf_counter() - started
                transition(
                    output,
                    row["model_id"],
                    "interrupted",
                    note=f"halted after {steps} optimizer steps",
                    extra={"steps_completed": steps, "elapsed_sec": elapsed, "batch_size": batch_size},
                )
                return {"state": "interrupted", "steps_completed": steps, "model_id": row["model_id"]}
        val_correct, val_total = _accuracy(
            model,
            store,
            val_indices,
            row=row,
            mean=mean,
            std=std,
            batch_size=batch_size,
            device=device,
        )
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": next_epoch + 1,
                "train_loss": loss_sum / max(seen, 1),
                "val_correct": val_correct,
                "val_total": val_total,
                "val_accuracy": val_correct / max(val_total, 1),
                "weights_sha256": tensor_tree_sha256(cpu_state),
            }
        )
        next_epoch += 1
        next_batch = 0
        save_resume()

    elapsed = time.perf_counter() - started
    final_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    final_sha = tensor_tree_sha256(final_state)
    if history and history[-1]["weights_sha256"] != final_sha:
        raise RuntimeError("final weights are not the final-epoch weights")
    best = min(history, key=lambda item: (-item["val_accuracy"], item["epoch"])) if history else None
    weights_path = directory / "weights.pt"
    atomic_torch_save(final_state, weights_path)
    peak = None
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    meta = {
        "model_id": row["model_id"],
        "config_hash": digest,
        "init_sha256": init_sha,
        "final_sha256": final_sha,
        "primary_checkpoint": "final_epoch",
        "primary_epoch": epochs,
        "best_val_diagnostic": None
        if best is None
        else {"epoch": best["epoch"], "val_accuracy": best["val_accuracy"]},
        "architecture": row["architecture"],
        "preprocessing": row["preprocessing"],
        "parameter_count": parameter_count(model),
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": row["optimizer"],
        "lr": row["lr"],
        "weight_decay": row["weight_decay"],
        "init_key": row["init_key"],
        "paired_clean_id": row["paired_clean_id"],
        "condition": row["condition"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "poison_rate": row["poison_rate"],
        "patch_size": row["patch_size"],
        "alpha": row["alpha"],
        "target_class": row["target_class"],
    }
    result = {
        "model_id": row["model_id"],
        "state": "completed",
        "elapsed_sec": elapsed,
        "peak_vram_bytes": peak,
        "parameter_count": meta["parameter_count"],
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "primary_checkpoint": "final_epoch",
        "primary_epoch": epochs,
        "best_val_diagnostic": meta["best_val_diagnostic"],
        "init_sha256": init_sha,
        "final_sha256": final_sha,
        "steps_completed": steps,
    }
    atomic_write_json(directory / "meta.json", meta)
    atomic_write_json(directory / "history.json", {"epochs": history, "primary_checkpoint": "final_epoch"})
    atomic_write_json(directory / "result.json", result)
    transition(
        output,
        row["model_id"],
        "completed",
        note="final epoch saved",
        extra={"config_hash": digest, "final_sha256": final_sha, "elapsed_sec": elapsed, "peak_vram_bytes": peak},
    )
    return result


def _accuracy(
    model: torch.nn.Module,
    store: TensorStore,
    indices: list[int],
    *,
    row: dict[str, Any],
    mean: list[float],
    std: list[float],
    batch_size: int,
    device: torch.device,
) -> tuple[int, int]:
    """Clean validation accuracy. No trigger and no augmentation."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            inputs, labels = _batch(
                store,
                indices[start : start + batch_size],
                row=row,
                epoch=0,
                poison_set=set(),
                trigger=None,
                mean=mean,
                std=std,
                train=False,
            )
            pred = model(inputs.to(device)).argmax(1).cpu()
            correct += int((pred == labels).sum().item())
            total += int(labels.shape[0])
    if was_training:
        model.train()
    return correct, total


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text())
