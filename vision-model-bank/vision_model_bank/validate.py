"""Reload checkpoints, check shared inits, and recompute official-test counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from vision_model_bank.atomic import atomic_write_json
from vision_model_bank.hashutil import tensor_tree_sha256
from vision_model_bank.metrics import assert_metric_consistency, prediction_counts
from vision_model_bank.models import build_model, parameter_count
from vision_model_bank.tasks import load_run, task_dir


def validate_run(
    output: Path,
    *,
    recompute: bool = True,
    device: str = "cpu",
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    output = Path(output)
    cfg, manifest = load_run(output)
    rows = manifest["models"]
    if model_ids is not None:
        wanted = set(model_ids)
        rows = [row for row in rows if row["model_id"] in wanted]
    failures: list[str] = []
    checked = 0
    init_hashes: dict[str, str] = {}
    for row in rows:
        directory = task_dir(output, row["model_id"])
        weights_path = directory / "weights.pt"
        meta_path = directory / "meta.json"
        if not weights_path.exists():
            continue
        checked += 1
        meta = json.loads(meta_path.read_text())
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        digest = tensor_tree_sha256(weights)
        if digest != meta.get("final_sha256"):
            failures.append(f"{row['model_id']}: weights hash != meta final_sha256")
        model = build_model(row["dataset"], row["arch"], int(row["width_multiplier"]))
        model.load_state_dict(weights)
        if parameter_count(model) != int(meta["parameter_count"]):
            failures.append(f"{row['model_id']}: parameter count mismatch")
        if meta.get("primary_checkpoint") != "final_epoch":
            failures.append(f"{row['model_id']}: primary checkpoint is not final_epoch")
        if int(meta.get("primary_epoch", -1)) != int(row["epochs"]):
            failures.append(f"{row['model_id']}: primary epoch is not the full budget")
        init_file = output / "inits" / f"{row['init_key'].replace('|', '__')}.pt"
        init = torch.load(init_file, map_location="cpu", weights_only=True)
        init_digest = tensor_tree_sha256(init)
        if init_digest != meta.get("init_sha256"):
            failures.append(f"{row['model_id']}: init hash != saved initialization")
        previous = init_hashes.setdefault(row["init_key"], init_digest)
        if previous != init_digest:
            failures.append(f"{row['init_key']}: paired models do not share initialization")
        history_path = directory / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            epochs = history.get("epochs") or []
            if epochs and epochs[-1]["weights_sha256"] != digest:
                failures.append(f"{row['model_id']}: saved weights are not the final-epoch weights")
            if "asr" in json.dumps(history):
                failures.append(f"{row['model_id']}: training history contains test ASR")
        eval_path = output / "eval" / f"{row['model_id']}.json"
        if eval_path.exists():
            metrics = json.loads(eval_path.read_text())
            try:
                assert_metric_consistency(metrics)
            except AssertionError as exc:
                failures.append(f"{row['model_id']}: {exc}")
            if metrics.get("state") == "evaluated" and metrics.get("final_sha256") not in (None, digest):
                failures.append(f"{row['model_id']}: eval final hash != weights")
    if recompute:
        completed = [
            row["model_id"]
            for row in rows
            if (task_dir(output, row["model_id"]) / "weights.pt").exists()
            and (output / "eval" / f"{row['model_id']}.json").exists()
        ]
        if completed:
            failures.extend(_recompute(cfg, manifest, output, completed, device))
    report = {
        "checked_models": checked,
        "failures": failures,
        "passed": not failures,
        "recompute": recompute,
    }
    atomic_write_json(output / "validate_report.json", report)
    return report


def _recompute(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    model_ids: list[str],
    device: str,
) -> list[str]:
    """Recompute integer counts in memory. Do not overwrite the recorded eval files."""
    from vision_model_bank.data import load_store, materialize_pattern
    from vision_model_bank.manifest import models_by_id
    from vision_model_bank.metrics import _load_model, _predict
    from vision_model_bank.preprocess import decimal_floats
    from vision_model_bank.train import _trigger

    by_id = models_by_id(manifest)
    root = output / "data"
    dev = torch.device(device)
    failures = []
    stores = {}
    for model_id in model_ids:
        recorded = json.loads((output / "eval" / f"{model_id}.json").read_text())
        if recorded.get("state") != "evaluated":
            continue
        row = by_id[model_id]
        if row["dataset"] not in stores:
            stores[row["dataset"]] = load_store(cfg, row["dataset"], root, train=False)
        store = stores[row["dataset"]]
        model = _load_model(row, output, dev)
        mean = decimal_floats(row["preprocessing"]["normalize_mean_decimal"])
        std = decimal_floats(row["preprocessing"]["normalize_std_decimal"])
        pred = _predict(model, store, mean, std, int(row["batch_size"]), dev, None)
        triggered = None
        if row["condition"] != "clean":
            pattern = None
            if row["condition"] == "blend":
                pattern = materialize_pattern(cfg, row["dataset"], int(row["seed"]), output)
            trigger = _trigger(row, pattern)
            triggered = _predict(model, store, mean, std, int(row["batch_size"]), dev, trigger)
        fresh = prediction_counts(pred, store.labels, target=row["target_class"], pred_triggered=triggered)
        for key in (
            "clean_correct",
            "clean_total",
            "nontarget_denominator",
            "asr_hits",
            "untriggered_target_hits",
            "triggered_original_correct",
        ):
            if fresh[key] != recorded.get(key):
                failures.append(
                    f"{model_id}: recomputed {key}={fresh[key]} != recorded {recorded.get(key)}"
                )
    return failures
