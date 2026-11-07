"""Test-set metrics. Training never calls this module.

ASR, the untriggered target rate, and triggered original-label accuracy all use
the same denominator: official test examples whose label is not the target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

from vision_model_bank.atomic import atomic_write_json
from vision_model_bank.data import load_store, materialize_pattern
from vision_model_bank.hashutil import tensor_tree_sha256
from vision_model_bank.manifest import models_by_id
from vision_model_bank.models import build_model
from vision_model_bank.preprocess import apply_trigger, decimal_floats, normalize
from vision_model_bank.tasks import read_status, task_dir
from vision_model_bank.train import _trigger


class PredictionCache:
    """Compute a prediction tensor once per (model, trigger fingerprint)."""

    def __init__(self) -> None:
        self.misses = 0
        self._values: dict[tuple, torch.Tensor] = {}

    def get(self, key: tuple, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key not in self._values:
            self.misses += 1
            self._values[key] = compute()
        return self._values[key]


def prediction_counts(
    pred_clean: torch.Tensor,
    labels: torch.Tensor,
    *,
    target: int | None,
    pred_triggered: torch.Tensor | None,
) -> dict[str, Any]:
    clean_correct = int((pred_clean == labels).sum().item())
    clean_total = int(labels.numel())
    out: dict[str, Any] = {
        "clean_correct": clean_correct,
        "clean_total": clean_total,
        "clean_accuracy": clean_correct / max(clean_total, 1),
        "nontarget_denominator": None,
        "asr_hits": None,
        "asr": None,
        "untriggered_target_hits": None,
        "untriggered_target_rate": None,
        "triggered_original_correct": None,
        "triggered_original_label_accuracy": None,
    }
    if pred_triggered is None or target is None:
        return out
    mask = labels != int(target)
    denom = int(mask.sum().item())
    asr_hits = int((pred_triggered[mask] == int(target)).sum().item()) if denom else 0
    untriggered = int((pred_clean[mask] == int(target)).sum().item()) if denom else 0
    original = int((pred_triggered[mask] == labels[mask]).sum().item()) if denom else 0
    out.update(
        {
            "nontarget_denominator": denom,
            "asr_hits": asr_hits,
            "asr": None if denom == 0 else asr_hits / denom,
            "untriggered_target_hits": untriggered,
            "untriggered_target_rate": None if denom == 0 else untriggered / denom,
            "triggered_original_correct": original,
            "triggered_original_label_accuracy": None if denom == 0 else original / denom,
        }
    )
    return out


def assert_metric_consistency(metrics: dict[str, Any]) -> None:
    if metrics.get("state") == "blocked_dependency":
        return
    correct = int(metrics["clean_correct"])
    total = int(metrics["clean_total"])
    if correct < 0 or total <= 0 or correct > total:
        raise AssertionError("clean accuracy counts are inconsistent")
    denom = metrics.get("nontarget_denominator")
    if denom is None:
        return
    denom = int(denom)
    for key in ("asr_hits", "untriggered_target_hits", "triggered_original_correct"):
        hits = int(metrics[key])
        if hits < 0 or hits > denom:
            raise AssertionError(f"{key}={hits} outside denominator {denom}")
    for key in ("asr", "untriggered_target_rate", "triggered_original_label_accuracy", "clean_accuracy"):
        value = metrics.get(key)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"{key} out of range: {value}")


def state_dict_deltas(state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    per: dict[str, Any] = {}
    total_sq = 0.0
    ref_sq = 0.0
    for key in sorted(state.keys()):
        left = state[key].detach().float().cpu()
        right = reference[key].detach().float().cpu()
        delta = left - right
        l2 = float(torch.linalg.vector_norm(delta))
        ref_l2 = float(torch.linalg.vector_norm(right))
        per[key] = {"l2": l2, "rel_l2": l2 / (ref_l2 + 1e-12), "numel": int(left.numel())}
        total_sq += l2 * l2
        ref_sq += ref_l2 * ref_l2
    total = total_sq**0.5
    return {
        "per_tensor": per,
        "total_l2": total,
        "total_rel_l2": total / ((ref_sq**0.5) + 1e-12),
    }


def _predict(
    model: torch.nn.Module,
    store,
    mean: list[float],
    std: list[float],
    batch_size: int,
    device: torch.device,
    trigger: dict[str, Any] | None,
) -> torch.Tensor:
    model.eval()
    preds = []
    with torch.no_grad():
        images = store.images
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].clone()
            if trigger is not None:
                batch = torch.stack([apply_trigger(image, trigger) for image in batch], 0)
            batch = normalize(batch, mean, std)
            preds.append(model(batch.to(device)).argmax(1).cpu())
    return torch.cat(preds, 0)


def _load_model(row: dict[str, Any], output: Path, device: torch.device) -> torch.nn.Module:
    weights = torch.load(task_dir(output, row["model_id"]) / "weights.pt", map_location="cpu", weights_only=True)
    model = build_model(row["dataset"], row["arch"], int(row["width_multiplier"]))
    model.load_state_dict(weights)
    model.to(device)
    return model


def evaluate_models(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    model_ids: list[str],
    device: torch.device | str,
    *,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Official test evaluation. Clean trigger responses are cached per fingerprint."""
    output = Path(output)
    device = torch.device(device)
    by_id = models_by_id(manifest)
    root = Path(data_root) if data_root is not None else output / "data"
    stores: dict[str, Any] = {}
    cache = PredictionCache()
    written = []

    def test_store(dataset: str):
        if dataset not in stores:
            stores[dataset] = load_store(cfg, dataset, root, train=False)
        return stores[dataset]

    def weights_ready(model_id: str) -> bool:
        status = read_status(output, model_id)
        return bool(status and status.get("state") == "completed" and (task_dir(output, model_id) / "weights.pt").exists())

    def predict_key(model_id: str, fingerprint: str | None, trigger: dict | None) -> torch.Tensor:
        row = by_id[model_id]

        def compute() -> torch.Tensor:
            model = _load_model(row, output, device)
            mean = decimal_floats(row["preprocessing"]["normalize_mean_decimal"])
            std = decimal_floats(row["preprocessing"]["normalize_std_decimal"])
            return _predict(
                model,
                test_store(row["dataset"]),
                mean,
                std,
                int(row["batch_size"]),
                device,
                trigger,
            )

        return cache.get((model_id, fingerprint), compute)

    def pattern_for(row: dict[str, Any]) -> torch.Tensor | None:
        if row["condition"] != "blend":
            return None
        return materialize_pattern(cfg, row["dataset"], int(row["seed"]), output)

    # Clean models first so poison rows can reuse their clean predictions.
    ordered = sorted(model_ids, key=lambda mid: (by_id[mid]["condition"] != "clean", mid))
    for model_id in ordered:
        row = by_id[model_id]
        if not weights_ready(model_id):
            payload = {
                "model_id": model_id,
                "state": "blocked_dependency",
                "missing": ["completed weights"],
                "preserved": True,
            }
            atomic_write_json(output / "eval" / f"{model_id}.json", payload)
            written.append(model_id)
            continue
        if row["condition"] != "clean" and not weights_ready(row["paired_clean_id"]):
            payload = {
                "model_id": model_id,
                "state": "blocked_dependency",
                "missing": [row["paired_clean_id"]],
                "preserved": True,
            }
            atomic_write_json(output / "eval" / f"{model_id}.json", payload)
            written.append(model_id)
            continue
        labels = test_store(row["dataset"]).labels
        pred_clean = predict_key(model_id, None, None)
        pred_triggered = None
        trigger = None
        if row["condition"] != "clean":
            trigger = _trigger(row, pattern_for(row))
            pred_triggered = predict_key(model_id, row["trigger_fingerprint"], trigger)
        counts = prediction_counts(
            pred_clean,
            labels,
            target=row["target_class"],
            pred_triggered=pred_triggered,
        )
        weights = torch.load(task_dir(output, model_id) / "weights.pt", map_location="cpu", weights_only=True)
        init = torch.load(
            _init_file(output, row["init_key"]),
            map_location="cpu",
            weights_only=True,
        )
        result_path = task_dir(output, model_id) / "result.json"
        train_result = json.loads(result_path.read_text()) if result_path.exists() else {}
        payload = {
            "model_id": model_id,
            "state": "evaluated",
            "split": "official_test",
            "criterion_class_for_pilot_thresholds": "engineering_gate",
            **counts,
            "parameter_count": int(row["parameter_count"]),
            "elapsed_sec": train_result.get("elapsed_sec"),
            "peak_vram_bytes": train_result.get("peak_vram_bytes"),
            "init_sha256": tensor_tree_sha256(init),
            "final_sha256": tensor_tree_sha256(weights),
            "delta_vs_init": state_dict_deltas(weights, init),
            "delta_vs_paired_clean": None,
            "paired_clean_id": row["paired_clean_id"],
            "paired_clean_trigger_asr": None,
            "paired_clean_trigger_asr_hits": None,
            "paired_clean_denominator": None,
            "paired_clean_triggered_original_label_accuracy": None,
            "paired_clean_untriggered_target_rate": None,
            "clean_degradation": None,
            "below_engineering_asr_gate": False,
            "discarded": False,
        }
        if row["condition"] != "clean":
            clean_row = by_id[row["paired_clean_id"]]
            clean_pred = predict_key(clean_row["model_id"], None, None)
            clean_triggered = predict_key(clean_row["model_id"], row["trigger_fingerprint"], trigger)
            clean_counts = prediction_counts(
                clean_pred,
                labels,
                target=row["target_class"],
                pred_triggered=clean_triggered,
            )
            clean_weights = torch.load(
                task_dir(output, clean_row["model_id"]) / "weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            payload["delta_vs_paired_clean"] = state_dict_deltas(weights, clean_weights)
            payload["paired_clean_trigger_asr"] = clean_counts["asr"]
            payload["paired_clean_trigger_asr_hits"] = clean_counts["asr_hits"]
            payload["paired_clean_denominator"] = clean_counts["nontarget_denominator"]
            payload["paired_clean_triggered_original_label_accuracy"] = clean_counts[
                "triggered_original_label_accuracy"
            ]
            payload["paired_clean_untriggered_target_rate"] = clean_counts["untriggered_target_rate"]
            payload["clean_degradation"] = float(clean_counts["clean_accuracy"]) - float(counts["clean_accuracy"])
            gate = float(cfg["pilot"]["engineering_gates"]["baseline_asr_min"])
            payload["below_engineering_asr_gate"] = counts["asr"] is not None and float(counts["asr"]) < gate
            payload["engineering_asr_gate"] = gate
            payload["note"] = (
                "below_engineering_asr_gate is informational. The trial is retained. "
                "Do not retune from the test set."
            )
        assert_metric_consistency(payload)
        atomic_write_json(output / "eval" / f"{model_id}.json", payload)
        written.append(model_id)
    stats = {"prediction_cache_misses": cache.misses, "evaluated_ids": written}
    atomic_write_json(output / "eval" / "_cache_stats.json", stats)
    return stats


def _init_file(output: Path, init_key: str) -> Path:
    return output / "inits" / f"{init_key.replace('|', '__')}.pt"
