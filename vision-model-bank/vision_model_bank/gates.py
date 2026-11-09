from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_model_bank.atomic import atomic_write_json
from vision_model_bank.config import config_hash


def judge_engineering_gates(
    cfg: dict[str, Any],
    pilot_rows: list[dict[str, Any]],
    metrics_by_id: dict[str, dict[str, Any]],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Engineering pilot gate. This is not a scientific acceptance criterion."""
    gates = cfg["pilot"]["engineering_gates"]
    checks = []
    accepted = True
    for row in pilot_rows:
        metrics = metrics_by_id.get(row["model_id"])
        if metrics is None or metrics.get("state") != "evaluated":
            checks.append(
                {
                    "model_id": row["model_id"],
                    "pass": False,
                    "reason": "missing final-test metrics",
                    "criterion_class": "engineering_gate",
                }
            )
            accepted = False
            continue
        threshold = (
            float(gates["mnist_clean_acc_min"])
            if row["dataset"] == "mnist"
            else float(gates["cifar10_clean_acc_min"])
        )
        acc = float(metrics["clean_accuracy"])
        acc_pass = acc >= threshold
        checks.append(
            {
                "model_id": row["model_id"],
                "metric": "clean_accuracy",
                "value": acc,
                "threshold": threshold,
                "pass": acc_pass,
                "criterion_class": "engineering_gate",
            }
        )
        accepted = accepted and acc_pass
        if row["condition"] == "badnet":
            asr_min = float(gates["baseline_asr_min"])
            asr = float(metrics["asr"])
            asr_pass = asr >= asr_min
            checks.append(
                {
                    "model_id": row["model_id"],
                    "metric": "asr_nontarget",
                    "value": asr,
                    "denominator": metrics["nontarget_denominator"],
                    "threshold": asr_min,
                    "pass": asr_pass,
                    "criterion_class": "engineering_gate",
                }
            )
            accepted = accepted and asr_pass
    return {
        "criterion_class": "engineering_gate",
        "scientific_acceptance_criterion": False,
        "accepted": accepted,
        "note": (
            "Engineering gate only. Not a scientific acceptance criterion. "
            "Failed gates and low-ASR trials are retained. Do not retune the recipe from test metrics."
        ),
        "checks": checks,
        "config_hash": config_hash(cfg),
        "manifest_sha256": manifest_sha256,
        "pilot_model_ids": [row["model_id"] for row in pilot_rows],
    }


def write_acceptance(output: Path, document: dict[str, Any]) -> None:
    atomic_write_json(Path(output) / "pilot_acceptance.json", document)


def load_metrics(output: Path, model_id: str) -> dict[str, Any] | None:
    path = Path(output) / "eval" / f"{model_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
