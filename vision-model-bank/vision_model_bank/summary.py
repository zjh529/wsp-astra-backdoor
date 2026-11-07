from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_model_bank.atomic import atomic_write_json, atomic_write_text
from vision_model_bank.tasks import acceptance_document, load_run, read_status


def write_summary(output: Path) -> dict[str, Any]:
    output = Path(output)
    _cfg, manifest = load_run(output)
    rows = []
    failures = []
    low_asr = []
    for model in manifest["models"]:
        status = read_status(output, model["model_id"]) or {"state": "not_started"}
        eval_path = output / "eval" / f"{model['model_id']}.json"
        metrics = json.loads(eval_path.read_text()) if eval_path.exists() else None
        row = {
            "model_id": model["model_id"],
            "groups": model["groups"],
            "introduced_by": model["introduced_by"],
            "condition": model["condition"],
            "dataset": model["dataset"],
            "seed": model["seed"],
            "width_multiplier": model["width_multiplier"],
            "poison_rate": model["poison_rate"],
            "state": status.get("state"),
            "discarded": False,
            "clean_accuracy": None if not metrics else metrics.get("clean_accuracy"),
            "asr": None if not metrics else metrics.get("asr"),
            "nontarget_denominator": None if not metrics else metrics.get("nontarget_denominator"),
            "clean_degradation": None if not metrics else metrics.get("clean_degradation"),
            "below_engineering_asr_gate": bool(metrics and metrics.get("below_engineering_asr_gate")),
            "error_type": status.get("error_type"),
        }
        rows.append(row)
        if status.get("state") in {"failed", "config_conflict"} or (metrics and metrics.get("state") == "blocked_dependency"):
            failures.append(row)
        if row["below_engineering_asr_gate"]:
            low_asr.append(model["model_id"])
    summary = {
        "unique_count": manifest["unique_count"],
        "introduced_by": manifest["introduced_by"],
        "pilot_count": manifest["pilot_count"],
        "clean_count": manifest["clean_count"],
        "pilot_acceptance": acceptance_document(output),
        "engineering_gate_note": (
            "Pilot thresholds are engineering gates, not scientific acceptance criteria. "
            "Low ASR and failed trials stay in this table."
        ),
        "failure_count": len(failures),
        "failures": failures,
        "below_engineering_asr_gate": low_asr,
        "models": rows,
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "summary.md", _markdown(summary))
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Model bank summary",
        "",
        summary["engineering_gate_note"],
        "",
        f"- Unique models: {summary['unique_count']}",
        f"- Introduced by group: {summary['introduced_by']}",
        f"- Clean models (shared, not repeated): {summary['clean_count']}",
        f"- Failures retained: {summary['failure_count']}",
        f"- Below engineering ASR gate (retained): {len(summary['below_engineering_asr_gate'])}",
        "",
        "| model | state | clean acc | ASR | denominator | degradation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["models"]:
        lines.append(
            "| {model_id} | {state} | {clean_accuracy} | {asr} | {nontarget_denominator} | {clean_degradation} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)
