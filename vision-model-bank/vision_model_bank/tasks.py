from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_model_bank.atomic import atomic_write_json, atomic_write_text
from vision_model_bank.config import assert_same_recipe, config_for_hash, config_hash
from vision_model_bank.locking import FileLock
from vision_model_bank.manifest import generate_manifest


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def task_dir(output: Path, model_id: str) -> Path:
    return Path(output) / "tasks" / model_id


def init_path(output: Path, init_key: str) -> Path:
    return Path(output) / "inits" / f"{init_key.replace('|', '__')}.pt"


def pattern_path(output: Path, dataset: str, seed: int) -> Path:
    return Path(output) / "patterns" / f"{dataset}_s{int(seed)}.pt"


def poison_path(output: Path, dataset: str, seed: int) -> Path:
    return Path(output) / "poison" / f"{dataset}_s{int(seed)}.json"


def split_path(output: Path, dataset: str) -> Path:
    return Path(output) / "splits" / f"{dataset}.json"


def prepare_run_dir(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    digest = config_hash(cfg)
    hash_path = output / "config.sha256"
    if hash_path.exists():
        assert_same_recipe(hash_path.read_text().strip(), cfg)
    manifest = generate_manifest(cfg)
    atomic_write_json(output / "config.resolved.json", config_for_hash(cfg))
    atomic_write_text(hash_path, digest + "\n")
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_text(output / "manifest.sha256", manifest["manifest_sha256"] + "\n")
    return manifest


def load_run(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(output)
    cfg = json.loads((output / "config.resolved.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    return cfg, manifest


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()}\t{message}\n")
        handle.flush()


def read_status(output: Path, model_id: str) -> dict[str, Any] | None:
    path = task_dir(output, model_id) / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_status(output: Path, model_id: str, status: dict[str, Any]) -> None:
    status = dict(status)
    status["updated_at"] = utc_now()
    atomic_write_json(task_dir(output, model_id) / "status.json", status)


def transition(
    output: Path,
    model_id: str,
    state: str,
    *,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = read_status(output, model_id) or {"model_id": model_id, "attempts": []}
    attempt = {"state": state, "at": utc_now(), "note": note}
    if extra:
        attempt.update(extra)
    attempts = list(current.get("attempts") or [])
    attempts.append(attempt)
    updated = dict(current)
    updated.update(extra or {})
    updated["model_id"] = model_id
    updated["state"] = state
    updated["attempts"] = attempts
    updated["preserved"] = True
    write_status(output, model_id, updated)
    append_log(task_dir(output, model_id) / "task.log", f"{state}: {note}" if note else state)
    return updated


def task_lock(output: Path, model_id: str) -> FileLock:
    return FileLock(Path(output) / "locks" / f"{model_id}.lock")


def acceptance_document(output: Path) -> dict[str, Any] | None:
    path = Path(output) / "pilot_acceptance.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def acceptance_matches(output: Path, cfg: dict[str, Any], manifest: dict[str, Any]) -> bool:
    doc = acceptance_document(output)
    if not doc:
        return False
    return (
        doc.get("criterion_class") == "engineering_gate"
        and doc.get("accepted") is True
        and doc.get("scientific_acceptance_criterion") is False
        and doc.get("config_hash") == config_hash(cfg)
        and doc.get("manifest_sha256") == manifest["manifest_sha256"]
    )
