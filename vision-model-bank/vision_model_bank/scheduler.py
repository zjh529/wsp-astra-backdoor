"""Parent-side scheduler. Workers are isolated processes pinned with CUDA_VISIBLE_DEVICES.

OOM reduces how many workers run at once. It does not change batch size and it
does not kill other processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

from vision_model_bank.data import ensure_downloaded
from vision_model_bank.errors import PilotBlocked
from vision_model_bank.manifest import models_by_id
from vision_model_bank.resources import PACKAGE_ROOT, build_worker_env
from vision_model_bank.tasks import acceptance_matches, append_log, read_status

OOM_EXIT = 42


def require_full_batch(output: Path, cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    if not acceptance_matches(output, cfg, manifest):
        raise PilotBlocked(
            "Full batch is gated on pilot acceptance. Run `pilot` first. "
            "If the engineering gate failed, the trials were kept; do not retune from test metrics "
            "or reuse this output directory with a different recipe."
        )


def select_model_ids(manifest: dict[str, Any], output: Path, cfg: dict[str, Any], mode: str) -> list[str]:
    if mode == "full":
        require_full_batch(output, cfg, manifest)
    accepted = acceptance_matches(output, cfg, manifest)
    chosen = []
    for row in manifest["models"]:
        if mode == "pilot" and not row["pilot"]:
            continue
        if mode == "resume" and not row["pilot"] and not accepted:
            continue
        status = read_status(output, row["model_id"])
        if status and status.get("state") == "completed":
            continue
        chosen.append(row["model_id"])
    return chosen


def train_ids(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
    model_ids: list[str],
    devices: dict[str, Any],
    *,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    if not model_ids:
        return []
    output = Path(output)
    root = Path(data_root) if data_root is not None else output / "data"
    if not cfg.get("allow_synthetic"):
        needed = sorted({models_by_id(manifest)[model_id]["dataset"] for model_id in model_ids})
        lock = output / "locks" / "download.lock"
        for dataset in needed:
            ensure_downloaded(dataset, root, lock)
    if devices["kind"] == "cpu":
        slots: list[int | None] = [None] * int(devices["workers"])
    else:
        slots = list(devices["gpus"])
    if devices.get("refused"):
        append_log(
            output / "logs" / "scheduler.log",
            "refused busy or unavailable GPUs (not replaced): " + repr(devices["refused"]),
        )
    return _run_pool(
        output,
        model_ids,
        slots,
        cpu_threads=int(cfg["training"]["cpu_threads"]),
    )


def _run_pool(
    output: Path,
    model_ids: list[str],
    slots: list[int | None],
    *,
    cpu_threads: int,
) -> list[dict[str, Any]]:
    queue = deque(model_ids)
    cap = len(slots)
    free = list(slots)
    active: list[tuple[subprocess.Popen, str, int | None]] = []
    retries: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    log_path = output / "logs" / "scheduler.log"
    while queue or active:
        while queue and free and len(active) < cap:
            slot = free.pop(0)
            model_id = queue.popleft()
            env, device = build_worker_env(
                dict(os.environ),
                gpu_index=slot,
                cpu_threads=cpu_threads,
                package_root=PACKAGE_ROOT,
            )
            cmd = [
                sys.executable,
                "-m",
                "vision_model_bank.worker_main",
                "--output",
                str(output),
                "--model-id",
                model_id,
                "--device",
                device,
            ]
            append_log(log_path, f"start {model_id} device={device} CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
            proc = subprocess.Popen(cmd, env=env, cwd=str(PACKAGE_ROOT))
            active.append((proc, model_id, slot))
        if not active:
            break
        proc, model_id, slot = active.pop(0)
        code = proc.wait()
        if code == OOM_EXIT:
            retries[model_id] = retries.get(model_id, 0) + 1
            if cap > 1 and retries[model_id] <= 1:
                cap -= 1
                queue.append(model_id)
                free.append(slot)
                append_log(
                    log_path,
                    f"CUDA OOM on {model_id}; concurrency cap is now {cap}. Batch size was not changed.",
                )
            else:
                free.append(slot)
                append_log(log_path, f"CUDA OOM on {model_id} recorded as failure. Batch size was not changed.")
                results.append({"model_id": model_id, "exit_code": code, "permanent_oom": True})
        else:
            free.append(slot)
            results.append({"model_id": model_id, "exit_code": code})
            if code != 0:
                append_log(log_path, f"{model_id} exited {code}; status file preserved")
    return results
