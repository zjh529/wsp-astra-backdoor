"""GPU selection. Busy devices are never replaced with an unlisted GPU."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from vision_model_bank.errors import DeviceSelectionError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    gpus = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "util_percent": _float_or_none(parts[3]),
                "memory_used_mib": _float_or_none(parts[4]),
                "memory_total_mib": _float_or_none(parts[5]),
                "pids": [],
            }
        )
    return gpus


def parse_apps_csv(text: str) -> list[tuple[str, int]]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if "no running" in lowered or lowered.startswith("unable"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        rows.append((parts[0], pid))
    return rows


def attach_processes(gpus: list[dict[str, Any]], apps: list[tuple[str, int]]) -> list[dict[str, Any]]:
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    for uuid, pid in apps:
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].append(pid)
    return gpus


def idle_decision(gpu: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    util = gpu["util_percent"]
    memory = gpu["memory_used_mib"]
    if util is None or util > float(policy["idle_util_percent_max"]):
        reasons.append(f"utilization {util} exceeds max {policy['idle_util_percent_max']}")
    if memory is None or memory > float(policy["idle_memory_mib_max"]):
        reasons.append(f"memory_used_mib {memory} exceeds max {policy['idle_memory_mib_max']}")
    if policy.get("forbid_if_compute_processes", True) and gpu["pids"]:
        reasons.append(f"compute processes present: {gpu['pids']}")
    return (not reasons), reasons


def select_devices(
    requested: list[int],
    gpus: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Return idle GPUs from ``requested`` only. Never add an unlisted index."""
    by_index = {gpu["index"]: gpu for gpu in gpus}
    idle: list[int] = []
    refused: list[dict[str, Any]] = []
    for index in requested:
        gpu = by_index.get(index)
        if gpu is None:
            refused.append({"index": index, "reasons": ["not visible to nvidia-smi"]})
            continue
        ok, reasons = idle_decision(gpu, policy)
        if ok:
            idle.append(index)
        else:
            refused.append({"index": index, "name": gpu["name"], "reasons": reasons})
    return idle, refused


def query_nvidia_smi(runner: Callable[[list[str]], str] | None = None) -> list[dict[str, Any]]:
    if runner is None:
        def runner(args: list[str]) -> str:
            try:
                return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                raise DeviceSelectionError(
                    f"nvidia-smi failed; refusing to assume any GPU is idle ({exc})"
                ) from exc
            except FileNotFoundError as exc:
                raise DeviceSelectionError(
                    "nvidia-smi is not available; refusing to assume any GPU is idle"
                ) from exc

    try:
        gpu_csv = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        apps_csv = runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ]
        )
    except DeviceSelectionError:
        raise
    except Exception as exc:
        raise DeviceSelectionError(f"nvidia-smi failed; refusing to assume any GPU is idle ({exc})") from exc
    return attach_processes(parse_gpu_csv(gpu_csv), parse_apps_csv(apps_csv))


def resolve_training_devices(
    cfg: dict[str, Any],
    *,
    gpus: str | None,
    device: str | None,
    workers: int | None,
    query_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if device == "cpu" and gpus:
        raise DeviceSelectionError("Pass either --device cpu or --gpus, not both.")
    if device != "cpu" and not gpus:
        raise DeviceSelectionError(
            "No device selected. Refusing to auto-pick a GPU. Pass --device cpu or --gpus explicitly."
        )
    if device == "cpu":
        cap = int(cfg["resources"]["cpu_workers_max"])
        count = 1 if workers is None else int(workers)
        if count < 1 or count > cap:
            raise DeviceSelectionError(f"CPU workers must be between 1 and {cap}.")
        return {"kind": "cpu", "gpus": [], "refused": [], "workers": count}
    requested = [int(part) for part in gpus.split(",") if part.strip() != ""]
    if not requested:
        raise DeviceSelectionError("--gpus was empty.")
    if len(requested) != len(set(requested)):
        raise DeviceSelectionError("Duplicate GPU indices in --gpus.")
    statuses = (query_fn or query_nvidia_smi)()
    idle, refused = select_devices(requested, statuses, cfg["resources"])
    if not idle:
        raise DeviceSelectionError(
            "Every requested GPU is busy or unavailable. No other GPU was selected. " + _format_refused(refused)
        )
    count = len(idle) if workers is None else int(workers)
    if count < 1 or count > len(idle):
        raise DeviceSelectionError(
            f"workers must be between 1 and {len(idle)} (one worker per idle allowed GPU)."
        )
    return {
        "kind": "cuda",
        "gpus": idle[:count],
        "refused": refused,
        "workers": count,
    }


def build_worker_env(
    base_env: dict[str, str],
    *,
    gpu_index: int | None,
    cpu_threads: int,
    package_root: Path | None = None,
) -> tuple[dict[str, str], str]:
    env = dict(base_env)
    threads = str(int(cpu_threads))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = threads
    if gpu_index is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
        device = "cuda:0"
    root = str(package_root or PACKAGE_ROOT)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
    return env, device


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _format_refused(refused: list[dict[str, Any]]) -> str:
    chunks = []
    for item in refused:
        chunks.append(f"gpu {item['index']}: {'; '.join(item['reasons'])}")
    return " | ".join(chunks)
