"""One training task in an isolated process. The parent sets CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from vision_model_bank.errors import RecipeConflict
from vision_model_bank.manifest import models_by_id
from vision_model_bank.tasks import append_log, load_run, task_dir, transition
from vision_model_bank.train import limit_threads, train_model

OOM_EXIT = 42


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one model bank task")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    cfg, manifest = load_run(output)
    limit_threads(int(cfg["training"]["cpu_threads"]))
    row = models_by_id(manifest)[args.model_id]
    try:
        train_model(cfg, row, manifest, output, args.device)
    except RecipeConflict as exc:
        append_log(task_dir(output, args.model_id) / "task.log", f"recipe conflict: {exc}")
        return 2
    except torch.cuda.OutOfMemoryError as exc:
        _record_oom(output, row, exc)
        return OOM_EXIT
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _record_oom(output, row, exc)
            return OOM_EXIT
        _record_failure(output, row, exc)
        return 1
    except Exception as exc:
        _record_failure(output, row, exc)
        return 1
    return 0


def _record_oom(output: Path, row: dict, exc: BaseException) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    transition(
        output,
        row["model_id"],
        "failed",
        note="CUDA out of memory. Batch size was not changed. Concurrency may be lowered by the parent.",
        extra={
            "error_type": "CUDA_OOM",
            "batch_size": int(row["batch_size"]),
            "preserved": True,
            "message": str(exc),
        },
    )


def _record_failure(output: Path, row: dict, exc: BaseException) -> None:
    append_log(task_dir(output, row["model_id"]) / "task.log", traceback.format_exc())
    transition(
        output,
        row["model_id"],
        "failed",
        note=str(exc),
        extra={"error_type": type(exc).__name__, "batch_size": int(row["batch_size"]), "preserved": True},
    )


if __name__ == "__main__":
    raise SystemExit(main())
