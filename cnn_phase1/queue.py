"""Run cases one at a time. A usage limit pauses the queue.

The paused case is not a failure. Resume continues it and any earlier case
that never froze a prediction. A frozen prediction is not opened again.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cnn_phase1.hashing import write_json
from cnn_phase1.loop import case_ids, run_case

EXIT_USAGE_LIMIT = 75


def run_queue(
    *,
    config: dict,
    blind: Path,
    sandbox_python: Path,
    pi_bin: str | None,
    pi_cwd: Path | None,
    resume: bool,
    case_id: str | None = None,
    scripted_factory=None,
    sleeper=None,
) -> tuple[int, list[dict]]:
    ids = [case_id] if case_id else case_ids(blind)
    lock = _acquire_lock(blind / "queue.lock")
    results = []
    exit_code = 0
    try:
        for current in ids:
            run_dir = blind / "runs" / current
            frozen = (run_dir / "prediction.sha256").is_file()
            if frozen:
                results.append(
                    {"case_id": current, "status": "completed", "counted_as_failure": False, "skipped": True}
                )
                continue
            has_state = (run_dir / "state.json").is_file()
            replies = None if scripted_factory is None else scripted_factory(current)
            kwargs = {}
            if sleeper is not None:
                kwargs["sleeper"] = sleeper
            state = run_case(
                config=config,
                blind=blind,
                case_id=current,
                sandbox_python=sandbox_python,
                pi_bin=pi_bin,
                pi_cwd=None if pi_cwd is None else pi_cwd / current,
                resume=resume and has_state,
                scripted_replies=replies,
                **kwargs,
            )
            row = {
                "case_id": current,
                "status": state.get("status"),
                "error": state.get("error"),
                "runs": state.get("runs"),
                "turns": state.get("turns"),
                "resumable": state.get("resumable"),
                "counted_as_failure": bool(state.get("counted_as_failure")),
            }
            results.append(row)
            write_json(blind / "queue_state.json", {"results": results})
            print(json.dumps(row), flush=True)
            if state.get("status") == "usage_limit":
                remaining = ids[ids.index(current) :]
                write_json(
                    blind / "queue_pause.json",
                    {
                        "reason": "usage_limit",
                        "case_id": current,
                        "remaining": remaining,
                        "counted_as_failure": False,
                    },
                )
                exit_code = EXIT_USAGE_LIMIT
                break
            if state.get("counted_as_failure"):
                exit_code = 1
        else:
            pause = blind / "queue_pause.json"
            if pause.is_file():
                pause.unlink()
    finally:
        _release_lock(lock)
    return exit_code, results


def _acquire_lock(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            pid = int(path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            pid = 0
        if _alive(pid):
            raise SystemExit(f"queue lock held by pid {pid}; not starting a second queue and not killing it")
        print(f"replacing stale queue lock left by pid {pid}", flush=True)
    path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    return path


def _release_lock(path: Path) -> None:
    try:
        current = int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return
    if current == os.getpid():
        path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
