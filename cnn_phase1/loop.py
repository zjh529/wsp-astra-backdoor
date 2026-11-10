"""One pi session per case. This module does not read ground truth.

Hypothesis check, nudge, infra backoff, compaction-off, --no-tools, and the
numpy bubblewrap sandbox are the existing padbench-audit behavior. Usage-limit
errors pause instead of counting as a failed attempt. Every case uses the same
prompt, the same nudge text, and the same retry schedule.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cnn_phase1.hashing import canonical_json, sha256_bytes, sha256_file, write_json
from cnn_phase1.paths import assert_mount_path_clean
from cnn_phase1.prelude import PRELUDE
from cnn_phase1.protocol import (
    BUDGET,
    FORMAT_CODE,
    FORMAT_COUNT,
    FORMAT_JSON,
    SYSTEM_PROMPT_PATH,
    config_sha256,
    load_config,
    prompt_sha256,
    prompt_text,
    system_prompt_sha256,
)
from cnn_phase1.report import artifact_errors, schema_errors

_PADBENCH = Path(__file__).resolve().parents[1] / "padbench-audit"
if str(_PADBENCH) not in sys.path:
    sys.path.insert(0, str(_PADBENCH))

from padbench_audit.common import git_identity  # noqa: E402
from padbench_audit.extract import extract_verdict, hypothesis_stated, preamble, python_blocks  # noqa: E402
from padbench_audit.loop import nudge_text  # noqa: E402
from padbench_audit.pi_client import PiDriver, PiToolsEnabled, PiTurn, is_infra_failure  # noqa: E402
from padbench_audit.sandbox import run_code  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def is_usage_limit(turn: PiTurn) -> bool:
    if not is_infra_failure(
        text=turn.text,
        usage=turn.usage,
        stop_reason=turn.stop_reason,
        error_message=turn.error_message,
        exit_code=turn.exit_code,
    ):
        return False
    blob = "\n".join(
        [
            turn.error_message or "",
            turn.raw_stderr or "",
            turn.raw_stdout or "",
            turn.text or "",
        ]
    ).lower()
    return "usage limit" in blob


def run_case(
    *,
    config: dict,
    blind: Path,
    case_id: str,
    sandbox_python: Path,
    pi_bin: str | None = None,
    pi_cwd: Path | None = None,
    resume: bool = False,
    scripted_replies: list | None = None,
    sleeper=time.sleep,
) -> dict:
    case_dir = blind / "cases" / case_id
    if not (case_dir / "shapes.json").is_file():
        raise SystemExit(f"missing case {case_dir}")
    has_weights = (case_dir / "weights.npz").is_file()
    has_placeholder = (case_dir / "WEIGHTS_UNAVAILABLE.txt").is_file()
    if has_weights == has_placeholder:
        raise SystemExit(f"{case_id} must contain either weights.npz or the unavailable placeholder")
    assert_mount_path_clean(case_dir, what="case")
    run_dir = blind / "runs" / case_id
    saved = {}
    if resume:
        if not (run_dir / "state.json").is_file():
            raise SystemExit(f"{run_dir} has no state.json to resume")
        if (run_dir / "prediction.sha256").is_file():
            raise SystemExit(f"{case_id} already has a frozen prediction; refusing to resume")
        saved = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            raise SystemExit(f"{case_id} is completed; refusing to resume")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} already exists; refusing to start a second session")
    work = run_dir / "work"
    programs = run_dir / "programs"
    session_dir = run_dir / "pi_sessions"
    if resume and saved.get("pi_cwd"):
        if pi_cwd is not None and Path(saved["pi_cwd"]).resolve() != Path(pi_cwd).resolve():
            raise SystemExit(f"this case was started with pi cwd {saved['pi_cwd']}")
        pi_cwd = Path(saved["pi_cwd"])
    elif pi_cwd is None:
        pi_cwd = Path("/dev/shm/cnn-p1-pi") / case_id if Path("/dev/shm").is_dir() else run_dir / "pi_cwd"
    for path in (work, programs, pi_cwd, session_dir, run_dir / "turns"):
        path.mkdir(parents=True, exist_ok=True)
    assert_mount_path_clean(work, what="work")
    assert_mount_path_clean(programs, what="programs")
    task = prompt_text()
    if sha256_bytes(task.encode("utf-8")) != prompt_sha256():
        raise SystemExit("prompt file changed while it was being read")
    (run_dir / "task.txt").write_text(task, encoding="utf-8")
    (run_dir / "system_prompt.txt").write_text(system_prompt_text_cached(), encoding="utf-8")
    if resume and (run_dir / "manifest.json").is_file():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest.setdefault("resumes", []).append(
            {
                "at": _now(),
                "from_status": saved.get("status"),
                "turns": saved.get("turns"),
                "runs": saved.get("runs"),
            }
        )
        write_json(run_dir / "manifest.json", manifest)
    else:
        write_json(run_dir / "manifest.json", _manifest(config, case_dir, case_id, has_weights))
    nudge_cfg = config["nudge"]
    nudge_on = bool(nudge_cfg["enabled"])
    min_runs = int(nudge_cfg["min_runs_before_accept"])
    max_nudges = int(nudge_cfg["max_nudges"])
    max_runs = int(config["max_runs"])
    max_format = int(config["max_format_reminders"])
    backoff = tuple(int(value) for value in config["infra_backoff_s"])
    max_attempts = int(config["infra_max_attempts"])
    if resume:
        state = dict(saved)
        state["status"] = "running"
        state["consecutive_infra_failures"] = 0
        state["resumable"] = True
        state["counted_as_failure"] = False
        state.setdefault("pending_message", task)
        state["pi_cwd"] = str(pi_cwd)
    else:
        state = {
            "case_id": case_id,
            "session_id": f"cnnp1-{case_id}",
            "runs": 0,
            "nudges": 0,
            "format_reminders": 0,
            "turns": 0,
            "status": "running",
            "consecutive_infra_failures": 0,
            "resumable": True,
            "counted_as_failure": False,
            "pending_message": task,
            "pi_cwd": str(pi_cwd),
        }
    if scripted_replies is not None:
        active = _Scripted(list(scripted_replies))
    else:
        if not pi_bin:
            raise SystemExit("pi bin is required")
        active = PiDriver(
            pi_bin=pi_bin,
            provider=config["provider"],
            model_id=config["model_id"],
            effort=config["effort"],
            system_prompt=run_dir / "system_prompt.txt",
            session_dir=session_dir,
            session_id=state["session_id"],
            cwd=pi_cwd,
            timeout_s=int(config["pi_timeout_s"]),
        )
    tool_log = run_dir / "tool_calls.jsonl"
    nudge_log = run_dir / "nudges.jsonl"

    def save() -> None:
        write_json(run_dir / "state.json", state)

    try:
        while state["status"] == "running":
            message = state["pending_message"]
            state["turns"] += 1
            turn_dir = run_dir / "turns" / f"t{state['turns']:03d}"
            turn_dir.mkdir()
            (turn_dir / "prompt.txt").write_text(message, encoding="utf-8")
            try:
                turn = active.run_turn(user_message=message)
            except PiToolsEnabled as exc:
                state["status"] = "invalid_tools"
                state["resumable"] = False
                state["counted_as_failure"] = True
                state["error"] = str(exc)
                (turn_dir / "error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                break
            if is_usage_limit(turn):
                _write_turn(turn_dir, turn, extra={"usage_limit": True, "infra": False})
                with (run_dir / "usage_limit.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"turn": state["turns"], "error": turn.error_message}) + "\n")
                state["status"] = "usage_limit"
                state["resumable"] = True
                state["counted_as_failure"] = False
                state["error"] = turn.error_message or "usage limit"
                break
            if is_infra_failure(
                text=turn.text,
                usage=turn.usage,
                stop_reason=turn.stop_reason,
                error_message=turn.error_message,
                exit_code=turn.exit_code,
            ):
                state["consecutive_infra_failures"] += 1
                attempt = state["consecutive_infra_failures"]
                give_up = attempt >= max_attempts
                wait = 0 if give_up else backoff[min(attempt - 1, len(backoff) - 1)]
                record = {
                    "turn": state["turns"],
                    "attempt": attempt,
                    "stop_reason": turn.stop_reason,
                    "error_message": turn.error_message,
                    "exit_code": turn.exit_code,
                    "backoff_s": wait,
                    "resend": not give_up,
                }
                _write_turn(turn_dir, turn, extra={"infra": True, **record})
                with (run_dir / "infra_errors.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if give_up:
                    state["status"] = "pi_error"
                    state["resumable"] = True
                    state["counted_as_failure"] = True
                    state["error"] = turn.error_message or turn.stop_reason or f"pi exit {turn.exit_code}"
                    break
                save()
                sleeper(wait)
                continue
            state["consecutive_infra_failures"] = 0
            _write_turn(turn_dir, turn)
            if turn.compaction_events:
                with (run_dir / "compactions.jsonl").open("a", encoding="utf-8") as handle:
                    for event in turn.compaction_events:
                        handle.write(json.dumps({"turn": state["turns"], **event}) + "\n")
            if turn.model and (
                turn.model.get("provider") != config["provider"] or turn.model.get("id") != config["model_id"]
            ):
                state["status"] = "invalid_model"
                state["resumable"] = False
                state["counted_as_failure"] = True
                state["error"] = f"pi replied as {turn.model}"
                break
            blocks = python_blocks(turn.text)
            if len(blocks) > 1:
                _remind(state, FORMAT_COUNT, max_format)
                save()
                continue
            if len(blocks) == 1:
                if not hypothesis_stated(turn.text):
                    (turn_dir / "preamble.txt").write_text(preamble(turn.text), encoding="utf-8")
                    _remind(state, FORMAT_CODE, max_format)
                    save()
                    continue
                if state["runs"] >= max_runs:
                    _remind(state, BUDGET, max_format, status_on_exceed="budget_exhausted")
                    save()
                    continue
                code = blocks[0]
                (turn_dir / "code.py").write_text(code + "\n", encoding="utf-8")
                try:
                    result = run_code(
                        code,
                        sandbox_python=sandbox_python,
                        case_dir=case_dir,
                        work_dir=work,
                        base_dir=None,
                        timeout_s=int(config["timeout_s"]),
                        max_output_chars=int(config["max_output_chars"]),
                        memory_bytes=int(config["memory_bytes"]),
                        work_quota_bytes=int(config["work_quota_bytes"]),
                        code_dir=programs,
                        prelude=PRELUDE,
                    )
                except SystemExit as exc:
                    state["status"] = "sandbox_refused"
                    state["resumable"] = False
                    state["counted_as_failure"] = True
                    state["error"] = str(exc)
                    (turn_dir / "error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                    break
                state["runs"] += 1
                (turn_dir / "sandbox_stdout.txt").write_text(result.stdout, encoding="utf-8")
                (turn_dir / "sandbox_stderr.txt").write_text(result.stderr, encoding="utf-8")
                record = {
                    "turn": state["turns"],
                    "run": state["runs"],
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "wall_s": result.wall_s,
                    "code_sha256": sha256_bytes(code.encode()),
                }
                write_json(turn_dir / "sandbox.json", record)
                with tool_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**record, "code": code, "output": result.text_for_model}) + "\n")
                state["pending_message"] = (
                    f"Run {state['runs']} of {max_runs} finished "
                    f"(exit {result.exit_code}, {result.wall_s} s).\n\n{result.text_for_model}"
                )
                save()
                continue
            candidate = extract_verdict(turn.text)
            errors = schema_errors(candidate)
            if errors:
                _remind(state, FORMAT_JSON + "\n" + "\n".join(f"- {item}" for item in errors), max_format)
                (turn_dir / "verdict_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
                save()
                continue
            if nudge_on and state["runs"] < min_runs and state["nudges"] < max_nudges:
                state["nudges"] += 1
                (run_dir / f"early_answer_{state['nudges']}.md").write_text(turn.text, encoding="utf-8")
                state["pending_message"] = nudge_text(state["runs"], max_runs)
                with nudge_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"turn": state["turns"], "nudge": state["nudges"], "runs": state["runs"]}) + "\n"
                    )
                (turn_dir / "nudge.txt").write_text(state["pending_message"], encoding="utf-8")
                save()
                continue
            file_errors = artifact_errors(candidate, work)
            if file_errors:
                _remind(state, FORMAT_JSON + "\n" + "\n".join(f"- {item}" for item in file_errors), max_format)
                (turn_dir / "verdict_errors.json").write_text(json.dumps(file_errors, indent=2) + "\n", encoding="utf-8")
                save()
                continue
            _freeze(run_dir, turn.text, candidate, work)
            state["status"] = "completed"
            state["resumable"] = False
            state["counted_as_failure"] = False
            state["pending_message"] = None
            write_json(run_dir / "state.json", state)
    finally:
        write_json(run_dir / "state.json", state)
        _transcript(run_dir)
    return state


def system_prompt_text_cached() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _remind(state: dict, message: str, max_format: int, status_on_exceed: str = "invalid_format") -> None:
    state["format_reminders"] += 1
    state["pending_message"] = message
    if state["format_reminders"] > max_format:
        state["status"] = status_on_exceed
        state["counted_as_failure"] = True
        state["resumable"] = status_on_exceed != "invalid_tools"


def _freeze(run_dir: Path, text: str, verdict: dict, work: Path) -> None:
    (run_dir / "final_answer.md").write_text(text, encoding="utf-8")
    blob = canonical_json(verdict)
    (run_dir / "prediction.json").write_bytes(blob)
    (run_dir / "prediction.sha256").write_text(sha256_bytes(blob) + "\n", encoding="utf-8")
    delta = work / "delta.npz"
    artifacts = {
        "trigger_npz_sha256": sha256_file(work / "trigger.npz"),
        "delta_npz_sha256": sha256_file(delta) if delta.is_file() else None,
    }
    artifact_blob = canonical_json(artifacts)
    (run_dir / "artifacts.json").write_bytes(artifact_blob)
    (run_dir / "artifacts.sha256").write_text(sha256_bytes(artifact_blob) + "\n", encoding="utf-8")


def _manifest(config: dict, case_dir: Path, case_id: str, has_weights: bool) -> dict:
    weights = case_dir / "weights.npz"
    placeholder = case_dir / "WEIGHTS_UNAVAILABLE.txt"
    return {
        "harness": "cnn-phase1",
        "protocol": "phase1",
        "case_id": case_id,
        "model": config["model"],
        "effort": config["effort"],
        "max_runs": int(config["max_runs"]),
        "timeout_s": int(config["timeout_s"]),
        "nudge": config["nudge"],
        "infra_backoff_s": list(config["infra_backoff_s"]),
        "infra_max_attempts": int(config["infra_max_attempts"]),
        "prompt_sha256": prompt_sha256(),
        "system_prompt_sha256": system_prompt_sha256(),
        "prelude_sha256": sha256_bytes(PRELUDE.encode("utf-8")),
        "config_sha256": config_sha256(),
        "shapes_sha256": sha256_file(case_dir / "shapes.json"),
        "weights_sha256": sha256_file(weights) if has_weights else None,
        "placeholder_sha256": sha256_file(placeholder) if placeholder.is_file() else None,
        **git_identity(REPO),
        "note": "Labels, the trigger, and the condition are not in this file.",
    }


def _write_turn(turn_dir: Path, turn: PiTurn, extra: dict | None = None) -> None:
    (turn_dir / "answer.md").write_text(turn.text, encoding="utf-8")
    (turn_dir / "events.jsonl").write_text(turn.raw_stdout, encoding="utf-8")
    (turn_dir / "stderr.log").write_text(turn.raw_stderr, encoding="utf-8")
    logged = list(turn.argv)
    if len(logged) >= 2 and logged[-2] == "--":
        logged[-1] = f"<prompt.txt, {len(turn.argv[-1])} chars>"
    payload = {
        "exit_code": turn.exit_code,
        "wall_s": turn.wall_s,
        "usage": turn.usage,
        "model": turn.model,
        "session_id": turn.session_id,
        "tool_events": turn.tool_events,
        "stop_reason": turn.stop_reason,
        "error_message": turn.error_message,
        "compaction_events": turn.compaction_events,
        "argv": logged,
    }
    if extra:
        payload.update(extra)
    write_json(turn_dir / "run.json", payload)


def _transcript(run_dir: Path) -> None:
    parts = []
    for turn in sorted((run_dir / "turns").glob("t*")):
        prompt = (turn / "prompt.txt").read_text(encoding="utf-8") if (turn / "prompt.txt").is_file() else ""
        answer = (turn / "answer.md").read_text(encoding="utf-8") if (turn / "answer.md").is_file() else ""
        parts.append(f"# {turn.name}\n\n## prompt\n\n{prompt}\n\n## answer\n\n{answer}\n")
    (run_dir / "transcript.md").write_text("\n".join(parts), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Scripted:
    def __init__(self, replies: list) -> None:
        self._replies = list(replies)

    def run_turn(self, *, user_message: str) -> PiTurn:
        if not self._replies:
            raise SystemExit("scripted driver ran out of replies")
        item = self._replies.pop(0)
        if isinstance(item, PiTurn):
            return item
        return PiTurn(
            text=item,
            raw_stdout="",
            raw_stderr="",
            usage={"input": 3, "output": 5, "totalTokens": 8},
            model={"provider": "openai-codex", "id": "gpt-6-astra"},
            wall_s=0.01,
            exit_code=0,
            stop_reason="stop",
            session_id="scripted",
            argv=["scripted", "--no-tools", "--", user_message],
        )


def case_ids(blind: Path) -> list[str]:
    order = blind / "case_order.txt"
    if not order.is_file():
        raise SystemExit(f"missing {order}")
    return [line.strip() for line in order.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_study_config() -> dict:
    return load_config()
