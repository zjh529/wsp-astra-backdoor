"""Drive pi non-interactively with every pi tool disabled.

pi 1.5.0 treats ``--no-tools`` as "disable all tools by default (built-in and
extension)". ``--no-mcp``, ``--no-extensions``, ``--no-skills``, and
``--no-context-files`` are also passed so a project file cannot put a tool
back. ``--session-id`` resumes the same session on later turns, and ``--no-tools``
is repeated so a resume cannot restore tools. The process cwd is an empty
directory that does not contain the case.

This module never reads or copies pi auth files. pi is executed as a child and
uses whatever credentials the operator's environment already provides.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

_TOOL_EVENTS = {
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "bash_execution_update",
}
_EFFORTS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}


class PiToolsEnabled(RuntimeError):
    """The reply or the event stream shows a pi tool. The case must stop."""


@dataclass
class PiTurn:
    text: str
    raw_stdout: str
    raw_stderr: str
    usage: dict | None
    model: dict | None
    wall_s: float
    exit_code: int
    tool_events: list = field(default_factory=list)
    session_id: str | None = None
    argv: list[str] = field(default_factory=list)


def assert_audit_argv(argv: list[str]) -> None:
    """Fail closed before exec if the command could leave a pi tool enabled."""
    joined = " ".join(argv)
    required = [
        "--no-tools",
        "--no-extensions",
        "--no-mcp",
        "--no-skills",
        "--no-context-files",
        "--no-prompt-templates",
        "--no-approve",
        "--offline",
        "--session-id",
        "--session-dir",
    ]
    missing = [flag for flag in required if flag not in argv]
    if missing:
        raise SystemExit(f"refusing to launch pi; argv is missing {missing}")
    if "--mode" not in argv or "json" not in argv:
        raise SystemExit("refusing to launch pi without --mode json")
    if "-e" in argv or "--extension" in argv:
        raise SystemExit("refusing to launch the audit driver with a pi extension")
    if "--api-key" in argv or "print-api-key" in joined or "print-bearer-token" in joined:
        raise SystemExit("refusing to launch pi with an explicit credential argument")
    if "--no-builtin-tools" in argv and "--no-tools" not in argv:
        raise SystemExit("--no-builtin-tools keeps extension tools; use --no-tools")


def build_run_argv(
    *,
    pi_bin: str,
    provider: str,
    model_id: str,
    effort: str,
    system_prompt: Path,
    session_dir: Path,
    session_id: str,
    user_message: str,
) -> list[str]:
    if effort not in _EFFORTS:
        raise SystemExit(f"effort {effort!r} is not a pi thinking level")
    if not session_id[:1].isalnum() or not session_id[-1:].isalnum():
        raise SystemExit(f"session id {session_id!r} must start and end with a letter or number")
    if user_message.startswith("@") or user_message.startswith("-"):
        raise SystemExit("refusing to pass a user message that pi would parse as a file or a flag")
    argv = [
        pi_bin,
        "--mode",
        "json",
        "--no-tools",
        "--no-extensions",
        "--no-skills",
        "--no-mcp",
        "--no-context-files",
        "--no-prompt-templates",
        "--no-approve",
        "--offline",
        "--provider",
        provider,
        "--model",
        model_id,
        "--thinking",
        effort,
        "--system-prompt",
        str(system_prompt),
        "--session-dir",
        str(session_dir),
        "--session-id",
        session_id,
        "--name",
        session_id,
        "--",
        user_message,
    ]
    assert_audit_argv(argv)
    return argv


def parse_pi_stdout(raw: str) -> dict:
    events = []
    session_id = None
    bad_lines = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "session" and isinstance(event.get("id"), str):
            session_id = event["id"]
    text, usage, model, tools = _from_events(events)
    return {
        "events": events,
        "text": text,
        "usage": usage,
        "model": model,
        "tool_events": tools,
        "session_id": session_id,
        "non_json_lines": bad_lines,
    }


def _from_events(events: list[dict]) -> tuple[str, dict | None, dict | None, list]:
    tools: list = []
    usage = None
    model = None
    text_parts: list[str] = []
    for event in events:
        kind = event.get("type")
        if kind in _TOOL_EVENTS:
            tools.append({"type": kind, "toolName": event.get("toolName")})
        inner = event.get("assistantMessageEvent") if kind == "message_update" else None
        if isinstance(inner, dict) and str(inner.get("type", "")).startswith("toolcall"):
            tools.append({"type": inner.get("type"), "toolName": inner.get("toolName")})
        if kind == "message_update" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if kind != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if isinstance(message.get("usage"), dict):
            usage = message["usage"]
        if message.get("provider") and message.get("model"):
            model = {"provider": message["provider"], "id": message["model"]}
        text_parts.append(_content_text(message.get("content")))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"toolCall", "tool_call"}:
                    tools.append({"type": block.get("type"), "toolName": block.get("name")})
    return "\n".join(part for part in text_parts if part).strip(), usage, model, tools


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or block.get("content") or ""))
    return "\n".join(parts)


class PiDriver:
    def __init__(
        self,
        *,
        pi_bin: str,
        provider: str,
        model_id: str,
        effort: str,
        system_prompt: Path,
        session_dir: Path,
        session_id: str,
        cwd: Path,
        timeout_s: int,
    ) -> None:
        self.pi_bin = pi_bin
        self.provider = provider
        self.model_id = model_id
        self.effort = effort
        self.system_prompt = system_prompt
        self.session_dir = session_dir
        self.session_id = session_id
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.cwd.mkdir(parents=True, exist_ok=True)
        if any(self.cwd.iterdir()):
            raise SystemExit(f"pi cwd {self.cwd} is not empty; refusing to run where the case might be visible")

    def run_turn(self, *, user_message: str) -> PiTurn:
        if any(self.cwd.iterdir()):
            raise PiToolsEnabled(f"pi cwd {self.cwd} gained files before the turn")
        argv = build_run_argv(
            pi_bin=self.pi_bin,
            provider=self.provider,
            model_id=self.model_id,
            effort=self.effort,
            system_prompt=self.system_prompt,
            session_dir=self.session_dir,
            session_id=self.session_id,
            user_message=user_message,
        )
        env = os.environ.copy()
        env.pop("PADBENCH_TOOL_PROBE", None)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=self.cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            parsed = parse_pi_stdout(stdout)
            return PiTurn(
                text=parsed["text"],
                raw_stdout=stdout,
                raw_stderr=stderr + f"\n[pi timed out after {self.timeout_s} s]",
                usage=parsed["usage"],
                model=parsed["model"],
                wall_s=round(time.monotonic() - started, 3),
                exit_code=124,
                tool_events=parsed["tool_events"],
                session_id=parsed["session_id"],
                argv=argv,
            )
        parsed = parse_pi_stdout(completed.stdout or "")
        turn = PiTurn(
            text=parsed["text"],
            raw_stdout=completed.stdout or "",
            raw_stderr=completed.stderr or "",
            usage=parsed["usage"],
            model=parsed["model"],
            wall_s=round(time.monotonic() - started, 3),
            exit_code=int(completed.returncode),
            tool_events=parsed["tool_events"],
            session_id=parsed["session_id"],
            argv=argv,
        )
        if turn.tool_events:
            raise PiToolsEnabled(f"pi used tools: {turn.tool_events}")
        if any(self.cwd.iterdir()):
            names = sorted(p.name for p in self.cwd.iterdir())
            raise PiToolsEnabled(f"pi cwd gained files {names}; tools may not be disabled")
        return turn
