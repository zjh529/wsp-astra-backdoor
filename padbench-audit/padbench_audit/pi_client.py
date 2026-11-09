"""Drive pi non-interactively with every pi tool disabled.

pi 1.5.0 treats ``--no-tools`` as "disable all tools by default (built-in and
extension)". ``--no-mcp``, ``--no-extensions``, ``--no-skills``, and
``--no-context-files`` are also passed so a project file cannot put a tool
back. ``--session-id`` resumes the same session on later turns, and ``--no-tools``
is repeated so a resume cannot restore tools. The process cwd does not contain
the case. Its only file is the compaction override written below.

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

# pi 1.5.0 enables auto-compaction by default and has no CLI flag for it.
# PI_CODING_AGENT_DIR would also move auth.json, so it is not set. Project
# settings at <cwd>/.pi/settings.json override the global file when the cwd is
# trusted with --approve. The cwd is created by this harness and contains only
# that file. Global settings and auth are left untouched and are not read.
COMPACTION_SETTINGS = {"compaction": {"enabled": False}}
COMPACTION_SETTINGS_TEXT = json.dumps(COMPACTION_SETTINGS, indent=2) + "\n"


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
    compaction_events: list = field(default_factory=list)
    stop_reason: str | None = None
    error_message: str | None = None
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
        "--approve",
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
    if "--no-approve" in argv:
        raise SystemExit("refusing to launch pi with --no-approve; the isolated cwd settings would be ignored")
    if "--api-key" in argv or "print-api-key" in joined or "print-bearer-token" in joined:
        raise SystemExit("refusing to launch pi with an explicit credential argument")
    if "PI_CODING_AGENT_DIR" in joined:
        raise SystemExit("refusing to relocate the pi agent directory; that would move auth.json")
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
        "--approve",
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
    text, usage, model, tools, compactions, stop_reason, error_message = _from_events(events)
    return {
        "events": events,
        "text": text,
        "usage": usage,
        "model": model,
        "tool_events": tools,
        "compaction_events": compactions,
        "stop_reason": stop_reason,
        "error_message": error_message,
        "session_id": session_id,
        "non_json_lines": bad_lines,
    }


def is_infra_failure(
    *,
    text: str,
    usage: dict | None,
    stop_reason: str | None,
    error_message: str | None,
    exit_code: int | None,
) -> bool:
    """True when pi did not produce a model reply.

    A fetch failure is exit 0 with stopReason "error", an errorMessage, and no
    tokens. That must not be read as an empty final answer.
    """
    if stop_reason == "error" or error_message:
        return True
    if exit_code not in (0, None):
        return True
    output = usage.get("output") if isinstance(usage, dict) else None
    if isinstance(output, bool):
        output = None
    return not (text or "").strip() and output == 0


def _from_events(events: list[dict]) -> tuple[str, dict | None, dict | None, list, list, str | None, str | None]:
    tools: list = []
    compactions: list = []
    usage = None
    model = None
    text = ""
    stop_reason = None
    error_message = None
    for event in events:
        kind = event.get("type")
        if kind in {"compaction_start", "compaction_end"}:
            compactions.append(
                {
                    "type": kind,
                    "reason": event.get("reason"),
                    "aborted": event.get("aborted"),
                    "willRetry": event.get("willRetry"),
                    "errorMessage": event.get("errorMessage"),
                    "has_summary": isinstance((event.get("result") or {}).get("summary"), str)
                    if isinstance(event.get("result"), dict)
                    else False,
                }
            )
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
        # The last assistant message is authoritative. Earlier message_end
        # events in the same process are retries, not a second answer.
        text = _content_text(message.get("content")).strip()
        stop_reason = message.get("stopReason") or event.get("stopReason")
        error_message = message.get("errorMessage") or event.get("errorMessage")
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"toolCall", "tool_call"}:
                    tools.append({"type": block.get("type"), "toolName": block.get("name")})
    if error_message is not None:
        error_message = str(error_message)
    if stop_reason is not None:
        stop_reason = str(stop_reason)
    return text, usage, model, tools, compactions, stop_reason, error_message


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


def _turn_from_parsed(
    parsed: dict,
    *,
    raw_stdout: str,
    raw_stderr: str,
    wall_s: float,
    exit_code: int,
    argv: list[str],
) -> PiTurn:
    return PiTurn(
        text=parsed["text"],
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        usage=parsed["usage"],
        model=parsed["model"],
        wall_s=wall_s,
        exit_code=exit_code,
        tool_events=parsed["tool_events"],
        compaction_events=parsed["compaction_events"],
        stop_reason=parsed["stop_reason"],
        error_message=parsed["error_message"],
        session_id=parsed["session_id"],
        argv=argv,
    )


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
        assert_cwd_ancestors_clean(self.cwd)
        install_compaction_settings(self.cwd)
        assert_pi_cwd(self.cwd)

    def run_turn(self, *, user_message: str) -> PiTurn:
        assert_cwd_ancestors_clean(self.cwd)
        assert_pi_cwd(self.cwd)
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
        # Leave PI_CODING_AGENT_DIR alone if the operator already set it: that
        # directory is also where auth.json is resolved. This process never
        # reads that file and never points the variable somewhere else.
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
            return _turn_from_parsed(
                parsed,
                raw_stdout=stdout,
                raw_stderr=stderr + f"\n[pi timed out after {self.timeout_s} s]",
                wall_s=round(time.monotonic() - started, 3),
                exit_code=124,
                argv=argv,
            )
        parsed = parse_pi_stdout(completed.stdout or "")
        turn = _turn_from_parsed(
            parsed,
            raw_stdout=completed.stdout or "",
            raw_stderr=completed.stderr or "",
            wall_s=round(time.monotonic() - started, 3),
            exit_code=int(completed.returncode),
            argv=argv,
        )
        if turn.tool_events:
            raise PiToolsEnabled(f"pi used tools: {turn.tool_events}")
        assert_pi_cwd(self.cwd)
        return turn


def install_compaction_settings(cwd: Path) -> None:
    """Write the only project file pi is allowed to see in its cwd."""
    directory = cwd / ".pi"
    directory.mkdir(exist_ok=True)
    target = directory / "settings.json"
    target.write_text(COMPACTION_SETTINGS_TEXT, encoding="utf-8")


def assert_cwd_ancestors_clean(cwd: Path) -> None:
    """Refuse if --approve could load project files from a parent directory.

    pi 1.5.0 trusts ``.pi`` resources in the working directory and
    ``.agents/skills`` in the working directory or any ancestor. This check
    rejects any ancestor that already contains ``.pi`` or ``.agents`` entries.
    The settings file this harness writes lives in the cwd, not in an ancestor.
    A bare empty ``.pi`` directory is ignored; pi does not treat that as a
    project resource.
    """
    current = cwd.resolve().parent
    hits: list[Path] = []
    while True:
        hits.extend(_project_entries(current / ".pi"))
        hits.extend(_project_entries(current / ".agents"))
        if current.parent == current:
            break
        current = current.parent
    if hits:
        shown = ", ".join(str(path) for path in hits[:8])
        raise SystemExit(
            "refusing to launch pi: an ancestor of the working directory contains "
            f"project files ({shown}). --approve would trust .pi and .agents resources "
            "from the cwd and its parents. Choose a cwd whose parents have neither."
        )


def _project_entries(path: Path) -> list[Path]:
    if path.is_symlink() or path.is_file():
        return [path]
    if not path.is_dir():
        return []
    # Any child counts, including an empty skills directory. A bare .pi
    # directory has no children and is not a project resource.
    return list(path.rglob("*"))


def build_probe_argv(
    *,
    pi_bin: str,
    provider: str,
    model_id: str,
    effort: str,
    system_prompt: Path,
    session_dir: Path,
    session_id: str,
    probe: Path,
) -> list[str]:
    """Audit argv, with only the changes required to inspect state and not call the model.

    ``--mode json`` becomes ``rpc`` so ``get_state`` can report compaction.
    One ``-e`` loads the tool probe. The initial prompt is omitted so RPC does
    not start a model turn. Every other flag matches :func:`build_run_argv`.
    """
    audit = build_run_argv(
        pi_bin=pi_bin,
        provider=provider,
        model_id=model_id,
        effort=effort,
        system_prompt=system_prompt,
        session_dir=session_dir,
        session_id=session_id,
        user_message="Begin.",
    )
    if audit[-2:] != ["--", "Begin."]:
        raise SystemExit("audit argv does not end with the probe's placeholder prompt")
    argv = audit[:-2]
    mode = argv.index("--mode")
    argv[mode + 1] = "rpc"
    argv.extend(["-e", str(probe)])
    if "--no-approve" in argv or "--approve" not in argv:
        raise SystemExit("probe argv lost --approve")
    if argv.count("-e") != 1:
        raise SystemExit("probe argv must contain exactly one -e")
    return argv


def assert_pi_cwd(cwd: Path) -> None:
    """The cwd may contain only the compaction override this harness wrote."""
    allowed = {cwd / ".pi", cwd / ".pi" / "settings.json"}
    found = list(cwd.rglob("*"))
    extra = [path for path in found if path not in allowed]
    if extra or not (cwd / ".pi" / "settings.json").is_file():
        names = sorted(str(path.relative_to(cwd)) for path in found)
        raise PiToolsEnabled(f"pi cwd contains unexpected files {names}")
    text = (cwd / ".pi" / "settings.json").read_text(encoding="utf-8")
    if text != COMPACTION_SETTINGS_TEXT:
        raise PiToolsEnabled("pi cwd settings.json no longer disables compaction")
