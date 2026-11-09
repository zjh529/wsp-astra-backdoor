"""Fail-closed checks before any audit session.

Isolation is checked by running code in the real bubblewrap sandbox. Pi tools
are checked by reading ``pi --help`` and by starting pi with a probe extension
that writes the active tool list. If that list is missing or not empty, or the
model id is not the requested one, the process exits non-zero.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path

from padbench_audit.common import assert_mount_path_clean, strip_ansi, write_json
from padbench_audit.pi_client import build_run_argv
from padbench_audit.safetensors_io import write_safetensors
from padbench_audit.sandbox import SANDBOX_ENV_KEYS, bwrap_available, run_code

PROBE = Path(__file__).resolve().parent / "pi_tool_probe.ts"
EXPECTED_MODEL = ("openai-codex", "gpt-6-astra")
HELP_NEEDLES = (
    "--no-tools",
    "Disable all tools",
    "--no-mcp",
    "--no-extensions",
    "--session-id",
    "--mode",
    "--thinking",
    "--offline",
    "--no-approve",
    "--system-prompt",
    "--no-skills",
    "--no-context-files",
)


def isolation_report(
    *,
    sandbox_python: Path,
    private_dir: Path,
    mount_paths: list[Path],
) -> dict:
    if not bwrap_available():
        return {"ok": False, "error": "bwrap is not installed"}
    if not sandbox_python.is_file():
        return {"ok": False, "error": f"sandbox python is missing: {sandbox_python}"}
    for path in mount_paths:
        try:
            assert_mount_path_clean(path, what="declared mount")
        except SystemExit as exc:
            return {"ok": False, "error": str(exc)}
    canary = private_dir / "DO_NOT_MOUNT"
    private_dir.mkdir(parents=True, exist_ok=True)
    if not canary.exists():
        canary.write_text("private ground truth\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="iso-") as tmp:
        root = Path(tmp)
        case = root / "case"
        work = root / "work"
        case.mkdir()
        work.mkdir()
        raw = (1).to_bytes(4, "little")  # one F32 value, little-endian 1.0 is not required
        # A 1-element F32 tensor. Bytes are 4 zero-ish values of 0.0.
        raw = b"\x00\x00\x00\x00"
        write_safetensors(case / "adapter.safetensors", [("layer_0.proj_q.A", "F32", [1], raw)])
        write_json(case / "shapes.json", {"tensors": [{"dtype": "F32", "name": "layer_0.proj_q.A", "shape": [1]}]})
        code = (
            "import os, socket\n"
            "print('numpy', __import__('numpy').__version__)\n"
            "try:\n"
            "    import torch\n"
            "    print('TORCH-PRESENT')\n"
            "except Exception as exc:\n"
            "    print('torch', type(exc).__name__)\n"
            f"print('private', os.path.exists({str(canary)!r}))\n"
            "print('nvidia', [name for name in os.listdir('/dev') if 'nvidia' in name])\n"
            "print('driver', os.path.exists('/proc/driver/nvidia'))\n"
            "sock = socket.socket(); sock.settimeout(2)\n"
            "try:\n"
            "    sock.connect(('1.1.1.1', 80)); print('CONNECT-OK')\n"
            "except Exception as exc:\n"
            "    print('CONNECT', type(exc).__name__)\n"
            "print('case', sorted(os.listdir('/case')))\n"
            "print('ENV', ' '.join(sorted(os.environ)))\n"
        )
        try:
            result = run_code(
                code,
                sandbox_python=sandbox_python,
                case_dir=case,
                work_dir=work,
                timeout_s=60,
                memory_bytes=8 << 30,
                max_output_chars=4000,
                code_dir=root / "prog",
            )
        except SystemExit as exc:
            return {"ok": False, "error": str(exc)}
    text = result.text_for_model
    env_keys = _env_keys(text)
    checks = {
        "numpy_imported": "numpy " in text and "TORCH-PRESENT" not in text,
        "torch_missing": "torch ModuleNotFoundError" in text,
        "private_hidden": "private False" in text,
        "no_nvidia_device": "nvidia []" in text,
        "no_nvidia_driver": "driver False" in text,
        "network_blocked": "CONNECT-OK" not in text and "CONNECT " in text,
        "case_visible": "adapter.safetensors" in text,
        "env_only_expected": env_keys == set(SANDBOX_ENV_KEYS),
        "sentinel_absent": "PADBENCH_SENTINEL" not in text,
        "exit_ok": result.exit_code == 0 and not result.timed_out,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "env_keys": sorted(env_keys),
        "output": text,
        "exit_code": result.exit_code,
    }


def _env_keys(text: str) -> set[str]:
    for line in text.splitlines():
        if line.startswith("ENV "):
            return set(line.split()[1:])
    return set()


def pi_report(*, pi_bin: str, provider: str, model_id: str, effort: str) -> dict:
    if not Path(pi_bin).is_file() and _which(pi_bin) is None:
        return {"ok": False, "error": f"pi binary not found: {pi_bin}", "verified": False}
    version = subprocess.run([pi_bin, "--version"], capture_output=True, text=True, check=False)
    version_text = strip_ansi((version.stdout or "") + (version.stderr or "")).strip()
    help_run = subprocess.run([pi_bin, "--help"], capture_output=True, text=True, check=False)
    help_text = strip_ansi((help_run.stdout or "") + (help_run.stderr or ""))
    missing = [needle for needle in HELP_NEEDLES if needle not in help_text]
    sample = build_run_argv(
        pi_bin=pi_bin,
        provider=provider,
        model_id=model_id,
        effort=effort,
        system_prompt=Path("/tmp/system_prompt.txt"),
        session_dir=Path("/tmp/pi-sessions"),
        session_id="padbench-case-001",
        user_message="Begin.",
    )
    probe = _probe_tools(pi_bin=pi_bin, provider=provider, model_id=model_id, effort=effort)
    version_ok = "1.5.0" in version_text.split()
    # Accept "1.5.0" as its own token or as the whole output.
    if version_text == "1.5.0" or version_text.endswith("1.5.0"):
        version_ok = True
    report = {
        "path": pi_bin,
        "version": version_text,
        "version_ok": bool(version_ok),
        "help_ok": help_run.returncode == 0 and not missing,
        "help_missing": missing,
        "run_argv_disables_tools": (
            "--no-tools" in sample
            and "--approve" in sample
            and "--no-approve" not in sample
            and "-e" not in sample
            and "--extension" not in sample
        ),
        "probe": probe,
    }
    report["ok"] = bool(
        report["version_ok"] and report["help_ok"] and report["run_argv_disables_tools"] and probe.get("ok")
    )
    report["verified"] = report["ok"]
    if not report["ok"]:
        report["error"] = "pi tools are not verified disabled, or the model id does not match; refusing to run"
    return report


def _which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(":"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _probe_tools(*, pi_bin: str, provider: str, model_id: str, effort: str) -> dict:
    if not PROBE.is_file():
        return {"ok": False, "error": f"probe extension missing: {PROBE}"}
    with tempfile.TemporaryDirectory(prefix="pi-probe-") as tmp:
        root = Path(tmp)
        cwd = root / "cwd"
        sessions = root / "sessions"
        cwd.mkdir()
        sessions.mkdir()
        probe_path = root / "tools.json"
        argv = [
            pi_bin,
            "--mode",
            "rpc",
            "--no-tools",
            "--no-extensions",
            "-e",
            str(PROBE),
            "--no-skills",
            "--no-mcp",
            "--no-context-files",
            "--no-prompt-templates",
            "--no-approve",
            "--offline",
            "--no-session",
            "--provider",
            provider,
            "--model",
            model_id,
            "--thinking",
            effort,
            "--session-dir",
            str(sessions),
            "--session-id",
            "probe1",
        ]
        env = os.environ.copy()
        env["PADBENCH_TOOL_PROBE"] = str(probe_path)
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = None
        stderr = b""
        state = None
        deadline = time.monotonic() + 60
        try:
            assert proc.stdout is not None and proc.stderr is not None and proc.stdin is not None
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            selector.register(proc.stderr, selectors.EVENT_READ)
            while time.monotonic() < deadline and payload is None and proc.poll() is None:
                if probe_path.is_file() and probe_path.stat().st_size > 0:
                    payload = json.loads(probe_path.read_text(encoding="utf-8"))
                    break
                events = selector.select(timeout=0.1)
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if key.fileobj is proc.stderr:
                        stderr += chunk
            if payload is None and probe_path.is_file() and probe_path.stat().st_size > 0:
                payload = json.loads(probe_path.read_text(encoding="utf-8"))
            if payload is not None and proc.poll() is None:
                proc.stdin.write(b'{"id":"1","type":"get_state"}\n')
                proc.stdin.flush()
                state = _read_response(proc.stdout, timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                if proc.stderr is not None:
                    stderr += proc.stderr.read() or b""
            except Exception:
                pass
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass
            proc.wait(timeout=5)
    if payload is None:
        return {
            "ok": False,
            "error": "pi exited before the tool probe was written; tools are not verified disabled",
            "stderr": stderr.decode("utf-8", "replace")[-2000:],
        }
    active = payload.get("active")
    model = payload.get("model") or {}
    state_model = ((state or {}).get("data") or {}).get("model") or {}
    model_ok = model.get("provider") == provider and model.get("id") == model_id
    state_ok = state_model.get("provider") == provider and state_model.get("id") == model_id
    ok = active == [] and model_ok and state_ok
    return {
        "ok": ok,
        "active": active,
        "model": model,
        "get_state_model": {"provider": state_model.get("provider"), "id": state_model.get("id")},
        "error": None if ok else "probe did not confirm an empty tool list and the expected model",
    }


def _read_response(stdout, timeout: float) -> dict | None:
    selector = selectors.DefaultSelector()
    selector.register(stdout, selectors.EVENT_READ)
    pending = b""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        events = selector.select(timeout=max(0.0, end - time.monotonic()))
        if not events:
            return None
        chunk = os.read(stdout.fileno(), 65536)
        if not chunk:
            return None
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if event.get("type") == "response" and event.get("command") == "get_state":
                return event
    return None


def run_preflight(
    *,
    sandbox_python: Path,
    private_dir: Path,
    pi_bin: str,
    mount_paths: list[Path],
    provider: str = EXPECTED_MODEL[0],
    model_id: str = EXPECTED_MODEL[1],
    effort: str = "xhigh",
    skip_pi: bool = False,
) -> dict:
    isolation = isolation_report(
        sandbox_python=sandbox_python,
        private_dir=private_dir,
        mount_paths=mount_paths,
    )
    if skip_pi:
        pi = {"ok": False, "skipped": True, "verified": False, "error": "pi checks were skipped"}
        ok = False
    else:
        pi = pi_report(pi_bin=pi_bin, provider=provider, model_id=model_id, effort=effort)
        ok = bool(isolation.get("ok") and pi.get("ok"))
    report = {
        "ok": ok,
        "isolation": isolation,
        "pi": pi,
        "expected_model": f"{provider}/{model_id}",
        "fail_closed": True,
    }
    return report
