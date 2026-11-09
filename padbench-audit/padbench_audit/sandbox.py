"""Run one Python program in a fresh bubblewrap process.

Adapted from harness/sandbox.py in https://github.com/RikiyaT/lm_understand-logs
(MIT). Each call is a new process. /work is the only writable mount and is kept
between calls. The case and the optional base network are read-only. The
process environment is cleared. GPU device nodes are not mounted.

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from padbench_audit.common import assert_mount_path_clean, assert_no_symlinks
from padbench_audit.prelude import prelude_text

DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_OUTPUT = 15_000
DEFAULT_MEMORY = 48 << 30
DEFAULT_WORK_QUOTA = 20 << 30

# bubblewrap 0.4.0 (Ubuntu 20.04) has no --clearenv. The parent process is
# started with an empty environment instead, and these --setenv values are the
# only ones the sandbox is given. PWD is added by bwrap itself when it chdirs.
# Flags added after 0.4.0 (--clearenv, --chmod, --perms) are not used.
SANDBOX_ENV = (
    ("HOME", "/tmp"),
    ("PATH", "/venv/bin:/usr/bin"),
    ("VIRTUAL_ENV", "/venv"),
    ("PYTHONNOUSERSITE", "1"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("OMP_NUM_THREADS", "8"),
    ("CUDA_VISIBLE_DEVICES", ""),
    ("LANG", "C.UTF-8"),
)
SANDBOX_ENV_KEYS = frozenset(key for key, _value in SANDBOX_ENV) | {"PWD"}
# Present in bwrap >= 0.5.0 only. Kept here so a review can see they are excluded.
_BWRAP_FLAGS_AFTER_0_4 = ("--clearenv", "--chmod", "--perms")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    wall_s: float
    text_for_model: str


def _library_symlinks() -> list[str]:
    """Match the WSP runner: the new root has /bin and /lib pointing at /usr."""
    args = ["--symlink", "usr/bin", "/bin"]
    if Path("/usr/lib").exists():
        args.extend(["--symlink", "usr/lib", "/lib"])
    if Path("/usr/lib64").exists():
        args.extend(["--symlink", "usr/lib64", "/lib64"])
    return args


def bwrap_available() -> bool:
    return Path("/usr/bin/bwrap").is_file() or _which("bwrap") is not None


def _which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(":"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _bwrap_bin() -> str:
    override = os.environ.get("BWRAP_BIN")
    if override:
        if not Path(override).is_file():
            raise SystemExit(f"BWRAP_BIN is not a file: {override}")
        return override
    path = "/usr/bin/bwrap"
    if Path(path).is_file():
        return path
    found = _which("bwrap")
    if not found:
        raise SystemExit("bwrap is not installed; refusing to run unsandboxed")
    return found


def build_bwrap_command(
    *,
    sandbox_python: Path,
    case_dir: Path,
    work_dir: Path,
    base_dir: Path | None,
    code_path: Path,
) -> list[str]:
    for label, path in (
        ("sandbox python", sandbox_python),
        ("case", case_dir),
        ("work", work_dir),
        ("code", code_path),
    ):
        assert_mount_path_clean(path, what=label)
    assert_no_symlinks(case_dir)
    assert_no_symlinks(work_dir)
    if code_path.is_symlink() or not code_path.is_file():
        raise SystemExit(f"program file must be a regular file: {code_path}")
    if base_dir is not None:
        assert_mount_path_clean(base_dir, what="base")
        assert_no_symlinks(base_dir)
    python = sandbox_python.resolve()
    venv = _venv_root(python)
    assert_mount_path_clean(venv, what="sandbox venv")
    cmd = [
        _bwrap_bin(),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--hostname",
        "sandbox",
        "--ro-bind",
        "/usr",
        "/usr",
        *_library_symlinks(),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(venv),
        "/venv",
        "--ro-bind",
        str(case_dir.resolve()),
        "/case",
        "--bind",
        str(work_dir.resolve()),
        "/work",
        "--ro-bind",
        str(code_path.resolve()),
        "/tmp/program.py",
        "--chdir",
        "/work",
    ]
    for key, value in SANDBOX_ENV:
        cmd.extend(["--setenv", key, value])
    cmd.extend(
        [
            "/venv/bin/python",
            "-I",
            "-c",
            "import pathlib,runpy,sys; runpy.run_path('/tmp/program.py', run_name='__main__')",
        ]
    )
    for flag in _BWRAP_FLAGS_AFTER_0_4:
        if flag in cmd:
            raise SystemExit(f"refusing to pass {flag}; it is not in bubblewrap 0.4.0")
    if base_dir is not None:
        # Insert the base mount before --chdir so it is part of the namespace.
        insert_at = cmd.index("--chdir")
        cmd[insert_at:insert_at] = ["--ro-bind", str(base_dir.resolve()), "/base"]
    return cmd


def _venv_root(python: Path) -> Path:
    """The directory that contains bin/python. Bound read-only as /venv."""
    if python.parent.name == "bin":
        return python.parent.parent
    return python.parent


def _limits(memory_bytes: int, work_quota_bytes: int):
    import resource

    def _apply() -> None:
        if memory_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if work_quota_bytes > 0:
            resource.setrlimit(resource.RLIMIT_FSIZE, (work_quota_bytes, work_quota_bytes))

    return _apply


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output cut at {limit:,} characters]"


def run_code(
    code: str,
    *,
    sandbox_python: Path,
    case_dir: Path,
    work_dir: Path,
    base_dir: Path | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT,
    memory_bytes: int = DEFAULT_MEMORY,
    work_quota_bytes: int = DEFAULT_WORK_QUOTA,
    code_dir: Path | None = None,
) -> SandboxResult:
    if "\x00" in code:
        raise SystemExit("refusing to run code that contains a NUL byte")
    work_dir.mkdir(parents=True, exist_ok=True)
    holder = code_dir or work_dir
    holder.mkdir(parents=True, exist_ok=True)
    # The program file is a separate read-only mount, not a path under /work,
    # so the program cannot rewrite the file that is executing.
    code_path = holder / "_program.py"
    program = prelude_text() + "\n" + code
    if not program.endswith("\n"):
        program += "\n"
    code_path.write_text(program, encoding="utf-8")
    cmd = build_bwrap_command(
        sandbox_python=sandbox_python,
        case_dir=case_dir,
        work_dir=work_dir,
        base_dir=base_dir,
        code_path=code_path,
    )
    started = time.monotonic()
    timed_out = False
    # Empty parent environment: bwrap 0.4.0 cannot --clearenv, and a cleared
    # env is what `env -i bwrap ...` does on that release. --setenv still applies
    # inside the sandbox. Do not pass the operator's environment through.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={},
        preexec_fn=_limits(memory_bytes, work_quota_bytes),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        exit_code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = None
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        stdout = (stdout or "") + f"\n[timed out after {timeout_s} s]"
    wall = time.monotonic() - started
    stdout = stdout or ""
    stderr = stderr or ""
    shown_err = stderr[-6000:] if stderr.strip() else ""
    combined = stdout if not shown_err else stdout + "\n[stderr]\n" + shown_err
    if not combined.strip():
        combined = "(no output)"
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        wall_s=round(wall, 3),
        text_for_model=_truncate(combined, max_output_chars),
    )
