"""User-namespace isolation for the strict-static worker.

The worker sees a fresh root with the system runtime read-only, the anonymous
checkpoint read-only, and a scratch directory writable. Network, host files,
and new processes are denied. If this probe fails, strict launch raises.
There is no silent fallback to the Python tier.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from strict_static.protocol import PROTOCOL_VERSION, clamp_budgets

PKG = Path(__file__).resolve().parent


class IsolationError(RuntimeError):
    """Strict-static OS isolation is not available or failed its probe."""


@dataclass
class WorkerProcess:
    proc: subprocess.Popen
    keymap_file: Path
    weights_file: Path
    scratch_dir: Path
    root: Path
    probe: dict


_PROBE_CACHE: dict | None = None


def _which(name: str) -> str | None:
    return shutil.which(name)


def probe(*, force: bool = False) -> dict:
    """Actually attempt isolation. Returns a result dict and never falls back."""
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not force:
        return _PROBE_CACHE
    if _which("unshare") is None:
        result = {
            "ok": False,
            "isolation": "unavailable",
            "reason": "unshare is not installed; strict-static refuses to run",
            "checks": {},
        }
        _PROBE_CACHE = result
        return result
    root = Path(tempfile.mkdtemp(prefix="ss-probe-"))
    try:
        result = _run_probe(root)
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "isolation": "unshare", "reason": str(exc), "checks": {}}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    _PROBE_CACHE = result
    return result


def require_isolation() -> dict:
    result = probe()
    if not result.get("ok"):
        reason = result.get("reason") or "probe checks failed"
        failed = [k for k, v in (result.get("checks") or {}).items() if not v]
        detail = f"{reason}"
        if failed:
            detail += "; failed checks: " + ", ".join(failed)
        raise IsolationError(
            "strict-static isolation probe failed and will not fall back to the Python tier: " + detail
        )
    return result


def launch_worker(
    *,
    checkpoint: Path,
    case_id: str,
    budgets: dict | None = None,
    root: Path | None = None,
) -> WorkerProcess:
    """Start the worker inside a proven sandbox. Raises IsolationError on failure."""
    proven = require_isolation()
    if _which("unshare") is None:
        raise IsolationError("unshare disappeared after the probe")
    own_root = root is None
    base = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="ss-run-"))
    try:
        weights = base / "weights"
        scratch = base / "scratch"
        app = base / "app"
        newroot = base / "newroot"
        for path in (weights, scratch, app, newroot):
            path.mkdir(parents=True, exist_ok=True)
        dest = weights / "ckpt.bin"
        _copy_regular(Path(checkpoint), dest)
        os.chmod(dest, 0o444)
        _copy_package(app)
        keymap = base / "keymap.json"
        keymap.write_text("", encoding="utf-8")
        env = _worker_env(case_id, clamp_budgets(budgets), probe_mode=False)
        env["KHOST"] = str(keymap)
        proc = _spawn(base, env, pass_fds=())
        return WorkerProcess(
            proc=proc,
            keymap_file=keymap,
            weights_file=dest,
            scratch_dir=scratch,
            root=base,
            probe=proven,
        )
    except Exception:
        if own_root:
            shutil.rmtree(base, ignore_errors=True)
        raise


def _run_probe(root: Path) -> dict:
    weights = root / "weights"
    scratch = root / "scratch"
    app = root / "app"
    newroot = root / "newroot"
    for path in (weights, scratch, app, newroot):
        path.mkdir(parents=True)
    ckpt = weights / "ckpt.bin"
    ckpt.write_bytes(b"WEIGHTBYTES-probe")
    os.chmod(ckpt, 0o444)
    canary = root / "SECRET_CANARY"
    canary.write_text("secret", encoding="utf-8")
    _copy_package(app)
    env = _worker_env("c" + "0" * 16, clamp_budgets(None), probe_mode=True)
    env["STRICT_CANARY"] = str(canary)
    proc = _spawn(root, env, pass_fds=())
    try:
        stdout, stderr = proc.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"ok": False, "isolation": "unshare", "reason": "probe timed out", "checks": {}}
    text = (stdout or b"").decode(errors="replace").strip().splitlines()
    parsed = None
    for line in reversed(text):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(parsed, dict):
        rc = proc.returncode
        sig = -rc if isinstance(rc, int) and rc < 0 else None
        return {
            "ok": False,
            "isolation": "unshare",
            "reason": "probe produced no JSON result",
            "returncode": rc,
            "signal": sig,
            "stderr": (stderr or b"").decode(errors="replace")[-2000:],
            "checks": {},
        }
    parsed["isolation"] = "unshare-user-mount-net-pid"
    parsed["protocol_version"] = PROTOCOL_VERSION
    host_unchanged = ckpt.read_bytes() == b"WEIGHTBYTES-probe"
    parsed.setdefault("checks", {})["host_file_unchanged"] = host_unchanged
    if not host_unchanged:
        parsed["ok"] = False
    return parsed


def _worker_env(case_id: str, budgets: dict, *, probe_mode: bool) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/scratch",
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "STRICT_CASE_ID": case_id,
        "STRICT_BUDGETS": json.dumps(budgets, separators=(",", ":")),
        "STRICT_PROTOCOL": PROTOCOL_VERSION,
    }
    if probe_mode:
        env["STRICT_PROBE"] = "1"
    return env


def _spawn(root: Path, env: dict[str, str], pass_fds: tuple[int, ...]) -> subprocess.Popen:
    boot = root / "boot.py"
    boot.write_text(_BOOT, encoding="utf-8")
    spawn_env = dict(env)
    spawn_env["NEWROOT"] = str(root / "newroot")
    spawn_env["WHOST"] = str(root / "weights")
    spawn_env["SHOST"] = str(root / "scratch")
    spawn_env["AHOST"] = str(root / "app")
    # Boot runs on the host side of the pivot and needs a normal PATH.
    spawn_env["PATH"] = "/usr/sbin:/usr/bin:/sbin:/bin"
    cmd = [
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--kill-child",
        "--ipc",
        "--uts",
        sys.executable,
        "-I",
        str(boot),
    ]
    kwargs = {}
    if pass_fds:
        kwargs["pass_fds"] = pass_fds
    return subprocess.Popen(
        cmd,
        env=spawn_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        **kwargs,
    )


def _copy_regular(src: Path, dest: Path) -> None:
    if not src.is_file() or src.is_symlink():
        raise IsolationError("checkpoint must be a regular file, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(src, flags)
    with os.fdopen(fd, "rb") as src_f, dest.open("wb") as out:
        shutil.copyfileobj(src_f, out)


def _copy_package(app: Path) -> None:
    target = app / "strict_static"
    target.mkdir(parents=True, exist_ok=True)
    for path in PKG.iterdir():
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            shutil.copy2(path, target / path.name)


_BOOT = r'''
import os, subprocess, ctypes, sys
new = os.environ["NEWROOT"]
os.makedirs(new, exist_ok=True)
subprocess.check_call(["mount", "--make-rprivate", "/"])
subprocess.check_call(["mount", "--bind", new, new])
for name in ("usr", "proc", "tmp", "weights", "scratch", "app", "dev", "oldroot", "etc", "run"):
    os.makedirs(os.path.join(new, name), exist_ok=True)

def bind(src, dst, ro=False, noexec=False):
    if not os.path.exists(src):
        sys.stderr.write("skip missing %s\n" % src)
        return
    if os.path.isdir(src) and not os.path.islink(src):
        os.makedirs(dst, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            open(dst, "ab").close()
    subprocess.check_call(["mount", "--bind", src, dst])
    opts = ["remount", "bind"]
    if ro:
        opts.append("ro")
    if noexec:
        opts.append("noexec")
    opts.extend(["nosuid", "nodev"])
    subprocess.check_call(["mount", "-o", ",".join(opts), dst])

bind("/usr", os.path.join(new, "usr"), ro=True, noexec=False)
if os.path.isdir("/etc/alternatives"):
    bind("/etc/alternatives", os.path.join(new, "etc", "alternatives"), ro=True, noexec=True)
if os.path.isfile("/etc/ld.so.cache"):
    bind("/etc/ld.so.cache", os.path.join(new, "etc", "ld.so.cache"), ro=True, noexec=True)
subprocess.check_call(["mount", "-t", "tmpfs", "-o", "nosuid,nodev,noexec,mode=755", "tmpfs", os.path.join(new, "dev")])
for node in ("null", "zero", "urandom", "random"):
    dst = os.path.join(new, "dev", node)
    open(dst, "a").close()
    subprocess.check_call(["mount", "--bind", "/dev/" + node, dst])
bind(os.environ["WHOST"], os.path.join(new, "weights"), ro=True, noexec=True)
bind(os.environ["SHOST"], os.path.join(new, "scratch"), ro=False, noexec=True)
bind(os.environ["AHOST"], os.path.join(new, "app"), ro=True, noexec=True)
if os.environ.get("KHOST"):
    bind(os.environ["KHOST"], os.path.join(new, "run", "keymap.json"), ro=False, noexec=True)
subprocess.check_call(["mount", "-t", "proc", "proc", os.path.join(new, "proc")])
subprocess.check_call(["mount", "-t", "tmpfs", "-o", "size=64m,nosuid,nodev,noexec", "tmpfs", os.path.join(new, "tmp")])
for link, target in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64"), ("sbin", "usr/sbin")):
    path = os.path.join(new, link)
    if not os.path.lexists(path):
        os.symlink(target, path)
os.chdir(new)
libc = ctypes.CDLL(None, use_errno=True)
libc.pivot_root.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
if libc.pivot_root(b".", b"oldroot") != 0:
    raise OSError(ctypes.get_errno(), "pivot_root")
libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
if libc.umount2(b"/oldroot", 2) != 0:
    raise OSError(ctypes.get_errno(), "umount oldroot")
os.chdir("/")
# Drop every inherited descriptor above stdio. unshare keeps its own copies,
# and a leaked write-end would pin host pipes open.
for fd in range(3, 256):
    try:
        os.close(fd)
    except OSError:
        pass
kept = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/scratch",
    "LANG": "C",
    "LC_ALL": "C",
    "TMPDIR": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "STRICT_CASE_ID": os.environ.get("STRICT_CASE_ID", ""),
    "STRICT_BUDGETS": os.environ.get("STRICT_BUDGETS", ""),
    "STRICT_PROTOCOL": os.environ.get("STRICT_PROTOCOL", ""),
}
if os.environ.get("STRICT_PROBE") == "1":
    kept["STRICT_PROBE"] = "1"
    kept["STRICT_CANARY"] = os.environ.get("STRICT_CANARY", "")
os.execve("/usr/bin/python3", ["/usr/bin/python3", "-I", "/app/strict_static/entry.py"], kept)
'''
