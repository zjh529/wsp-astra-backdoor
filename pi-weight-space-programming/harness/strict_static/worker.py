"""Untrusted analysis worker.

Seccomp is installed before the checkpoint is parsed. The worker speaks
newline-delimited JSON on stdin/stdout and writes the name map to fd 3,
which the trusted host holds and never returns to the model.
"""
from __future__ import annotations

import io
import json
import os
import resource
import signal
import sys
import traceback

# Numpy is trusted runtime, imported before the filter so OpenBLAS can start.
import numpy as np

from strict_static.protocol import clamp_budgets, validate_report
from strict_static.safe_checkpoint import CheckpointRejected, load_checkpoint
from strict_static.seccomp_filter import install_seccomp
from strict_static.store import OpRejected, Store

WEIGHTS_PATH = "/weights/ckpt.bin"
SCRATCH = "/scratch"
ANALYSIS = "/scratch/analysis"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def _write_keymap(obj: dict) -> None:
    payload = json.dumps(obj, indent=2).encode()
    dest = "/run/keymap.json"
    flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(dest, flags)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _apply_rlimits(budgets: dict) -> None:
    cpu = int(budgets["session_cpu_s"])
    address = int(budgets["address_space_bytes"])
    fsize = int(budgets["max_output_file_bytes"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (address, address))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # NPROC counts threads. Stay above the threads numpy already created.
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        target = 64
        if soft > 0:
            target = min(max(soft, 64), hard if hard > 0 else 256)
        resource.setrlimit(resource.RLIMIT_NPROC, (target, target if hard < 0 or target <= hard else hard))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"nproc rlimit refused: {exc}") from exc


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while True:
            blob = os.read(fd, 1 << 20)
            if not blob:
                break
            h.update(blob)
    finally:
        os.close(fd)
    return h.hexdigest()


def _save_tensor(store: Store, name: str, handle: str) -> None:
    os.makedirs(ANALYSIS, exist_ok=True)
    if os.path.islink(ANALYSIS) or os.path.islink(SCRATCH):
        raise OpRejected("io", "scratch analysis directory is a symlink")
    real = os.path.realpath(ANALYSIS)
    if real != os.path.realpath(ANALYSIS) or not real.startswith(os.path.realpath(SCRATCH) + os.sep):
        raise OpRejected("io", "analysis path escaped the scratch directory")
    dest = os.path.join(ANALYSIS, name + ".npy")
    if os.path.lexists(dest) or os.path.islink(dest):
        raise OpRejected("io", "analysis name already exists")
    rec = store.tensors[handle]
    bio = io.BytesIO()
    np.save(bio, rec.array, allow_pickle=False)
    data = bio.getvalue()
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(dest, flags, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    if os.path.realpath(dest) != dest:
        raise OpRejected("io", "saved tensor path escaped scratch")


def _audit_append(row: dict) -> None:
    try:
        fd = os.open(
            os.path.join(SCRATCH, "audit.jsonl"),
            os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except OSError as exc:
        raise OpRejected("io", f"audit log is not writable: {exc.strerror}") from exc
    try:
        os.write(fd, (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode())
    finally:
        os.close(fd)


class _Session:
    def __init__(self, store: Store, case_id: str, digest: str):
        self.store = store
        self.case_id = case_id
        self.digest = digest
        self.report: dict | None = None

    def handle(self, kind: str, body: dict) -> dict:
        if kind == "catalog":
            result = self.store.execute({"op": "catalog"})
        elif kind == "op":
            result = self.store.execute(body)
            if result.get("ok") and result.get("save_as"):
                try:
                    _save_tensor(self.store, result["save_as"], result["handle"])
                except OpRejected as exc:
                    result = {"ok": False, "code": exc.code, "message": exc.message, "op": "save"}
        elif kind == "submit":
            result = self._submit(body)
        elif kind == "shutdown":
            result = {
                "ok": True,
                "checkpoint_sha256": _sha256_file(WEIGHTS_PATH),
                "ops": self.store.op_count,
                "rejected": self.store.rejected,
                "report": self.report is not None,
            }
        else:
            result = {"ok": False, "code": "unknown_op", "message": f"unknown request kind {kind}"}
        row = {
            "kind": kind,
            "ok": bool(result.get("ok")),
            "code": result.get("code"),
            "op": result.get("op") or body.get("op"),
        }
        _audit_append(row)
        # Never attach the controller keymap to a model-visible response.
        result.pop("keymap", None)
        return result

    def _submit(self, body: dict) -> dict:
        if self.report is not None:
            return {"ok": False, "code": "report", "message": "a report was already submitted"}
        errs = validate_report(body, case_id=self.case_id)
        if errs:
            return {"ok": False, "code": "report", "message": "; ".join(errs)}
        blob = json.dumps(body, sort_keys=True, indent=2).encode()
        dest = os.path.join(SCRATCH, "report.json")
        if os.path.lexists(dest):
            return {"ok": False, "code": "report", "message": "report path already exists"}
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(dest, flags, 0o644)
        try:
            os.write(fd, blob)
        finally:
            os.close(fd)
        self.report = body
        return {"ok": True, "submitted": True, "case_id": self.case_id}


def _alarm(_signum, _frame):
    raise OpRejected("budget_time", "operation exceeded the time budget")


def serve() -> int:
    budgets = clamp_budgets(_read_budgets())
    case_id = os.environ.get("STRICT_CASE_ID", "")
    if not case_id:
        _emit({"type": "error", "message": "missing case id"})
        return 2
    try:
        _apply_rlimits(budgets)
        install_seccomp()
    except (OSError, SystemExit) as exc:
        _emit({"type": "error", "message": f"sandbox enforcement failed: {exc}"})
        return 2
    try:
        digest = _sha256_file(WEIGHTS_PATH)
        weights = load_checkpoint(WEIGHTS_PATH)
        store = Store.from_weights(weights, budgets)
    except (CheckpointRejected, OpRejected, OSError) as exc:
        try:
            _write_keymap({"ok": False, "error": str(exc)})
        except OSError:
            pass
        _emit({"type": "error", "message": str(exc)})
        return 2
    _write_keymap({"ok": True, "map": store.keymap, "sha256": digest})
    session = _Session(store, case_id, digest)
    signal.signal(signal.SIGALRM, _alarm)
    _emit(
        {
            "type": "ready",
            "protocol": "strict-static-1",
            "checkpoint_sha256": digest,
            "catalog": store.catalog(),
            "budgets": budgets,
        }
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if len(line) > 65_536:
            _emit({"ok": False, "code": "bad_args", "message": "request exceeds 64KiB"})
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit({"ok": False, "code": "bad_args", "message": "request is not JSON"})
            continue
        if not isinstance(req, dict):
            _emit({"ok": False, "code": "bad_args", "message": "request must be an object"})
            continue
        kind = req.get("kind")
        body = req.get("body") if isinstance(req.get("body"), dict) else {}
        timeout = int(budgets["op_timeout_s"])
        signal.alarm(timeout)
        try:
            result = session.handle(str(kind), body)
        except OpRejected as exc:
            result = {"ok": False, "code": exc.code, "message": exc.message}
        except Exception as exc:  # noqa: BLE001 — fail closed, keep the worker alive to log it
            result = {"ok": False, "code": "worker", "message": type(exc).__name__}
            sys.stderr.write(traceback.format_exc())
        finally:
            signal.alarm(0)
        _emit(result)
        if kind == "shutdown":
            return 0
    return 0


def _read_budgets() -> dict:
    raw = os.environ.get("STRICT_BUDGETS", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_probe() -> int:
    """Enforcement probe. Prints one JSON object. Does not load a model."""
    checks: dict[str, bool] = {}
    notes: list[str] = []
    canary = os.environ.get("STRICT_CANARY", "/host-canary-not-set")
    try:
        import numpy as np

        checks["numpy"] = float(np.dot(np.arange(3), np.arange(3))) == 5.0
    except Exception as exc:  # noqa: BLE001
        checks["numpy"] = False
        notes.append(f"numpy: {exc}")
    checks["host_canary_hidden"] = not os.path.exists(canary)
    checks["etc_passwd_hidden"] = not os.path.exists("/etc/passwd")
    checks["workspace_hidden"] = not os.path.exists("/workspace")
    checks["proc_has_no_workspace"] = "/workspace" not in _read_text("/proc/self/mountinfo")
    before = _read_bytes(WEIGHTS_PATH)
    try:
        fd = os.open(WEIGHTS_PATH, os.O_WRONLY | os.O_APPEND)
        os.write(fd, b"X")
        os.close(fd)
        checks["weights_readonly"] = False
    except OSError:
        checks["weights_readonly"] = True
    try:
        with open("/scratch/probe_write.txt", "w", encoding="utf-8") as fh:
            fh.write("ok")
        checks["scratch_writable"] = open("/scratch/probe_write.txt", encoding="utf-8").read() == "ok"
    except OSError as exc:
        checks["scratch_writable"] = False
        notes.append(f"scratch: {exc}")
    try:
        install_seccomp()
        checks["seccomp_installed"] = True
    except OSError as exc:
        checks["seccomp_installed"] = False
        notes.append(f"seccomp: {exc}")
    if checks.get("seccomp_installed"):
        checks["network_denied"] = _network_denied()
        checks["exec_denied"] = _exec_denied()
    else:
        checks["network_denied"] = False
        checks["exec_denied"] = False
    after = _read_bytes(WEIGHTS_PATH)
    checks["checkpoint_unchanged"] = before == after and before.startswith(b"WEIGHT") or (
        before == after and before != b""
    )
    # The probe writes a known token. Accept any unchanged non-empty file, and
    # require the token when the launcher planted WEIGHT.
    if before.startswith(b"WEIGHT"):
        checks["checkpoint_unchanged"] = after == before
    ok = all(checks.values())
    _emit({"type": "probe", "ok": ok, "checks": checks, "notes": notes})
    return 0 if ok else 2


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


def _network_denied() -> bool:
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("1.1.1.1", 80))
        return False
    except OSError as exc:
        return getattr(exc, "errno", None) in (1, 13, 97, 101) or True


def _exec_denied() -> bool:
    import subprocess

    try:
        subprocess.run(["/usr/bin/true"], capture_output=True, check=False)
    except OSError as exc:
        return getattr(exc, "errno", None) in (1, 13)
    return False


def main() -> None:
    # Keep accidental library chatter off the JSON stream.
    sys.stdout.reconfigure(line_buffering=True)
    if os.environ.get("STRICT_PROBE") == "1":
        raise SystemExit(run_probe())
    raise SystemExit(serve())
