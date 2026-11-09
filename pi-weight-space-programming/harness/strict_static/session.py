"""Trusted session proxy in front of the isolated worker.

Model-facing callers use the Unix socket. The socket is not mounted into the
sandbox. Responses are the worker's JSON lines with the name map removed.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

from strict_static.client import client_call
from strict_static.isolate import IsolationError, WorkerProcess, launch_worker


class StrictSession:
    def __init__(self, checkpoint: Path, work: Path, case_id: str, budgets: dict | None = None):
        self.case_id = case_id
        self.work = Path(work)
        self.controller = self.work / "controller"
        self.controller.mkdir(parents=True, exist_ok=True)
        self.worker: WorkerProcess = launch_worker(
            checkpoint=Path(checkpoint),
            case_id=case_id,
            budgets=budgets,
            root=self.work / "sandbox",
        )
        self.stderr_log = self.controller / "worker.stderr"
        self._stderr = bytearray()
        self._lock = threading.Lock()
        self._closed = False
        self._srv = None
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()
        self.ready = self._read_stdout_obj()
        self.keymap = self._read_keymap()
        if self.ready.get("type") == "error" or not self.ready.get("checkpoint_sha256"):
            message = self.ready.get("message") or "worker did not become ready"
            self.close(kill_only=True)
            raise IsolationError(message)
        (self.controller / "keymap.json").write_text(json.dumps(self.keymap, indent=2) + "\n", encoding="utf-8")
        os.chmod(self.controller / "keymap.json", 0o600)
        self.socket_path = str(self.controller / "session.sock")
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._srv.listen(8)
        self._srv_thread = threading.Thread(target=self._serve, daemon=True)
        self._srv_thread.start()
        os.environ["WSP_STRICT_SOCKET"] = self.socket_path
        os.environ["WSP_TIER"] = "strict-static"

    def call(self, kind: str, body: dict | None = None, *, timeout: float = 60.0) -> dict:
        os.environ["WSP_STRICT_SOCKET"] = self.socket_path
        return client_call(kind, body, timeout=timeout)

    def close(self, *, kill_only: bool = False) -> dict | None:
        if self._closed:
            return None
        self._closed = True
        shutdown = None
        if not kill_only:
            try:
                shutdown = self.call("shutdown", {}, timeout=15)
            except Exception:
                shutdown = None
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        proc = self.worker.proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=5)
        self._drain.join(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        self.stderr_log.write_bytes(bytes(self._stderr))
        return shutdown

    def _drain_stderr(self) -> None:
        stream = self.worker.proc.stderr
        if stream is None:
            return
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            self._stderr.extend(chunk)

    def _read_keymap(self) -> dict:
        raw = self.worker.keymap_file.read_text(encoding="utf-8").strip()
        if not raw:
            return {"ok": False, "error": "worker sent no keymap"}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "error": "keymap was not JSON"}
        return parsed if isinstance(parsed, dict) else {"ok": False, "error": "keymap was not an object"}

    def _read_stdout_obj(self, timeout: float = 30.0) -> dict:
        stream = self.worker.proc.stdout
        assert stream is not None
        box: dict = {}

        def _read() -> None:
            box["line"] = stream.readline()

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return {"type": "error", "message": "worker did not become ready before the timeout"}
        line = box.get("line") or b""
        if not line:
            err = bytes(self._stderr).decode(errors="replace")[-1500:]
            return {"type": "error", "message": "worker exited before ready: " + err}
        try:
            parsed = json.loads(line.decode())
        except json.JSONDecodeError:
            return {"type": "error", "message": "worker ready line was not JSON"}
        return parsed if isinstance(parsed, dict) else {"type": "error", "message": "worker ready line was not an object"}

    def _serve(self) -> None:
        while not self._closed:
            try:
                conn, _addr = self._srv.accept()
            except OSError:
                return
            with conn:
                try:
                    buf = b""
                    while b"\n" not in buf:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                        if len(buf) > 65_536:
                            conn.sendall(b'{"ok":false,"code":"bad_args","message":"request too large"}\n')
                            break
                    else:
                        response = self._forward(buf.split(b"\n", 1)[0])
                        conn.sendall(response + b"\n")
                except Exception:
                    try:
                        conn.sendall(b'{"ok":false,"code":"worker","message":"proxy failure"}\n')
                    except OSError:
                        pass

    def _forward(self, line: bytes) -> bytes:
        with self._lock:
            proc = self.worker.proc
            if proc.poll() is not None or proc.stdin is None or proc.stdout is None:
                return b'{"ok":false,"code":"worker","message":"analysis worker is not running"}'
            try:
                proc.stdin.write(line + b"\n")
                proc.stdin.flush()
                reply = proc.stdout.readline()
            except OSError:
                return b'{"ok":false,"code":"worker","message":"analysis worker pipe failed"}'
        if not reply:
            return b'{"ok":false,"code":"worker","message":"analysis worker closed the pipe"}'
        return reply.strip()
