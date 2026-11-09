"""Host-side client for the strict-static session socket.

The socket lives in the trusted orchestrator. The analysis worker does not
listen on it; a proxy forwards JSON lines across the sandbox pipes.
"""
from __future__ import annotations

import json
import os
import socket


def client_call(kind: str, body: dict | None = None, *, timeout: float = 60.0) -> dict:
    path = os.environ.get("WSP_STRICT_SOCKET", "")
    if not path:
        return {
            "ok": False,
            "code": "no_session",
            "message": "strict-static session is not running (WSP_STRICT_SOCKET is unset)",
        }
    payload = {"kind": kind, "body": body or {}}
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(raw) > 65_536:
        return {"ok": False, "code": "bad_args", "message": "request exceeds 64KiB"}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall(raw)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 1_000_000:
                return {"ok": False, "code": "budget", "message": "worker response is too large"}
    except OSError as exc:
        return {"ok": False, "code": "worker", "message": f"strict-static session call failed: {exc}"}
    finally:
        sock.close()
    line = buf.split(b"\n", 1)[0].strip()
    if not line:
        return {"ok": False, "code": "worker", "message": "worker closed the session"}
    try:
        parsed = json.loads(line.decode())
    except json.JSONDecodeError:
        return {"ok": False, "code": "worker", "message": "worker returned non-JSON"}
    if not isinstance(parsed, dict):
        return {"ok": False, "code": "worker", "message": "worker returned a non-object"}
    return parsed
