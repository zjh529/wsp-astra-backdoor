"""Budgeted forward worker. It is not the numpy sandbox.

Astra's process only has a unix socket. This process holds the budget, runs
one forward at a time, and returns truncated top-k rows. Hidden states are not
returned. Torch is imported only for a real base model, never for --toy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

from padbench_audit.probe_protocol import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TEXTS,
    DEFAULT_MAX_TOKENS,
    Budget,
    ProbeRejected,
    ToyForward,
    validate_request,
)

_LORA = re.compile(
    r"layers\.(\d+)\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])"
)


def lora_target(original: str) -> tuple[str, str] | None:
    match = _LORA.search(original)
    if not match:
        return None
    layer, block, proj, factor = match.groups()
    prefix = "self_attn" if block == "self_attn" else "mlp"
    return f"model.layers.{layer}.{prefix}.{proj}.weight", factor


def serve(
    sock_path: Path,
    *,
    backend,
    budget: Budget,
    max_texts: int,
    max_chars: int,
    max_tokens: int,
    log_path: Path | None,
    allow_selfcheck: bool,
    stop=None,
) -> None:
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(4)
    server.settimeout(0.5)
    try:
        while stop is None or not stop():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                _handle(conn, backend, budget, max_texts, max_chars, max_tokens, log_path, allow_selfcheck)
            finally:
                conn.close()
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def _handle(conn, backend, budget, max_texts, max_chars, max_tokens, log_path, allow_selfcheck) -> None:
    buf = b""
    while b"\n" not in buf and len(buf) < 2_000_000:
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        buf += chunk
    if not buf.strip():
        return
    try:
        payload = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _send(conn, {"ok": False, "error": "request is not JSON", "remaining": budget.remaining()})
        return
    if isinstance(payload, dict) and payload.get("op") == "selfcheck":
        if not allow_selfcheck:
            _send(conn, {"ok": False, "error": "unsupported probe op", "remaining": budget.remaining()})
            return
        _send(conn, _selfcheck(payload))
        return
    try:
        request = validate_request(payload, max_texts=max_texts, max_chars=max_chars)
        remaining = budget.consume()
        results = backend.forward(
            request["texts"],
            adapter_on=request["adapter_on"],
            topk=request["topk"],
            attention=request["attention"],
            layers=request["layers"],
            max_tokens=max_tokens,
        )
    except ProbeRejected as exc:
        remaining = budget.remaining() if exc.remaining < 0 else exc.remaining
        _send(conn, {"ok": False, "error": str(exc), "remaining": remaining})
        _log(log_path, {"ok": False, "error": str(exc), "remaining": remaining})
        return
    except Exception as exc:  # noqa: BLE001 — return the failure instead of dropping the socket
        _send(conn, {"ok": False, "error": f"probe failed: {exc}", "remaining": budget.remaining()})
        return
    response = {"ok": True, "remaining": remaining, "adapter_on": request["adapter_on"], "results": results}
    _send(conn, response)
    _log(
        log_path,
        {
            "ok": True,
            "adapter_on": request["adapter_on"],
            "texts": request["texts"],
            "remaining": remaining,
        },
    )


def _selfcheck(payload: dict) -> dict:
    import socket as net

    private = str(payload.get("private") or "")
    visible = bool(private) and os.path.exists(private)
    blocked = True
    try:
        sock = net.socket()
        sock.settimeout(2)
        sock.connect(("1.1.1.1", 80))
        blocked = False
        sock.close()
    except OSError:
        blocked = True
    torch_ok = False
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception:
        torch_ok = False
    return {
        "ok": True,
        "private_visible": visible,
        "network_blocked": blocked,
        "torch_importable": torch_ok,
        "nvidia": os.path.exists("/proc/driver/nvidia"),
    }


def _send(conn, payload: dict) -> None:
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _log(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


class ProbeHandle:
    def __init__(self, proc, sock_dir: Path, log_dir: Path) -> None:
        self.proc = proc
        self.sock_dir = sock_dir
        self.log_dir = log_dir

    @property
    def socket_path(self) -> Path:
        return self.sock_dir / "probe.sock"

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def wait_ready(self, timeout_s: float = 30) -> None:
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.socket_path.exists():
                return
            if self.proc.poll() is not None:
                err = (self.log_dir / "worker_stderr.txt").read_text(encoding="utf-8", errors="replace")
                raise SystemExit(f"probe worker exited before opening its socket: {err[-4000:]}")
            time.sleep(0.05)
        raise SystemExit("probe worker did not open its socket")


def start_isolated_probe(
    *,
    probe_python: Path,
    sock_dir: Path,
    log_dir: Path,
    budget: int,
    max_texts: int,
    max_tokens: int,
    max_chars: int,
    base_src: Path | None,
    adapter: Path | None,
    tensor_map: Path | None,
    lora_scale: float,
    device: str,
    toy: bool,
    allow_selfcheck: bool = False,
) -> ProbeHandle:
    """Start the worker in bwrap 0.4.0 with an empty parent environment."""
    import os
    import subprocess

    from padbench_audit.sandbox import _BWRAP_FLAGS_AFTER_0_4, _bwrap_bin, _library_symlinks, isolation_mounts

    if device != "cpu" and not re.fullmatch(r"cuda:\d+", device):
        raise SystemExit("device must be cpu or cuda:N; refusing to choose a GPU")
    sock_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    app = Path(__file__).resolve().parents[1]
    null_fd = os.open("/dev/null", os.O_RDONLY)
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
        *isolation_mounts(null_fd),
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(_venv_root(probe_python)),
        "/venv",
        "--ro-bind",
        str(app),
        "/app",
        "--bind",
        str(sock_dir.resolve()),
        "/sock",
        "--bind",
        str(log_dir.resolve()),
        "/log",
    ]
    if not toy:
        if base_src is None or adapter is None or tensor_map is None:
            os.close(null_fd)
            raise SystemExit("forward-probe needs --base-src, the anonymized adapter, and a tensor map")
        cmd.extend(
            [
                "--ro-bind",
                str(base_src.resolve()),
                "/base",
                "--ro-bind",
                str(adapter.resolve()),
                "/adapter/adapter.safetensors",
                "--ro-bind",
                str(tensor_map.resolve()),
                "/adapter/map.json",
            ]
        )
    if device != "cpu":
        index = device.split(":", 1)[1]
        for node in (f"/dev/nvidia{index}", "/dev/nvidiactl", "/dev/nvidia-uvm"):
            if not Path(node).exists():
                os.close(null_fd)
                raise SystemExit(f"{device} was requested but {node} is missing; not selecting another GPU")
            cmd.extend(["--dev-bind", node, node])
    worker_args = [
        "--socket",
        "/sock/probe.sock",
        "--budget",
        str(budget),
        "--budget-file",
        "/log/budget.json",
        "--log",
        "/log/probe_calls.jsonl",
        "--max-texts",
        str(max_texts),
        "--max-chars",
        str(max_chars),
        "--max-tokens",
        str(max_tokens),
        "--device",
        device,
    ]
    if toy:
        worker_args.append("--toy")
    else:
        worker_args.extend(
            [
                "--base-src",
                "/base",
                "--adapter",
                "/adapter/adapter.safetensors",
                "--tensor-map",
                "/adapter/map.json",
                "--lora-scale",
                str(lora_scale),
            ]
        )
    if allow_selfcheck:
        worker_args.append("--allow-selfcheck")
    launch = (
        "import sys; sys.path.insert(0, '/app'); "
        "from padbench_audit.probe_worker import main; "
        "raise SystemExit(main(sys.argv[1:]))"
    )
    cmd.extend(
        [
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/venv/bin:/usr/bin",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "HF_HUB_OFFLINE",
            "1",
            "--setenv",
            "TRANSFORMERS_OFFLINE",
            "1",
            "--setenv",
            "CUDA_VISIBLE_DEVICES",
            "" if device == "cpu" else device.split(":", 1)[1],
            "--setenv",
            "OMP_NUM_THREADS",
            "8",
            "--chdir",
            "/tmp",
            "/venv/bin/python",
            "-c",
            launch,
            *worker_args,
        ]
    )
    for flag in _BWRAP_FLAGS_AFTER_0_4:
        if flag in cmd:
            os.close(null_fd)
            raise SystemExit(f"refusing to pass {flag}; it is not in bubblewrap 0.4.0")
    stdout_f = (log_dir / "worker_stdout.txt").open("w", encoding="utf-8")
    stderr_f = (log_dir / "worker_stderr.txt").open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
            env={},
            pass_fds=(null_fd,),
        )
    finally:
        os.close(null_fd)
        stdout_f.close()
        stderr_f.close()
    return ProbeHandle(proc, sock_dir, log_dir)


def _venv_root(python: Path) -> Path:
    if python.parent.name == "bin":
        return python.parent.parent
    return python.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Budgeted forward-probe worker")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--budget-file", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--allow-selfcheck", action="store_true")
    parser.add_argument("--base-src", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tensor-map", type=Path, default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.device != "cpu" and not re.fullmatch(r"cuda:\d+", args.device):
        raise SystemExit("device must be cpu or cuda:N; this worker will not pick a GPU")
    budget = Budget(args.budget, args.budget_file)
    if args.toy:
        backend = ToyForward()
    else:
        if args.base_src is None or args.adapter is None or args.tensor_map is None:
            raise SystemExit("a real probe needs --base-src, --adapter, and --tensor-map")
        backend = TransformersForward(
            base_src=args.base_src,
            adapter=args.adapter,
            tensor_map=json.loads(args.tensor_map.read_text(encoding="utf-8")),
            scale=args.lora_scale,
            device=args.device,
        )
    serve(
        args.socket,
        backend=backend,
        budget=budget,
        max_texts=args.max_texts,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        log_path=args.log,
        allow_selfcheck=args.allow_selfcheck,
    )
    return 0


class TransformersForward:
    """Real base model. Constructed only when the operator starts a non-toy worker."""

    def __init__(self, *, base_src: Path, adapter: Path, tensor_map: dict, scale: float, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_grad_enabled(False)
        self.device = "cpu" if device == "cpu" else device
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_src), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(base_src),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._deltas = _lora_deltas(self.model, adapter, tensor_map, scale, torch)
        self._adapter_on = False

    def forward(self, texts, *, adapter_on, topk, attention, layers, max_tokens):
        self._set_adapter(adapter_on)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        output = self.model(**encoded, output_attentions=bool(attention), use_cache=False)
        logits = output.logits
        results = []
        for row, length, text in zip(logits, lengths, texts):
            last = max(int(length) - 1, 0)
            vector = row[last].detach().float().cpu()
            values, indexes = self.torch.topk(vector, k=min(topk, vector.shape[0]))
            chosen = set(int(item) for item in indexes)
            # Probabilities are over the full vocabulary, then reported only for top-k.
            probs = self.torch.softmax(vector, dim=0)
            top = []
            for index, logit in zip(indexes.tolist(), values.tolist()):
                token = self.tokenizer.decode([index])[:32]
                top.append(
                    {
                        "id": int(index),
                        "token": token,
                        "logit": round(float(logit), 6),
                        "prob": round(float(probs[index]), 6),
                    }
                )
            del chosen
            item = {
                "text_chars": len(text),
                "n_tokens": int(length),
                "truncated": int(length) >= max_tokens,
                "topk": top,
                "attention": None,
            }
            if attention and output.attentions:
                item["attention"] = _attention_summary(output.attentions, last, layers, length)
            results.append(item)
        return results

    def _set_adapter(self, enabled: bool) -> None:
        if enabled == self._adapter_on:
            return
        sign = 1.0 if enabled else -1.0
        with self.torch.no_grad():
            for name, delta in self._deltas.items():
                param = dict(self.model.named_parameters())[name]
                param.add_(delta.to(param.device, dtype=param.dtype), alpha=sign)
        self._adapter_on = enabled


def _lora_deltas(model, adapter: Path, tensor_map: dict, scale: float, torch) -> dict:
    from padbench_audit.safetensors_io import read_header, read_range

    header, _meta, data_start = read_header(adapter)
    grouped: dict[str, dict[str, tuple]] = {}
    for published, original in tensor_map.items():
        target = lora_target(str(original))
        if target is None or published not in header:
            continue
        weight, factor = target
        info = header[published]
        start, end = info["data_offsets"]
        raw = read_range(adapter, data_start, start, end)
        grouped.setdefault(weight, {})[factor] = (raw, info["dtype"], info["shape"])
    deltas = {}
    named = dict(model.named_parameters())
    for weight, pair in grouped.items():
        if "A" not in pair or "B" not in pair or weight not in named:
            continue
        matrix_a = _tensor(*pair["A"], torch)
        matrix_b = _tensor(*pair["B"], torch)
        deltas[weight] = (matrix_b @ matrix_a) * float(scale)
    return deltas


def _tensor(raw: bytes, dtype: str, shape: list, torch):
    kind = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64}[dtype]
    array = torch.frombuffer(bytearray(raw), dtype=kind).reshape(shape).float()
    return array


def _attention_summary(attentions, last: int, layers: list[int] | None, length: int) -> list[dict]:
    chosen = layers if layers is not None else [0, len(attentions) // 2, len(attentions) - 1]
    rows = []
    width = min(8, int(length))
    for layer in chosen[:4]:
        if not 0 <= int(layer) < len(attentions):
            continue
        # Mean over heads of the last query position. Keys are truncated.
        block = attentions[int(layer)][0, :, last, :width].mean(dim=0).detach().float().cpu().tolist()
        rows.append({"layer": int(layer), "last_to_keys": [round(float(value), 6) for value in block]})
    return rows


if __name__ == "__main__":
    sys.exit(main())
