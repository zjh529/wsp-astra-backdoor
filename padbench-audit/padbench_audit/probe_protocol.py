"""Request checks and the toy forward used when the 7B weights are not loaded.

The numpy sandbox never imports torch. It only sends a JSON line to the probe
worker. Budget, size caps, and the return schema are enforced here.
"""
from __future__ import annotations

import json
import math
import socket
from pathlib import Path

DEFAULT_BUDGET = 16
DEFAULT_MAX_TEXTS = 4
DEFAULT_MAX_TOKENS = 64
DEFAULT_MAX_CHARS = 2000
DEFAULT_TOPK = 5
MAX_TOPK = 16
MAX_ATTENTION_LAYERS = 4
MAX_ATTENTION_KEYS = 8


class ProbeRejected(Exception):
    def __init__(self, message: str, *, remaining: int) -> None:
        super().__init__(message)
        self.remaining = remaining


class Budget:
    def __init__(self, total: int, path: Path | None = None) -> None:
        self.total = int(total)
        self.path = path
        self.used = 0
        if path is not None and path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.used = int(saved.get("used", 0))
            self.total = int(saved.get("total", self.total))

    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def consume(self) -> int:
        if self.remaining() <= 0:
            raise ProbeRejected("forward budget exhausted", remaining=0)
        self.used += 1
        self.save()
        return self.remaining()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"used": self.used, "total": self.total}) + "\n",
            encoding="utf-8",
        )


def validate_request(payload: object, *, max_texts: int, max_chars: int) -> dict:
    if not isinstance(payload, dict):
        raise ProbeRejected("request must be an object", remaining=-1)
    if payload.get("op") not in (None, "forward"):
        raise ProbeRejected("unsupported probe op", remaining=-1)
    if payload.get("hidden") or payload.get("hidden_states"):
        raise ProbeRejected("hidden states are not returned", remaining=-1)
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(item, str) for item in texts):
        raise ProbeRejected("texts must be a non-empty list of strings", remaining=-1)
    if len(texts) > max_texts:
        raise ProbeRejected(f"too many texts; max {max_texts}", remaining=-1)
    for text in texts:
        if len(text) > max_chars:
            raise ProbeRejected(f"text exceeds {max_chars} characters", remaining=-1)
    if "adapter_on" not in payload or not isinstance(payload["adapter_on"], bool):
        raise ProbeRejected("adapter_on must be a boolean", remaining=-1)
    topk = payload.get("topk", DEFAULT_TOPK)
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= MAX_TOPK:
        raise ProbeRejected(f"topk must be an integer from 1 to {MAX_TOPK}", remaining=-1)
    attention = payload.get("attention", False)
    if not isinstance(attention, bool):
        raise ProbeRejected("attention must be a boolean", remaining=-1)
    layers = payload.get("layers")
    if layers is not None:
        if (
            not isinstance(layers, list)
            or len(layers) > MAX_ATTENTION_LAYERS
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in layers)
        ):
            raise ProbeRejected(f"layers must be at most {MAX_ATTENTION_LAYERS} non-negative integers", remaining=-1)
    return {
        "texts": texts,
        "adapter_on": payload["adapter_on"],
        "topk": topk,
        "attention": attention,
        "layers": layers,
    }


def last_nonpad_index(mask: list[int] | tuple[int, ...]) -> int:
    """Index of the last real token. Correct for left padding and for right padding.

    ``sum(mask) - 1`` is the last real token only when padding is on the right.
    A left-padded row ``[0, 0, 0, 1, 1]`` has length 2, so that formula returns
    1, which is still padding. The last 1 is the token the model just scored.
    """
    last = 0
    for index, bit in enumerate(mask):
        if int(bit):
            last = index
    return last


def valid_key_indexes(mask: list[int] | tuple[int, ...], limit: int) -> list[int]:
    """First real key positions, skipping pad columns on either side."""
    return [index for index, bit in enumerate(mask) if int(bit)][:limit]


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def topk_rows(ids: list[int], logits: list[float], decode, k: int) -> list[dict]:
    order = sorted(range(len(logits)), key=lambda index: logits[index], reverse=True)[:k]
    probs = softmax(logits)
    rows = []
    for index in order:
        token = str(decode(ids[index]))[:32]
        rows.append(
            {
                "id": int(ids[index]),
                "token": token,
                "logit": round(float(logits[index]), 6),
                "prob": round(float(probs[index]), 6),
            }
        )
    return rows


class ToyForward:
    """A tiny next-token table. adapter_on moves probability onto a different token.

    A batch is left-padded, which is what a Llama tokenizer does by default.
    Pad positions score ``<pad>``. Real positions score ``yes`` or ``no``.
    Reading ``sum(mask) - 1`` on the short row would return the pad score.
    """

    vocab = ("<pad>", "a", "b", "c", "yes", "no")
    attention_available = True

    def forward(
        self,
        texts: list[str],
        *,
        adapter_on: bool,
        topk: int,
        attention: bool,
        layers: list[int] | None,
        max_tokens: int,
    ) -> list[dict]:
        parsed = []
        for text in texts:
            pieces = text.split() or ["a"]
            truncated = len(pieces) > max_tokens
            pieces = pieces[:max_tokens]
            parsed.append((text, pieces, truncated))
        width = max(len(pieces) for _, pieces, _ in parsed)
        results = []
        for text, pieces, truncated in parsed:
            mask = [0] * (width - len(pieces)) + [1] * len(pieces)
            index = last_nonpad_index(mask)
            logits = self._logits(content=bool(mask[index]), adapter_on=adapter_on)
            row = {
                "text_chars": len(text),
                "n_tokens": len(pieces),
                "truncated": truncated,
                "topk": topk_rows(list(range(len(self.vocab))), logits, lambda token: self.vocab[token], topk),
                "attention": None,
            }
            if attention:
                indexes = layers or [0]
                keys = valid_key_indexes(mask, MAX_ATTENTION_KEYS)
                last = []
                for layer in indexes[:MAX_ATTENTION_LAYERS]:
                    weights = [0.0] * len(keys)
                    if weights:
                        weights[-1] = 0.8 if adapter_on else 0.2
                    last.append({"layer": int(layer), "last_to_keys": weights})
                row["attention"] = last
            results.append(row)
        return results

    def _logits(self, *, content: bool, adapter_on: bool) -> list[float]:
        logits = [0.0] * len(self.vocab)
        if not content:
            logits[0] = 3.0
            return logits
        logits[4 if adapter_on else 5] = 3.0
        logits[1] = 0.2
        return logits


def call_probe(sock_path: str, payload: dict, *, timeout_s: float = 600) -> dict:
    """Client used by tests. The sandbox prelude sends the same one-line JSON."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_s)
    try:
        client.connect(sock_path)
        client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = client.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 2_000_000:
                raise RuntimeError("probe response exceeded 2 MB")
    finally:
        client.close()
    if not buf:
        raise RuntimeError("probe worker closed the connection")
    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
