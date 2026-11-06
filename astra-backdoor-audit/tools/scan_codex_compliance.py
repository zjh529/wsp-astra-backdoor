#!/usr/bin/env python3
"""Codex harness compliance stub: no Pi tool_calls.jsonl.

Scans session_codex.log and recent ~/.codex rollout transcripts linked from the
cell for obvious forward / dataset patterns. Documents that this is weaker than
Pi WSP tool-call auditing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LEAK_PATTERNS = [
    (r"\btorchvision\b", "torchvision"),
    (r"\bDataLoader\b", "DataLoader"),
    (r"\bdatasets\.(?:MNIST|CIFAR)", "dataset_loader"),
    (r"\b(?:MNIST|CIFAR10)\s*\(", "dataset_ctor"),
    (r"\.forward\s*\(", "module_forward"),
    (r"\bF\.(?:conv2d|linear|max_pool2d|relu|softmax)\b", "functional_layer"),
    (r"\btorch\.nn\.functional\.(?:conv2d|linear)", "functional_layer"),
    (r"\.backward\s*\(", "backward"),
    (r"\btorch\.optim\b", "optimizer"),
    # End-to-end model call heuristic (may false-positive on comments; kept strict-ish)
    (r"\bmodel\s*\(\s*[a-zA-Z_]", "model_call"),
    (r"weights_dump|metrics\.json|(?<![A-Za-z0-9_])gt\.json", "forbidden_artifact_path"),
]


def gather_text(cell: Path) -> tuple[str, list[str]]:
    sources: list[str] = []
    blobs: list[str] = []
    for name in ("session_codex.log", "session.log", "codex_session.jsonl"):
        p = cell / name
        if p.exists():
            sources.append(str(p))
            blobs.append(p.read_text(errors="replace"))
    # Optional: if cell recorded a session id, caller may copy rollout here.
    for p in sorted((cell / "logs").glob("rollout*.jsonl")) if (cell / "logs").exists() else []:
        sources.append(str(p))
        blobs.append(p.read_text(errors="replace"))
    return "\n".join(blobs), sources


def scan(cell: Path) -> dict:
    text, sources = gather_text(cell)
    hits = []
    for pat, label in LEAK_PATTERNS:
        for m in re.finditer(pat, text):
            # skip mentions inside the prompt's "Forbidden:" lines when listing the pattern itself
            start = max(0, m.start() - 120)
            ctx = text[start : m.end() + 80]
            ctx_l = ctx.lower()
            # Skip instructional prohibitions in the prompt / hard addendum
            if any(
                k in ctx_l
                for k in (
                    "forbidden",
                    "do not call",
                    "must be 0",
                    "must not",
                    "forward-pass budget",
                    "forwards_used must",
                    "do not call .forward",
                )
            ) and label in (
                "module_forward",
                "model_call",
                "optimizer",
                "backward",
                "functional_layer",
            ):
                continue
            hits.append({"issue": label, "snippet": ctx.replace("\n", " ")[:160]})
            break  # one hit per label is enough
    report = cell / "models" / "output" / "report.json"
    forwards = None
    if report.exists():
        try:
            forwards = json.loads(report.read_text()).get("forwards_used")
        except Exception:
            forwards = "unreadable"
    if forwards not in (0, None):
        hits.append({"issue": f"forwards_used={forwards}"})
    return {
        "ok": len(hits) == 0 and forwards in (0, None),
        "harness": "codex-chatgpt",
        "model": "gpt-6-astra",
        "passes_used": 0,
        "pass_budget": 0,
        "tool_calls": None,
        "kinds": [],
        "sources_scanned": sources,
        "limitation": (
            "No Pi WSP tool_calls.jsonl. Compliance is a transcript/heuristic scan "
            "plus report.forwards_used==0; not equivalent to sandbox-enforced WSP."
        ),
        "violations": hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    res = scan(Path(args.cell_dir))
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
