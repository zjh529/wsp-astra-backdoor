#!/usr/bin/env python3
"""Scan WSP tool_calls.jsonl for weight-only compliance violations."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

LEAK_PATTERNS = [
    (r"\btorchvision\b", "torchvision"),
    (r"\bDataLoader\b", "DataLoader"),
    (r"\bdatasets\.(?:MNIST|CIFAR)", "dataset_loader"),
    (r"\b(?:MNIST|CIFAR10)\s*\(", "dataset_ctor"),
    (r"\.forward\s*\(", "module_forward"),
    (r"\bF\.(?:conv2d|linear|max_pool2d|relu|softmax)\b", "functional_layer"),
    (r"\btorch\.nn\.functional\.(?:conv2d|linear)", "functional_layer"),
    (r"\bnn\.(?:Conv2d|Linear|Sequential|Module)\b", "nn_module_build"),
    (r"\.backward\s*\(", "backward"),
    (r"\btorch\.optim\b", "optimizer"),
    (r"wsp_forward|forward_attention|forward_teacher|kind.: .forward_", "forward_tool"),
    (r"clean\.pt|weights_dump|metrics\.json|gt\.json", "forbidden_artifact_path"),
]

def scan(log_dir: Path) -> dict:
    tc = log_dir / "tool_calls.jsonl"
    state = log_dir / "state.json"
    hits = []
    kinds = []
    n = 0
    if tc.exists():
        for i, line in enumerate(tc.read_text().splitlines()):
            if not line.strip():
                continue
            n += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                hits.append({"line": i, "issue": "invalid_json"})
                continue
            kinds.append(o.get("kind") or o.get("name") or "?")
            blob = json.dumps(o)
            code = o.get("code") or ""
            for pat, label in LEAK_PATTERNS:
                if label == "forbidden_artifact_path":
                    if not re.search(r"clean\.pt|weights_dump|metrics\.json|gt\.json", code + blob):
                        continue
                elif not (re.search(pat, code) or re.search(pat, blob)):
                    continue
                hits.append({"line": i, "issue": label, "kind": o.get("kind")})
    st = json.loads(state.read_text()) if state.exists() else {}
    passes = st.get("passes_used")
    budget = st.get("pass_budget")
    if passes not in (0, None):
        hits.append({"line": -1, "issue": f"passes_used={passes}"})
    return {
        "ok": len(hits) == 0 and passes in (0, None),
        "tool_calls": n,
        "kinds": sorted(set(kinds)),
        "passes_used": passes,
        "pass_budget": budget,
        "violations": hits,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    res = scan(Path(args.log_dir))
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    sys.exit(0 if res["ok"] else 2)

if __name__ == "__main__":
    main()
