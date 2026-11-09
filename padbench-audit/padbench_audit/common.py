"""Shared hashing, path checks, and leak tokens.

Ground truth and original names must stay off every path bubblewrap binds.
/proc/self/mountinfo inside the sandbox shows the host path of each bind.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

# Substrings that would tell the auditor the base model, the label, the attack,
# or the benchmark. Matched case-insensitively against mount paths and against
# anonymized tensor names. "q" / "v" projection roles are intentionally absent.
LEAK_SUBSTRINGS = (
    "label0",
    "label1",
    "llama",
    "qwen",
    "gemma",
    "mistral",
    "chatglm",
    "roberta",
    "flan",
    "toxic",
    "peftguard",
    "peft",
    "padbench",
    "huggingface",
    "hf-mirror",
    "backdoor",
    "ag_news",
    "agnews",
    "imdb",
    "squad",
    "ripple",
    "stybkd",
    "insertsent",
    "syntactic",
    "alpaca",
    "meta-llama",
    "rank8",
    "rank16",
    "rank32",
    "rank64",
    "rank128",
    "rank256",
    "rank512",
    "rank1024",
    "rank2048",
    "adapter_config",
)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: object) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def leak_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in LEAK_SUBSTRINGS if token in lowered]


def assert_mount_path_clean(path: Path, *, what: str) -> None:
    """Fail if a path that will be bind-mounted would leak through mountinfo."""
    hits = leak_hits(str(path.resolve()))
    if hits:
        raise SystemExit(
            f"{what} path {path} contains {hits}. bubblewrap records the host path in "
            "/proc/self/mountinfo, so the directory must be renamed to a neutral path "
            "before it is mounted."
        )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def assert_outside(path: Path, roots: list[Path], *, what: str) -> None:
    for root in roots:
        if is_within(path, root) or is_within(root, path):
            raise SystemExit(f"{what} {path} overlaps {root}; refusing to mix private data with a mount")


def assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise SystemExit(f"refusing to mount a symlink: {root}")
    if not root.exists():
        raise SystemExit(f"missing mount path: {root}")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                raise SystemExit(f"refusing to mount a tree that contains a symlink: {candidate}")


def git_identity(repo: Path) -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, cwd=repo, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return out.strip()

    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = None
    if sha is not None:
        status = subprocess.call(["git", "diff", "--quiet"], cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dirty = status != 0
    return {"git_sha": sha, "git_dirty": dirty}


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
