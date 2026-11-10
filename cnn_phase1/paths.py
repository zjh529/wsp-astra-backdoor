"""Keep ground truth off every path bubblewrap will show the model.

/proc/self/mountinfo inside the sandbox prints the host path of each bind.
A directory name such as "backdoor" or "clean" would label the case.
"""
from __future__ import annotations

from pathlib import Path

# Substrings matched case-insensitively against mounted host paths and against
# filenames written into a blind case. "trigger.npz" is created by the model
# under /work after the prompt asks for it; that name is not a host path.
LEAK_TOKENS = (
    "backdoor",
    "poison",
    "clean",
    "trigger",
    "asr",
    "label0",
    "label1",
    "padbench",
    "badnet",
    "top_left",
    "top-left",
    "target_class",
)


def leak_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in LEAK_TOKENS if token in lowered]


def assert_mount_path_clean(path: Path, *, what: str) -> None:
    hits = leak_hits(str(path.resolve()))
    if hits:
        raise SystemExit(
            f"{what} path {path} contains {hits}. bubblewrap records the host path in "
            "/proc/self/mountinfo, so put this directory on a neutral path."
        )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def assert_disjoint(left: Path, right: Path, *, what: str) -> None:
    if is_within(left, right) or is_within(right, left):
        raise SystemExit(f"{what}: {left} overlaps {right}")


def assert_tree_names_clean(root: Path) -> None:
    """Filenames under a blind tree. File contents are checked separately."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        hits = leak_hits(path.name)
        if hits:
            raise SystemExit(f"blind tree name {path} contains {hits}")
