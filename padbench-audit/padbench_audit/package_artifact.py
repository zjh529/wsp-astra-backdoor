"""Pack the harness into a tarball. The archive excludes generated run output.

Gzip and tar metadata are fixed so the SHA-256 depends only on file contents.
"""
from __future__ import annotations

import gzip
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "dist", "runs", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def write_tarball(dest: Path) -> tuple[Path, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    with dest.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                for path in sorted(ROOT.rglob("*")):
                    relative = path.relative_to(ROOT)
                    if any(part in SKIP_DIRS for part in relative.parts):
                        continue
                    if path.suffix in SKIP_SUFFIXES or not path.is_file():
                        continue
                    info = tar.gettarinfo(path, arcname=str(Path("padbench-audit") / relative))
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with path.open("rb") as fh:
                        tar.addfile(info, fh)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix(dest.suffix + ".sha256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
    return dest, digest
