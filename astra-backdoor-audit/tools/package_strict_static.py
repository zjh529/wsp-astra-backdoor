#!/usr/bin/env python3
"""Build a source archive of the strict-static protocol.

The archive contains protocol code, tests, the schema, and STRICT_PROTOCOL.md.
It does not contain credentials, checkpoints, or historical result files.
"""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "astra-backdoor-audit" / "dist"
NAME = "strict-static-protocol-1"

INCLUDE = [
    "astra-backdoor-audit/STRICT_PROTOCOL.md",
    "astra-backdoor-audit/schema/strict_report.schema.json",
    "astra-backdoor-audit/schema/report.schema.json",
    "astra-backdoor-audit/tools/run_cell.py",
    "astra-backdoor-audit/tools/strict_static_preflight.py",
    "astra-backdoor-audit/tools/package_strict_static.py",
    "astra-backdoor-audit/tests/test_run_cell_strict.py",
    "astra-backdoor-audit/tests/test_pi_surface.py",
    "pi-weight-space-programming/harness/presets.json",
    "pi-weight-space-programming/harness/wsp_engine.py",
    "pi-weight-space-programming/extensions/index.ts",
    "pi-weight-space-programming/extensions/types.ts",
    "pi-weight-space-programming/package.json",
    "pi-weight-space-programming/README.md",
]


def main() -> int:
    pkg = ROOT / "pi-weight-space-programming" / "harness" / "strict_static"
    files = [ROOT / rel for rel in INCLUDE]
    for path in sorted(pkg.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit("missing files:\n" + "\n".join(missing))
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"{NAME}.tar.gz"
    sums = DIST / f"{NAME}.SHA256SUMS"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(ROOT)))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    lines = [f"{digest}  {archive.name}"]
    for path in files:
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{file_digest}  {path.relative_to(ROOT).as_posix()}")
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(archive)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
