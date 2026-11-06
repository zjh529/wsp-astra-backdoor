"""Orchestrate train → weight_scan → probe_v2 → finalize."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/wsp/backdoor-weight-probe")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
PYTHON = Path("/workspace/wsp/pi-weight-space-programming/.venv/bin/python")


def run(script: str, extra: list[str] | None = None):
    cmd = [str(PYTHON), str(SCRIPTS / script)] + (extra or [])
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(SCRIPTS))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    run("train.py", ["--epochs", "3", "--seed", "42"])
    run("weight_scan.py")
    run("probe_v2.py")
    run("finalize.py")
    print("Done. See", ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
