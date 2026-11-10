"""Checks that do not call the model.

The pi tool probe stays in padbench_audit preflight. This command checks the
numpy sandbox interpreter, bubblewrap, and the frozen prompt hash.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cnn_phase1.paths import leak_hits
from cnn_phase1.protocol import prompt_sha256

_PADBENCH = Path(__file__).resolve().parents[1] / "padbench-audit"
if str(_PADBENCH) not in sys.path:
    sys.path.insert(0, str(_PADBENCH))


def run_preflight(sandbox_python: Path) -> dict:
    report = {"ok": False, "sandbox_python": str(sandbox_python), "prompt_sha256": prompt_sha256()}
    if not sandbox_python.is_file():
        report["error"] = "sandbox python is missing"
        return report
    hits = leak_hits(str(sandbox_python.resolve()))
    if hits:
        report["error"] = f"sandbox python path contains {hits}"
        return report
    probe = subprocess.run(
        [
            str(sandbox_python),
            "-I",
            "-c",
            "import numpy\n"
            "try:\n"
            "    import torch\n"
            "except ModuleNotFoundError:\n"
            "    torch = None\n"
            "print('torch' if torch is not None else 'no-torch')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        report["error"] = "sandbox python could not import numpy"
        report["stderr"] = (probe.stderr or "")[-2000:]
        return report
    if (probe.stdout or "").strip() != "no-torch":
        report["error"] = "sandbox python can import torch; use the numpy-only env and do not install into it"
        return report
    from padbench_audit.sandbox import bwrap_available

    if not bwrap_available():
        report["error"] = "bwrap is not installed"
        return report
    report["bwrap"] = True
    report["ok"] = True
    return report


def main_report(sandbox_python: Path) -> int:
    report = run_preflight(sandbox_python)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1
