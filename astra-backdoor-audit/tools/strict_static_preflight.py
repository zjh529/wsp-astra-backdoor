#!/usr/bin/env python3
"""Server preflight for strict-static.

Runs the enforcement tests. This does not call a model and does not read
credentials. Exit 0 only when the tests pass. With --require-pi, also require
a local `pi` binary whose help lists the flags the launch spec uses.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _pi_check(require: bool) -> dict:
    pi = shutil.which("pi")
    if not pi:
        return {
            "present": False,
            "ok": not require,
            "skipped": not require,
            "reason": "pi is not on PATH",
        }
    version = subprocess.run([pi, "--version"], capture_output=True, text=True, check=False)
    help_run = subprocess.run([pi, "--help"], capture_output=True, text=True, check=False)
    blob = (help_run.stdout or "") + (help_run.stderr or "")
    flags = {
        "--no-builtin-tools": "--no-builtin-tools" in blob,
        "--offline": "--offline" in blob,
        "--exclude-tools": "--exclude-tools" in blob,
        "--no-extensions": "--no-extensions" in blob,
        "--tools": "--tools" in blob or "-t," in blob or "--tools" in blob,
    }
    ext = REPO / "pi-weight-space-programming" / "extensions" / "index.ts"
    ext_help = subprocess.run(
        [pi, "--no-extensions", "-e", str(ext), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    ext_blob = (ext_help.stdout or "") + (ext_help.stderr or "")
    return {
        "present": True,
        "path": pi,
        "version": (version.stdout or version.stderr).strip(),
        "version_ok": version.returncode == 0,
        "flags": flags,
        "extension_help_ok": ext_help.returncode == 0 and "wsp-mode" in ext_blob,
        "ok": version.returncode == 0 and all(flags.values()) and ext_help.returncode == 0 and "wsp-mode" in ext_blob,
    }


def _run_tests() -> dict:
    suites = []
    loader = unittest.defaultTestLoader
    for start in (
        REPO / "pi-weight-space-programming" / "harness" / "strict_static" / "tests",
        ROOT / "tests",
    ):
        suites.append(loader.discover(str(start), pattern="test_*.py", top_level_dir=str(start)))
    suite = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "ok": result.wasSuccessful(),
        "ran": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skip_reasons": [f"{test}: {reason}" for test, reason in result.skipped],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict-static enforcement preflight")
    parser.add_argument("--require-pi", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    tests = _run_tests()
    pi = _pi_check(args.require_pi)
    isolation_skips = [
        row for row in tests["skip_reasons"] if "isolation" in row.lower() or "fail closed" in row.lower()
    ]
    report = {
        "protocol": "strict-static-1",
        "tests": tests,
        "pi": pi,
        "isolation_skips": isolation_skips,
        "ok": bool(tests["ok"] and pi["ok"] and not isolation_skips),
    }
    report["skip_policy"] = (
        "A skipped OS-isolation test fails this gate. A missing pi CLI fails only with --require-pi. "
        "A missing torch install skips the state_dict roundtrip and does not by itself fail the gate."
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
