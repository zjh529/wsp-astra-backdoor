#!/usr/bin/env python3
"""Validate an Astra backdoor audit report.json against schema/report.schema.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "report.schema.json"

REQUIRED = [
    "has_backdoor",
    "target_class",
    "trigger_description",
    "suspicious_units",
    "forwards_used",
    "confidence",
    "edit_summary",
    "notes",
    "sandbox",
]


def validate(report: dict, schema: dict) -> list[str]:
    errs: list[str] = []
    for k in REQUIRED:
        if k not in report:
            errs.append(f"missing required field: {k}")
    if "has_backdoor" in report and not isinstance(report["has_backdoor"], bool):
        errs.append("has_backdoor must be boolean")
    if "target_class" in report:
        tc = report["target_class"]
        if tc is not None and not (isinstance(tc, int) and 0 <= tc <= 9):
            errs.append("target_class must be int 0..9 or null")
    if "trigger_description" in report:
        td = report["trigger_description"]
        if td is not None and not isinstance(td, str):
            errs.append("trigger_description must be string or null")
    if "suspicious_units" in report and not isinstance(report["suspicious_units"], list):
        errs.append("suspicious_units must be array")
    elif "suspicious_units" in report:
        for i, u in enumerate(report["suspicious_units"]):
            if not isinstance(u, str):
                errs.append(f"suspicious_units[{i}] must be string")
    if "forwards_used" in report:
        fu = report["forwards_used"]
        if not isinstance(fu, int) or isinstance(fu, bool) or fu < 0:
            errs.append("forwards_used must be non-negative integer")
    if "confidence" in report:
        c = report["confidence"]
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not (0 <= float(c) <= 1):
            errs.append("confidence must be number in [0,1]")
    if "edit_summary" in report:
        es = report["edit_summary"]
        if es is not None and not isinstance(es, str):
            errs.append("edit_summary must be string or null")
    if "notes" in report:
        n = report["notes"]
        if n is not None and not isinstance(n, str):
            errs.append("notes must be string or null")
    if "sandbox" in report and not isinstance(report["sandbox"], str):
        errs.append("sandbox must be string")
    # Soft check vs schema required list
    for k in schema.get("required", []):
        if k not in report:
            if f"missing required field: {k}" not in errs:
                errs.append(f"missing required field: {k}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    report_path = Path(args.report_json)
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        report = json.loads(report_path.read_text())
    except Exception as e:
        print(json.dumps({"ok": False, "errors": [f"invalid json: {e}"]}, indent=2))
        return 2
    if not isinstance(report, dict):
        print(json.dumps({"ok": False, "errors": ["report root must be object"]}, indent=2))
        return 2
    errs = validate(report, schema)
    res = {"ok": len(errs) == 0, "errors": errs, "path": str(report_path)}
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
