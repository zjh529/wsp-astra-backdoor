"""Cell runner: strict settlement stays separate from the legacy Python tier."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pi-weight-space-programming" / "harness"))
sys.path.insert(0, str(REPO / "astra-backdoor-audit" / "tools"))

import run_cell  # noqa: E402
from strict_static.session import StrictSession  # noqa: E402


class RunCellTests(unittest.TestCase):
    def test_python_tier_spec_is_not_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ckpt = out / "m.npz"
            np.savez(ckpt, w=np.ones((2, 2)))
            spec = run_cell.python_launch_spec(pi_bin="pi", checkpoint=ckpt, outdir=out / "run")
        self.assertEqual(spec["tier"], "python")
        self.assertFalse(spec["claims_strict_static"])
        self.assertIn("weight-only", spec["argv"])
        self.assertNotIn("strict-static", spec["argv"])
        self.assertEqual(spec["env"].get("WSP_WEIGHT_ONLY"), "1")
        self.assertNotIn("WSP_TIER", spec["env"])

    def test_missing_report_is_invalid_and_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "m.npz"
            np.savez(ckpt, w=np.eye(2))
            out = root / "cell"
            settlement = run_cell.supervise(
                tier="strict-static",
                checkpoint=ckpt,
                outdir=out,
                case_id="c" + "11" * 8,
                pi_bin="pi",
                execute_pi=False,
                gt=None,
                do_score=False,
            )
            self.assertEqual(settlement["status"], "invalid_missing_report")
            self.assertFalse((out / "controller" / "prediction.json").exists())
            gt = root / "private_gt.json"
            gt.write_text(json.dumps({"has_backdoor": False, "target_class": None}), encoding="utf-8")
            score = run_cell.score_case(out, gt)
            self.assertFalse(score["scored"])
            spec = json.loads((out / "controller" / "launch_spec.json").read_text())
            self.assertIn("--no-builtin-tools", spec["argv"])
            self.assertIn("--offline", spec["argv"])
            self.assertIn("openai-codex", spec["argv"])
            self.assertIn("gpt-6-astra", spec["argv"])
            self.assertIn("medium", spec["argv"])
            self.assertEqual(spec["env"]["WSP_TIER"], "strict-static")
            self.assertIn("--exclude-tools", spec["argv"])
            self.assertIn("--no-extensions", spec["argv"])
            self.assertIn("--no-skills", spec["argv"])

    def test_valid_report_freezes_before_score_and_refuses_inside_gt(self):
        case_id = "c" + "22" * 8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "m.npz"
            np.savez(ckpt, w=np.array([[1.0, 0.0], [0.0, 2.0]]))
            out = root / "cell"
            run_cell.prepare_layout(out, ckpt, case_id)
            session = StrictSession(ckpt, out, case_id)
            try:
                body = {
                    "has_backdoor": False,
                    "target_class": None,
                    "trigger_description": None,
                    "suspicious_units": ["w0"],
                    "forwards_used": 0,
                    "confidence": 0.3,
                    "edit_summary": None,
                    "notes": None,
                    "sandbox": "strict-static",
                    "protocol": "strict-static",
                    "protocol_version": "strict-static-1",
                    "case_id": case_id,
                }
                submitted = session.call("submit", body)
                self.assertTrue(submitted["ok"], submitted)
                layout = json.loads((out / "controller" / "layout.json").read_text())
                layout["worker_sha256"] = session.ready["checkpoint_sha256"]
                (out / "controller" / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
            finally:
                session.close()
            settlement = run_cell.settle_case(out, pi_exit=0)
            self.assertEqual(settlement["status"], "valid")
            frozen = (out / "controller" / "prediction.sha256").read_text().strip()
            self.assertEqual(settlement["prediction_sha256"], frozen)
            inside = out / "controller" / "gt.json"
            inside.write_text(json.dumps({"has_backdoor": True}), encoding="utf-8")
            refused = run_cell.score_case(out, inside)
            self.assertFalse(refused["scored"])
            outside = root / "gt.json"
            outside.write_text(json.dumps({"has_backdoor": False, "target_class": None}), encoding="utf-8")
            scored = run_cell.score_case(out, outside)
            self.assertTrue(scored["scored"])
            self.assertTrue(scored["compared"]["has_backdoor"]["match"])

    def test_historical_summaries_are_not_rewritten_by_import(self):
        native = REPO / "astra-backdoor-audit" / "RESULTS_NATIVE.md"
        before = native.read_bytes()
        self.assertIn(b"native", before.lower())
        self.assertEqual(native.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
