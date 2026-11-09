"""Static checks that strict-static does not publish the Python tool surface.

A live `pi` process is optional. The preflight runs that check when the CLI
exists and fails the server gate when `--require-pi` is set and it does not.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "pi-weight-space-programming" / "extensions" / "index.ts"
PRESETS = REPO / "pi-weight-space-programming" / "harness" / "presets.json"
STRICT = ["wsp_static_catalog", "wsp_static_op", "wsp_static_submit_report"]


class SurfaceTests(unittest.TestCase):
    def test_presets_and_extension_allowlist(self):
        presets = json.loads(PRESETS.read_text())
        self.assertEqual(presets["modes"]["strict-static"]["tools"], STRICT)
        legacy = presets["modes"]["weight-only"]["tools"]
        self.assertIn("wsp_python_sandbox", legacy)
        self.assertNotIn("wsp_python_sandbox", STRICT)
        text = INDEX.read_text()
        self.assertIn("STRICT_STATIC_TOOL_NAMES", text)
        for name in STRICT:
            self.assertIn(f'"{name}"', text)
        self.assertIn("WSP_TIER === \"strict-static\"", text)
        self.assertIn("skillPaths: []", text)
        self.assertIn("is not available in strict-static mode", text)
        self.assertIn("--no-builtin-tools", (REPO / "astra-backdoor-audit" / "tools" / "run_cell.py").read_text())

    def test_pi_cli_if_present(self):
        pi = shutil.which("pi")
        if not pi:
            # Presence is enforced by strict_static_preflight.py --require-pi.
            self.assertIsNone(pi)
            return
        version = subprocess.run([pi, "--version"], capture_output=True, text=True, check=False)
        self.assertEqual(version.returncode, 0, version.stderr)
        help_text = subprocess.run([pi, "--help"], capture_output=True, text=True, check=False)
        blob = help_text.stdout + help_text.stderr
        self.assertIn("--no-builtin-tools", blob)
        self.assertIn("--offline", blob)
        self.assertIn("--exclude-tools", blob)


if __name__ == "__main__":
    unittest.main()
