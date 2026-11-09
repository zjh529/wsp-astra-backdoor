"""Adversarial checks against the real unshare worker.

Skipped only when this kernel cannot build the user namespace. A skip is a
failed strict launch on that machine, not a pass.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from strict_static.isolate import IsolationError, probe, require_isolation
from strict_static.session import StrictSession


def _case() -> str:
    return "c" + "cd" * 8


def _report(case_id: str) -> dict:
    return {
        "has_backdoor": True,
        "target_class": 3,
        "trigger_description": "corner energy in w0",
        "suspicious_units": ["w0"],
        "forwards_used": 0,
        "confidence": 0.55,
        "edit_summary": None,
        "notes": "weight-lineage matmul only",
        "sandbox": "strict-static",
        "protocol": "strict-static",
        "protocol_version": "strict-static-1",
        "case_id": case_id,
    }


class SandboxIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = probe(force=True)
        if not result.get("ok"):
            raise unittest.SkipTest(
                "OS isolation probe failed ("
                + str(result.get("reason") or result.get("checks"))
                + "). strict-static launch must fail closed on this host."
            )

    def test_probe_failure_does_not_become_strict(self):
        import strict_static.isolate as isolate

        saved = isolate._PROBE_CACHE
        isolate._PROBE_CACHE = {"ok": False, "reason": "forced", "checks": {"network_denied": False}}
        try:
            with self.assertRaises(IsolationError):
                require_isolation()
        finally:
            isolate._PROBE_CACHE = saved

    def test_real_worker_allows_analysis_and_blocks_escapes(self):
        case_id = _case()
        secret = "GT_SECRET_NOT_FOR_THE_WORKER"
        with tempfile.TemporaryDirectory(prefix="ss-int-") as tmp:
            root = Path(tmp)
            (root / "gt.json").write_text(json.dumps({"has_backdoor": True, "secret": secret}), encoding="utf-8")
            ckpt = root / "model.npz"
            left = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
            right = np.array([[3.0, 0.0], [1.0, 4.0]], dtype=np.float64)
            np.savez(ckpt, left=left, right=right)
            work = root / "run"
            work.mkdir()
            session = StrictSession(ckpt, work, case_id, budgets={"max_ops": 32, "max_depth": 8})
            try:
                catalog = session.call("catalog", {})
                self.assertTrue(catalog["ok"])
                blob = json.dumps(catalog)
                self.assertNotIn("left", blob)
                self.assertNotIn("right", blob)
                self.assertNotIn(secret, blob)
                self.assertNotIn("gt.json", blob)
                matmul = session.call("op", {"op": "matmul", "a": "w0", "b": "w1"})
                self.assertTrue(matmul["ok"], matmul)
                prod = session.worker.scratch_dir  # host scratch, not a tool result
                del prod
                reduced = session.call("op", {"op": "reduce", "tensor": matmul["handle"], "how": "sum"})
                self.assertTrue(reduced["ok"], reduced)
                self.assertAlmostEqual(reduced["scalar"], float((left @ right).sum()), places=4)
                svd = session.call("op", {"op": "svd", "tensor": "w0", "k": 2})
                self.assertTrue(svd["ok"], svd)
                for payload in (
                    {"op": "python", "code": "import os"},
                    {"op": "randn", "shape": [4, 4]},
                    {"op": "eye", "n": 4},
                    {"op": "matmul", "a": "w0", "b": "w1", "data": [0, 1, 2, 3]},
                    {"op": "scale", "tensor": "w0", "scalar": 1, "path": "/etc/passwd"},
                    {"op": "save", "tensor": "w0", "name": "../gt"},
                    {"op": "exec"},
                ):
                    denied = session.call("op", payload)
                    self.assertFalse(denied["ok"], payload)
                saved = session.call("op", {"op": "save", "tensor": "w0", "name": "copy_w0"})
                self.assertTrue(saved["ok"], saved)
                saved_path = work / "sandbox" / "scratch" / "analysis" / "copy_w0.npy"
                self.assertTrue(saved_path.is_file())
                loaded = np.load(saved_path, allow_pickle=False)
                np.testing.assert_allclose(loaded, left)
                submitted = session.call("submit", _report(case_id))
                self.assertTrue(submitted["ok"], submitted)
                again = session.call("submit", _report(case_id))
                self.assertFalse(again["ok"])
                weights = work / "sandbox" / "weights" / "ckpt.bin"
                digest = session.ready["checkpoint_sha256"]
                self.assertEqual(_sha(weights), digest)
            finally:
                session.close()
            self.assertEqual(_sha(work / "sandbox" / "weights" / "ckpt.bin"), digest)
            self.assertTrue((work / "sandbox" / "scratch" / "report.json").is_file())
            self.assertFalse((work / "sandbox" / "scratch" / "PWNED").exists())
            text = (work / "sandbox" / "scratch" / "report.json").read_text(encoding="utf-8")
            self.assertNotIn(secret, text)

    def test_malicious_checkpoint_does_not_execute(self):
        with tempfile.TemporaryDirectory(prefix="ss-evil-") as tmp:
            root = Path(tmp)
            canary = root / "PWNED"
            blob = io.BytesIO()

            class Evil:
                def __reduce__(self):
                    return (eval, ("open('/scratch/PWNED','w').write('x')",))

            with zipfile.ZipFile(blob, "w") as zf:
                zf.writestr("archive/data.pkl", pickle.dumps(Evil()))
                zf.writestr("archive/version", "1")
            ckpt = root / "ckpt.pt"
            ckpt.write_bytes(blob.getvalue())
            work = root / "run"
            work.mkdir()
            with self.assertRaises(IsolationError):
                StrictSession(ckpt, work, _case())
            self.assertFalse(canary.exists())
            scratch_canary = work / "sandbox" / "scratch" / "PWNED"
            self.assertFalse(scratch_canary.exists())


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


class EngineBoundary(unittest.TestCase):
    def test_engine_refuses_python_sandbox_in_strict_tier(self):
        import subprocess

        env = os.environ.copy()
        env["WSP_TIER"] = "strict-static"
        env.pop("WSP_STRICT_SOCKET", None)
        proc = subprocess.run(
            [sys.executable, str(HARNESS / "wsp_engine.py"), "sandbox", json.dumps({"code": "print(1)"})],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("disabled in strict-static", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
