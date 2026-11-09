"""Unit tests for the declarative store and the safe checkpoint loader."""
from __future__ import annotations

import io
import os
import pickle
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from strict_static.protocol import build_manifest, manifest_mismatches, report_schema, validate_report
from strict_static.safe_checkpoint import CheckpointRejected, loads_checkpoint
from strict_static.store import Store


def _report(case_id: str, **overrides):
    body = {
        "has_backdoor": False,
        "target_class": None,
        "trigger_description": None,
        "suspicious_units": ["w0"],
        "forwards_used": 0,
        "confidence": 0.4,
        "edit_summary": None,
        "notes": None,
        "sandbox": "strict-static",
        "protocol": "strict-static",
        "protocol_version": "strict-static-1",
        "case_id": case_id,
    }
    body.update(overrides)
    return body


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.w = {
            "a.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        }
        self.store = Store.from_weights(self.w)

    def test_handles_are_anonymous_and_originals_are_frozen(self):
        catalog = self.store.catalog()
        blob = str(catalog)
        self.assertNotIn("a.weight", blob)
        self.assertNotIn("b.weight", blob)
        self.assertEqual(self.store.keymap["w0"], "a.weight")
        self.assertFalse(self.store.tensors["w0"].array.flags.writeable)

    def test_cross_layer_matmul_matches_numpy(self):
        self.store2 = Store.from_weights(
            {"left": np.arange(6, dtype=np.float64).reshape(2, 3), "right": np.arange(12, dtype=np.float64).reshape(3, 4)}
        )
        result = self.store2.execute({"op": "matmul", "a": "w0", "b": "w1"})
        self.assertTrue(result["ok"], result)
        prod = self.store2.tensors[result["handle"]].array
        np.testing.assert_allclose(prod, np.arange(6).reshape(2, 3) @ np.arange(12).reshape(3, 4))

    def test_rejects_synthetic_inputs_and_lineage_fields(self):
        for op in ("randn", "rand", "eye", "zeros", "ones", "arange", "forward", "relu", "python"):
            got = self.store.execute({"op": op})
            self.assertFalse(got["ok"], op)
            self.assertIn(got["code"], ("forbidden_op", "unknown_op"))
        spoof = self.store.execute({"op": "scale", "tensor": "w0", "scalar": 2, "data": [1, 2, 3], "lineage": ["w0"]})
        self.assertFalse(spoof["ok"])
        self.assertEqual(spoof["code"], "forbidden_field")
        self.assertFalse(self.store.tensors["w0"].array.flags.writeable)

    def test_stats_svd_and_patch_do_not_mutate_original(self):
        before = self.store.tensors["w0"].array.copy()
        stats = self.store.execute({"op": "stats", "tensor": "w0", "q": [0.5, 0.9]})
        self.assertTrue(stats["ok"], stats)
        self.assertIn("l2", stats["stats"])
        svd = self.store.execute({"op": "svd", "tensor": "w0", "k": 2})
        self.assertTrue(svd["ok"], svd)
        self.assertEqual(len(svd["singular_values"]), 2)
        cloned = self.store.execute({"op": "slice", "tensor": "w0", "spec": [[0, 1], [0, 1]]})
        patched = self.store.execute(
            {"op": "assign_slice", "dest": "w0", "src": cloned["handle"], "spec": [[0, 1], [0, 1]]}
        )
        self.assertTrue(patched["ok"], patched)
        np.testing.assert_array_equal(self.store.tensors["w0"].array, before)
        self.assertFalse(self.store.tensors["w0"].array.flags.writeable)

    def test_depth_and_op_budgets_fail_closed(self):
        tiny = Store.from_weights(self.w, {"max_depth": 1, "max_ops": 2})
        first = tiny.execute({"op": "clone", "tensor": "w0"})
        self.assertTrue(first["ok"], first)
        second = tiny.execute({"op": "clone", "tensor": first["handle"]})
        self.assertFalse(second["ok"])
        self.assertEqual(second["code"], "budget_depth")
        third = tiny.execute({"op": "stats", "tensor": "w0"})
        self.assertFalse(third["ok"])
        self.assertEqual(third["code"], "budget_ops")

    def test_rejected_ops_are_audited(self):
        self.store.execute({"op": "eye"})
        self.assertTrue(any(row.get("ok") is False for row in self.store.audit))


def _torch_module_type():
    import sys
    import torch

    module = sys.modules[__name__]
    existing = getattr(module, "Net", None)
    if existing is not None:
        return existing

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 2)

    Net.__module__ = __name__
    Net.__qualname__ = "Net"
    setattr(module, "Net", Net)
    return Net


class CheckpointTests(unittest.TestCase):
    def test_numeric_npz_and_metadata_drop(self):
        buf = io.BytesIO()
        np.savez(
            buf,
            **{"fc.weight": np.ones((2, 2), dtype=np.float32), "epoch": np.int64(4)},
        )
        out = loads_checkpoint(buf.getvalue())
        self.assertEqual(list(out), ["fc.weight"])
        self.assertEqual(out["fc.weight"].shape, (2, 2))

    def test_raw_pickle_and_module_globals_are_rejected(self):
        canary = Path("/tmp/strict-static-should-not-exist")
        if canary.exists():
            canary.unlink()

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {canary}",))

        raw = pickle.dumps(Evil())
        with self.assertRaises(CheckpointRejected):
            loads_checkpoint(raw)
        blob = io.BytesIO()
        with zipfile.ZipFile(blob, "w") as zf:
            zf.writestr("data.pkl", raw)
        with self.assertRaises(CheckpointRejected):
            loads_checkpoint(blob.getvalue())
        self.assertFalse(canary.exists())

    def test_torch_state_dict_roundtrip_and_module_rejection(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed; numeric npz coverage still ran")
        sd = {
            "fc.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "fc.bias": torch.tensor([0.25, -0.5]),
            "epoch": torch.tensor(3),
        }
        buf = io.BytesIO()
        torch.save(sd, buf)
        out = loads_checkpoint(buf.getvalue())
        self.assertNotIn("epoch", out)
        np.testing.assert_allclose(out["fc.weight"], sd["fc.weight"].numpy())

        Net = _torch_module_type()
        module_buf = io.BytesIO()
        torch.save(Net(), module_buf)
        with self.assertRaises(CheckpointRejected):
            loads_checkpoint(module_buf.getvalue())

    def test_symlink_checkpoint_is_refused(self):
        from strict_static.safe_checkpoint import load_checkpoint

        directory = Path("/tmp/strict-static-link-test")
        if directory.exists():
            if directory.is_dir() and not directory.is_symlink():
                for child in directory.iterdir():
                    child.unlink()
            else:
                directory.unlink()
        directory.mkdir(exist_ok=True)
        real = directory / "real.npy"
        with real.open("wb") as fh:
            np.save(fh, np.ones(3))
        link = directory / "link.npy"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(real)
        with self.assertRaises(CheckpointRejected):
            load_checkpoint(str(link))


class ProtocolTests(unittest.TestCase):
    def test_manifest_matches_checked_in_bytes(self):
        self.assertEqual(manifest_mismatches(), [])

    def test_report_schema_file_matches(self):
        schema_path = Path(__file__).resolve().parents[4] / "astra-backdoor-audit" / "schema" / "strict_report.schema.json"
        import json

        on_disk = json.loads(schema_path.read_text())
        self.assertEqual(on_disk, report_schema())

    def test_report_validation(self):
        case_id = "c" + "ab" * 8
        self.assertEqual(validate_report(_report(case_id), case_id=case_id), [])
        errs = validate_report(_report(case_id, forwards_used=1, sandbox="weight-only"), case_id=case_id)
        self.assertTrue(errs)
        self.assertTrue(any("forwards_used" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
