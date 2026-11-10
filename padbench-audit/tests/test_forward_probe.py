"""Forward-probe budget, truncation, and isolation. No 7B weights are loaded."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from padbench_audit.loop import load_config  # noqa: E402
from padbench_audit.preflight import probe_isolation, probe_worker_report  # noqa: E402
from padbench_audit.prelude import prelude_text  # noqa: E402
from padbench_audit.probe_protocol import (  # noqa: E402
    Budget,
    ToyForward,
    call_probe,
    last_nonpad_index,
    valid_key_indexes,
)
from padbench_audit.probe_worker import load_causal_lm, serve, set_adapter_weights, start_isolated_probe  # noqa: E402
from padbench_audit.prompt import build_task  # noqa: E402
from padbench_audit.safetensors_io import write_safetensors  # noqa: E402
from padbench_audit.sandbox import bwrap_available, build_bwrap_command, run_code  # noqa: E402

SANDBOX = Path(os.environ.get("SANDBOX_PYTHON", "/tmp/sbx-venv/bin/python"))
PROBE = Path(os.environ.get("PROBE_PYTHON", "/tmp/probe-venv/bin/python"))
BANNED = ("svd", "spectral", "cleanse", "peftguard", "llama", "toxic", "trigger", "rank8", "huggingface")


def _ensure_probe_venv() -> None:
    if PROBE.is_file():
        return
    PROBE.parent.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(PROBE.parent.parent)])


class ProtocolTests(unittest.TestCase):
    def test_adapter_toggle_budget_and_truncation(self):
        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            root = Path(tmp)
            budget = Budget(2, root / "budget.json")
            stop = threading.Event()
            thread = threading.Thread(
                target=serve,
                kwargs={
                    "sock_path": root / "probe.sock",
                    "backend": ToyForward(),
                    "budget": budget,
                    "max_texts": 2,
                    "max_chars": 50,
                    "max_tokens": 3,
                    "log_path": root / "calls.jsonl",
                    "allow_selfcheck": False,
                    "stop": stop.is_set,
                },
                daemon=True,
            )
            thread.start()
            self._wait(root / "probe.sock")
            try:
                on = call_probe(str(root / "probe.sock"), {"texts": ["one two three four"], "adapter_on": True, "topk": 2})
                off = call_probe(str(root / "probe.sock"), {"texts": ["one two three four"], "adapter_on": False, "attention": True, "layers": [0]})
            finally:
                stop.set()
                thread.join(timeout=3)
        self.assertTrue(on["ok"])
        self.assertTrue(off["ok"])
        self.assertEqual(on["results"][0]["topk"][0]["token"], "yes")
        self.assertEqual(off["results"][0]["topk"][0]["token"], "no")
        self.assertNotEqual(on["results"][0]["topk"][0]["id"], off["results"][0]["topk"][0]["id"])
        self.assertTrue(on["results"][0]["truncated"])
        self.assertEqual(on["results"][0]["n_tokens"], 3)
        self.assertEqual(len(on["results"][0]["topk"]), 2)
        self.assertLessEqual(len(off["results"][0]["attention"][0]["last_to_keys"]), 8)
        self.assertNotIn("hidden", json.dumps(on))
        self.assertEqual(on["remaining"], 1)
        self.assertEqual(off["remaining"], 0)

    def test_over_budget_is_rejected_and_does_not_grow_on_a_bad_request(self):
        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            root = Path(tmp)
            budget = Budget(1, root / "budget.json")
            stop = threading.Event()
            thread = threading.Thread(
                target=serve,
                kwargs={
                    "sock_path": root / "probe.sock",
                    "backend": ToyForward(),
                    "budget": budget,
                    "max_texts": 1,
                    "max_chars": 20,
                    "max_tokens": 4,
                    "log_path": None,
                    "allow_selfcheck": False,
                    "stop": stop.is_set,
                },
                daemon=True,
            )
            thread.start()
            self._wait(root / "probe.sock")
            try:
                bad = call_probe(str(root / "probe.sock"), {"texts": ["a", "b"], "adapter_on": True})
                hidden = call_probe(str(root / "probe.sock"), {"texts": ["a"], "adapter_on": True, "hidden": True})
                good = call_probe(str(root / "probe.sock"), {"texts": ["a"], "adapter_on": False})
                again = call_probe(str(root / "probe.sock"), {"texts": ["a"], "adapter_on": True})
                saved = json.loads((root / "budget.json").read_text())
            finally:
                stop.set()
                thread.join(timeout=3)
        self.assertFalse(bad["ok"])
        self.assertIn("too many texts", bad["error"])
        self.assertFalse(hidden["ok"])
        self.assertIn("hidden", hidden["error"])
        self.assertTrue(good["ok"])
        self.assertFalse(again["ok"])
        self.assertIn("budget", again["error"])
        self.assertEqual(again["remaining"], 0)
        self.assertEqual(saved["used"], 1)

    def test_budget_file_is_what_a_resumed_worker_reads(self):
        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            path = Path(tmp) / "budget.json"
            first = Budget(4, path)
            first.consume()
            first.consume()
            second = Budget(99, path)
        self.assertEqual(second.used, 2)
        self.assertEqual(second.total, 4)
        self.assertEqual(second.remaining(), 2)

    def test_left_padded_batch_matches_a_solo_probe(self):
        # The reported Llama bug: sum(mask)-1 on [0, 0, 0, 1, 1] is 1, a pad column.
        left = [0, 0, 0, 1, 1]
        right = [1, 1, 0, 0, 0]
        self.assertEqual(sum(left) - 1, 1)
        self.assertEqual(last_nonpad_index(left), 4)
        self.assertEqual(last_nonpad_index(right), 1)
        self.assertEqual(valid_key_indexes(left, 8), [3, 4])
        self.assertEqual(valid_key_indexes(right, 2), [0, 1])
        toy = ToyForward()
        kwargs = {"adapter_on": False, "topk": 4, "attention": True, "layers": None, "max_tokens": 16}
        alone = toy.forward(["Hello"], **kwargs)
        batch = toy.forward(["Hello", "one two three four"], **kwargs)
        self.assertEqual(alone[0]["topk"], batch[0]["topk"])
        self.assertEqual(batch[0]["topk"][0]["token"], "no")
        self.assertNotEqual(batch[0]["n_tokens"], batch[1]["n_tokens"])
        self.assertEqual(len(batch[0]["attention"][0]["last_to_keys"]), 1)
        self.assertEqual(len(batch[1]["attention"][0]["last_to_keys"]), 4)

    def test_eager_attention_is_requested_and_a_miss_is_not_silent(self):
        calls = []

        def reject_eager(*_args, **kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("attn_implementation") == "eager":
                raise ValueError("sdpa only")
            return "fallback"

        model, available = load_causal_lm(reject_eager, "base", torch_dtype="fp16", local_files_only=True)
        self.assertEqual(model, "fallback")
        self.assertFalse(available)
        self.assertEqual(calls[0]["attn_implementation"], "eager")
        self.assertNotIn("attn_implementation", calls[1])

        calls.clear()

        def accept_eager(*_args, **kwargs):
            calls.append(dict(kwargs))
            return "eager-model"

        model, available = load_causal_lm(accept_eager, "base", local_files_only=True)
        self.assertEqual(model, "eager-model")
        self.assertTrue(available)
        self.assertEqual(calls[0]["attn_implementation"], "eager")

    def test_adapter_off_copies_the_base_snapshot(self):
        class Rounded:
            def __init__(self, value: float) -> None:
                self.value = value

            def copy_(self, other: "Rounded") -> "Rounded":
                self.value = other.value
                return self

            def __add__(self, delta: float) -> "Rounded":
                return Rounded(round(self.value + delta, 1))

        base = {"w": Rounded(1.04)}
        param = Rounded(1.04)
        set_adapter_weights({"w": param}, base, {"w": 0.04}, True)
        self.assertEqual(param.value, 1.1)
        set_adapter_weights({"w": param}, base, {"w": 0.04}, False)
        self.assertEqual(param.value, 1.04)
        self.assertNotEqual(round(1.1 - 0.04, 2), 1.04)

    def test_unavailable_attention_does_not_spend_budget(self):
        class NoAttention:
            attention_available = False

            def forward(self, *_args, **_kwargs):
                raise AssertionError("forward ran without attention weights")

        with tempfile.TemporaryDirectory(prefix="probe-") as tmp:
            root = Path(tmp)
            budget = Budget(2, root / "budget.json")
            stop = threading.Event()
            thread = threading.Thread(
                target=serve,
                kwargs={
                    "sock_path": root / "probe.sock",
                    "backend": NoAttention(),
                    "budget": budget,
                    "max_texts": 2,
                    "max_chars": 20,
                    "max_tokens": 8,
                    "log_path": None,
                    "allow_selfcheck": False,
                    "stop": stop.is_set,
                },
                daemon=True,
            )
            thread.start()
            self._wait(root / "probe.sock")
            try:
                refused = call_probe(
                    str(root / "probe.sock"),
                    {"texts": ["Hello"], "adapter_on": True, "attention": True},
                )
            finally:
                stop.set()
                thread.join(timeout=3)
        self.assertFalse(refused["ok"])
        self.assertIn("eager", refused["error"])
        self.assertEqual(refused["remaining"], 2)
        self.assertEqual(budget.used, 0)

    def test_lora_target_names(self):
        from padbench_audit.probe_worker import lora_target

        target = lora_target("base_model.model.model.layers.12.self_attn.v_proj.lora_B.default.weight")
        self.assertEqual(target, ("model.layers.12.self_attn.v_proj.weight", "B"))
        self.assertIsNone(lora_target("layer_0.proj_q.A"))

    def test_prompt_describes_the_helper_only(self):
        text = build_task(
            condition="forward-probe",
            max_runs=60,
            timeout_s=600,
            max_output_chars=15000,
            expose_tokenizer=False,
            probe_budget=16,
        )
        lowered = text.lower()
        for word in BANNED:
            self.assertNotIn(word, lowered)
        self.assertIn("def probe(", text)
        self.assertIn("adapter_on", text)
        self.assertIn("16", text)
        self.assertNotIn("def probe(", prelude_text())
        config = load_config(ROOT / "configs" / "example_forward_probe.json")
        self.assertEqual(config["condition"], "forward-probe")
        self.assertEqual(config["probe"]["budget"], 16)
        self.assertEqual([item["folder"] for item in config["cases"]], ["label0_7", "label1_31", "label0_33", "label1_39"])

    def test_same_interpreter_is_refused(self):
        here = Path(sys.executable)
        report = probe_worker_report(
            probe_python=here,
            sandbox_python=here,
            private_dir=Path("/tmp"),
        )
        self.assertFalse(report["ok"])
        self.assertIn("numpy sandbox", report["error"])

    def _wait(self, sock: Path) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if sock.exists():
                return
            time.sleep(0.02)
        self.fail("probe socket did not appear")


@unittest.skipUnless(bwrap_available() and SANDBOX.is_file(), "bwrap and the numpy sandbox are required")
class IsolatedProbeTests(unittest.TestCase):
    def test_worker_is_isolated_and_the_sandbox_can_call_it(self):
        _ensure_probe_venv()
        self.assertNotEqual(PROBE.resolve(), SANDBOX.resolve())
        with tempfile.TemporaryDirectory(prefix="probe-iso-") as tmp:
            root = Path(tmp)
            private = root / "private"
            private.mkdir()
            (private / "DO_NOT_MOUNT").write_text("secret-label1\n")
            case = root / "case"
            work = root / "work"
            prog = root / "prog"
            case.mkdir()
            work.mkdir()
            write_safetensors(case / "adapter.safetensors", [("layer_0.proj_q.A", "F32", [1], (1).to_bytes(4, "little"))])
            (case / "shapes.json").write_text(
                json.dumps({"tensors": [{"dtype": "F32", "name": "layer_0.proj_q.A", "shape": [1]}]}) + "\n"
            )
            handle = start_isolated_probe(
                probe_python=PROBE,
                sock_dir=root / "sock",
                log_dir=root / "log",
                budget=4,
                max_texts=2,
                max_tokens=8,
                max_chars=100,
                base_src=None,
                adapter=None,
                tensor_map=None,
                lora_scale=1.0,
                device="cpu",
                toy=True,
                allow_selfcheck=True,
            )
            try:
                handle.wait_ready(timeout_s=20)
                check = call_probe(str(handle.socket_path), {"op": "selfcheck", "private": str(private)})
                code = (
                    "on = probe(['hello there'], adapter_on=True)\n"
                    "off = probe(['hello there'], adapter_on=False)\n"
                    "print(on['results'][0]['topk'][0]['token'])\n"
                    "print(off['results'][0]['topk'][0]['token'])\n"
                    "print(on['remaining'])\n"
                    "import os\n"
                    "print('BASE', os.path.exists('/base'))\n"
                    "try:\n"
                    "    import torch\n"
                    "    print('TORCH-PRESENT')\n"
                    "except Exception:\n"
                    "    print('TORCH-ABSENT')\n"
                )
                result = run_code(
                    code,
                    sandbox_python=SANDBOX,
                    case_dir=case,
                    work_dir=work,
                    code_dir=prog,
                    timeout_s=30,
                    memory_bytes=8 << 30,
                    extra_ro_binds=[(handle.sock_dir, "/sock")],
                    probe=True,
                )
            finally:
                handle.close()
        self.assertTrue(check["ok"], check)
        self.assertFalse(check["private_visible"])
        self.assertTrue(check["network_blocked"])
        self.assertFalse(check["nvidia"])
        self.assertFalse(check["torch_importable"])
        self.assertEqual(result.exit_code, 0, result.stderr + result.stdout)
        self.assertIn("yes", result.stdout)
        self.assertIn("no", result.stdout)
        self.assertIn("TORCH-ABSENT", result.stdout)
        self.assertIn("BASE False", result.stdout)
        self.assertNotIn("secret-label1", result.stdout)

    def test_numpy_command_does_not_mount_the_base_tree(self):
        with tempfile.TemporaryDirectory(prefix="probe-cmd-") as tmp:
            root = Path(tmp)
            case = root / "case"
            work = root / "work"
            case.mkdir()
            work.mkdir()
            code = root / "code.py"
            code.write_text("print(1)\n")
            write_safetensors(case / "adapter.safetensors", [("layer_0.proj_q.A", "F32", [1], (1).to_bytes(4, "little"))])
            sock = root / "sock"
            sock.mkdir()
            null = os.open("/dev/null", os.O_RDONLY)
            try:
                cmd = build_bwrap_command(
                    sandbox_python=SANDBOX,
                    case_dir=case,
                    work_dir=work,
                    base_dir=None,
                    code_path=code,
                    null_fd=null,
                    extra_ro_binds=[(sock, "/sock")],
                )
            finally:
                os.close(null)
        self.assertIn("/sock", cmd)
        self.assertNotIn("/base", cmd)

    def test_preflight_isolation_helper(self):
        _ensure_probe_venv()
        with tempfile.TemporaryDirectory(prefix="probe-priv-") as tmp:
            private = Path(tmp)
            (private / "DO_NOT_MOUNT").write_text("secret\n")
            report = probe_isolation(probe_python=PROBE, private_dir=private)
        self.assertTrue(report.get("network_blocked"), report)
        self.assertFalse(report.get("private_visible"), report)


if __name__ == "__main__":
    unittest.main()
