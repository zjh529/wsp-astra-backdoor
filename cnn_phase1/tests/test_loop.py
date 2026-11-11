"""Usage limit pauses the queue. Infra retries use the existing schedule."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cnn_phase1.cases import UNAVAILABLE_TEXT
from cnn_phase1.model import TENSOR_SPEC, shapes_document
from cnn_phase1.protocol import load_config
from cnn_phase1.loop import run_case
from cnn_phase1.queue import EXIT_USAGE_LIMIT, run_queue

import sys

_PADBENCH = Path(__file__).resolve().parents[2] / "padbench-audit"
sys.path.insert(0, str(_PADBENCH))
from padbench_audit.pi_client import PiDriver, PiTurn  # noqa: E402


def _write_case(root: Path, case_id: str) -> None:
    case = root / "cases" / case_id
    case.mkdir(parents=True)
    (case / "shapes.json").write_text(json.dumps(shapes_document()), encoding="utf-8")
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in TENSOR_SPEC}
    np.savez(case / "weights.npz", **arrays)


def _usage_turn() -> PiTurn:
    message = "Codex error: The usage limit has been reached"
    raw = json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": message,
                "usage": {"input": 1, "output": 0, "totalTokens": 1},
            },
        }
    )
    return PiTurn(
        text="",
        raw_stdout=raw + "\n",
        raw_stderr="",
        usage={"input": 1, "output": 0, "totalTokens": 1},
        model=None,
        wall_s=0.01,
        exit_code=0,
        stop_reason="error",
        error_message=message,
        session_id="scripted",
        argv=["scripted", "--no-tools"],
    )


def _infra_turn() -> PiTurn:
    raw = json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "fetch failed",
                "usage": {"input": 0, "output": 0, "totalTokens": 0},
            },
        }
    )
    return PiTurn(
        text="",
        raw_stdout=raw + "\n",
        raw_stderr="",
        usage={"input": 0, "output": 0, "totalTokens": 0},
        model=None,
        wall_s=0.01,
        exit_code=0,
        stop_reason="error",
        error_message="fetch failed",
        session_id="scripted",
        argv=["scripted", "--no-tools"],
    )


class QueueTests(unittest.TestCase):
    def test_usage_limit_pauses_without_counting_a_failure_or_sleeping(self):
        with tempfile.TemporaryDirectory(prefix="p1b-") as tmp:
            blind = Path(tmp)
            _write_case(blind, "case_001")
            _write_case(blind, "case_002")
            (blind / "case_order.txt").write_text("case_001\ncase_002\n", encoding="utf-8")
            calls = []
            sleeps = []

            def factory(case_id):
                calls.append(case_id)
                return [_usage_turn()]

            code, rows = run_queue(
                config=load_config(),
                blind=blind,
                sandbox_python=Path("/usr/bin/python3"),
                pi_bin=None,
                pi_cwd=blind / "pi",
                resume=False,
                scripted_factory=factory,
                sleeper=sleeps.append,
            )
            self.assertEqual(code, EXIT_USAGE_LIMIT)
            self.assertEqual(calls, ["case_001"])
            self.assertEqual(sleeps, [])
            self.assertFalse(rows[0]["counted_as_failure"])
            self.assertEqual(rows[0]["status"], "usage_limit")
            pause = json.loads((blind / "queue_pause.json").read_text(encoding="utf-8"))
            self.assertEqual(pause["remaining"], ["case_001", "case_002"])
            self.assertFalse(pause["counted_as_failure"])
            self.assertFalse((blind / "runs" / "case_001" / "prediction.sha256").exists())

    def test_infra_retries_use_the_existing_backoff_then_stay_resumable(self):
        with tempfile.TemporaryDirectory(prefix="p1b-") as tmp:
            blind = Path(tmp)
            _write_case(blind, "case_001")
            (blind / "case_order.txt").write_text("case_001\n", encoding="utf-8")
            sleeps = []
            code, rows = run_queue(
                config=load_config(),
                blind=blind,
                sandbox_python=Path("/usr/bin/python3"),
                pi_bin=None,
                pi_cwd=blind / "pi",
                resume=False,
                scripted_factory=lambda _case: [_infra_turn(), _infra_turn(), _infra_turn(), _infra_turn()],
                sleeper=sleeps.append,
            )
            self.assertEqual(code, 1)
            self.assertEqual(sleeps, [60, 300, 900])
            self.assertEqual(rows[0]["status"], "pi_error")
            self.assertTrue(rows[0]["counted_as_failure"])
            self.assertTrue(rows[0]["resumable"])

    def test_placeholder_text_is_stable(self):
        self.assertIn("Weights are unavailable", UNAVAILABLE_TEXT)

    def test_completion_clears_a_usage_limit_error(self):
        with tempfile.TemporaryDirectory(prefix="p1b-") as tmp:
            blind = Path(tmp)
            _write_case(blind, "case_001")
            paused = run_case(
                config=load_config(),
                blind=blind,
                case_id="case_001",
                sandbox_python=Path("/usr/bin/python3"),
                scripted_replies=[_usage_turn()],
                pi_cwd=blind / "pi",
            )
            self.assertEqual(paused["status"], "usage_limit")
            self.assertIn("usage limit", paused["error"].lower())
            work = blind / "runs" / "case_001" / "work"
            np.savez(
                work / "trigger.npz",
                mask=np.zeros((28, 28), dtype=np.float32),
                pattern=np.zeros((28, 28), dtype=np.float32),
            )
            body = {
                "verdict": "clean",
                "probability": 0.4,
                "target_class": None,
                "trigger_description": "",
                "suspicious_weights": [],
                "edit": "no_edit",
                "edit_summary": "",
            }
            reply = "```json\n" + json.dumps(body) + "\n```\n"
            # Two nudges, then the third valid answer is accepted.
            finished = run_case(
                config=load_config(),
                blind=blind,
                case_id="case_001",
                sandbox_python=Path("/usr/bin/python3"),
                scripted_replies=[reply, reply, reply],
                pi_cwd=blind / "pi",
                resume=True,
            )
            self.assertEqual(finished["status"], "completed")
            self.assertIsNone(finished.get("error"))
            saved = json.loads((blind / "runs" / "case_001" / "state.json").read_text(encoding="utf-8"))
            self.assertIsNone(saved.get("error"))


class PiLaunchTests(unittest.TestCase):
    def test_run_turn_closes_stdin(self):
        with tempfile.TemporaryDirectory(prefix="p1pi-") as tmp:
            cwd = Path(tmp) / "cwd"
            cwd.mkdir()
            system = Path(tmp) / "system.txt"
            system.write_text("Answer.\n", encoding="utf-8")
            driver = PiDriver(
                pi_bin="pi-test",
                provider="openai-codex",
                model_id="gpt-6-astra",
                effort="medium",
                system_prompt=system,
                session_dir=cwd / "sessions",
                session_id="cnnp1-case-001",
                cwd=cwd,
                timeout_s=5,
            )
            completed = mock.Mock(returncode=0, stdout="", stderr="")

            def fake_run(argv, **kwargs):
                self.assertIs(kwargs["stdin"], __import__("subprocess").DEVNULL)
                self.assertIn("--no-tools", argv)
                self.assertIn("--offline", argv)
                message = json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "ok"}],
                            "stopReason": "stop",
                            "usage": {"input": 1, "output": 1, "totalTokens": 2},
                            "provider": "openai-codex",
                            "model": "gpt-6-astra",
                        },
                    }
                )
                completed.stdout = message + "\n"
                return completed

            with mock.patch("padbench_audit.pi_client.subprocess.run", side_effect=fake_run):
                turn = driver.run_turn(user_message="Hello")
            self.assertEqual(turn.text, "ok")
            settings = (cwd / ".pi" / "settings.json").read_text(encoding="utf-8")
            self.assertIn('"enabled": false', settings)
