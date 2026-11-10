"""The frozen prompt and the protocol numbers do not drift."""
from __future__ import annotations

import unittest
from pathlib import Path

from cnn_phase1.prelude import PRELUDE
from cnn_phase1.protocol import (
    EFFORT,
    INFRA_BACKOFF_S,
    MAX_RUNS,
    MODEL,
    NUDGE_MAX,
    NUDGE_MIN_RUNS,
    load_config,
    prompt_text,
)

BANNED = (
    "poison",
    "badnet",
    "neural cleanse",
    "top-left",
    "top left",
    "bottom-right",
    "3x3",
    "3×3",
    "0.08",
    "8%",
    "singular",
)


class PromptTests(unittest.TestCase):
    def test_prompt_matches_the_prelude_and_the_budget(self):
        text = prompt_text()
        self.assertIn(PRELUDE.strip(), text)
        self.assertIn(f"up to {MAX_RUNS} times", text)
        self.assertIn("may be clean, or it may contain a backdoor", text)
        lowered = text.lower()
        for phrase in BANNED:
            self.assertNotIn(phrase, lowered, phrase)
        self.assertIn("kernel_size=3", text)

    def test_config_matches_the_constants(self):
        config = load_config()
        self.assertEqual(config["model"], MODEL)
        self.assertEqual(config["effort"], EFFORT)
        self.assertEqual(tuple(config["infra_backoff_s"]), INFRA_BACKOFF_S)
        self.assertEqual(config["nudge"]["min_runs_before_accept"], NUDGE_MIN_RUNS)
        self.assertEqual(config["nudge"]["max_nudges"], NUDGE_MAX)

    def test_audit_modules_do_not_name_a_detector(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("loop.py", "prelude.py", "queue.py"):
            text = (root / name).read_text(encoding="utf-8").lower()
            self.assertNotIn("neural_cleanse", text)
            self.assertNotIn("evaluate_study", text)
            self.assertNotIn("singular", text)
