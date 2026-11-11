"""Phase-2 arms, case exclusion, anchor Holm, and the frozen v1.1 prompt."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cnn_phase1.aggregate_phase2 import aggregate_phase2
from cnn_phase1.cases import make_phase2_cases
from cnn_phase1.hashing import sha256_file
from cnn_phase1.phase2 import (
    arm_ids,
    case_order_seed,
    target_for_seed,
    trigger_for,
)
from cnn_phase1.protocol import config_sha256, prompt_sha256
from cnn_phase1.spec import phase1_trigger


V11_PROMPT = "28316fe3a4efc36e58976c80d3ce8382e77bf7455c26e1c353ba0b84147f7789"
V11_CONFIG = "69ef508c263be3f1b7bc8855b0cd9c699a2459bcc4723f9450b7f31fb69253c0"


def _changed(arm: str, seed: int = 0) -> set[str]:
    anchor = phase1_trigger().to_dict()
    spec = trigger_for(arm, seed).to_dict()
    return {key for key in anchor if anchor[key] != spec[key]}


class PromptLockTests(unittest.TestCase):
    def test_prompt_and_harness_match_v11(self):
        self.assertEqual(prompt_sha256(), V11_PROMPT)
        self.assertTrue(prompt_sha256().startswith("28316fe3"))
        self.assertEqual(config_sha256(), V11_CONFIG)


class ArmTests(unittest.TestCase):
    def test_each_arm_changes_the_intended_factor(self):
        self.assertEqual(_changed("P-center"), {"position"})
        self.assertEqual(_changed("P-bottomright"), {"position"})
        self.assertEqual(trigger_for("P-center", 0).position, "center")
        self.assertEqual(trigger_for("P-bottomright", 0).position, "bottom_right")
        self.assertEqual(_changed("S-rect"), {"shape", "size", "height", "width"})
        self.assertEqual((trigger_for("S-rect", 0).height, trigger_for("S-rect", 0).width), (2, 6))
        self.assertEqual(_changed("R-1pct"), {"poison_rate"})
        self.assertEqual(_changed("R-0.3pct"), {"poison_rate"})
        self.assertAlmostEqual(trigger_for("R-1pct", 0).poison_rate, 0.01)
        self.assertAlmostEqual(trigger_for("R-0.3pct", 0).poison_rate, 0.003)
        self.assertEqual(_changed("T-vary", 0), {"target_class"})
        self.assertEqual([target_for_seed(seed) for seed in range(10)], [1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
        scatter = trigger_for("S-scatter", 0)
        self.assertEqual(scatter.shape, "scattered")
        self.assertEqual(len(scatter.pixels), 9)
        self.assertEqual(scatter.position, "scattered")
        self.assertIn("position", _changed("S-scatter"))
        self.assertEqual(scatter.poison_rate, 0.08)
        self.assertEqual(scatter.target_class, 0)
        self.assertEqual(len(arm_ids()), 7)
        self.assertEqual(case_order_seed(), 20261111)

    def test_same_poison_rate_keeps_the_batch_order(self):
        from cnn_phase1.plan import make_plan

        anchor = make_plan(seed=3, n_examples=500, poison_rate=0.08, holdout_size=20, epochs=1, batch_size=8)
        center = make_plan(seed=3, n_examples=500, poison_rate=0.08, holdout_size=20, epochs=1, batch_size=8)
        rare = make_plan(seed=3, n_examples=500, poison_rate=0.003, holdout_size=20, epochs=1, batch_size=8)
        self.assertEqual(anchor["order_sha256"], center["order_sha256"])
        self.assertNotEqual(anchor["n_poison"], rare["n_poison"])
        self.assertEqual(rare["n_poison"], int(500 * 0.003))


class CaseTests(unittest.TestCase):
    def test_failed_models_are_excluded_and_the_shuffle_is_fixed(self):
        with tempfile.TemporaryDirectory(prefix="p2c-") as tmp:
            root = Path(tmp)
            train = root / "train"
            weights = train / "w.npz"
            train.mkdir()
            np.savez(weights, value=np.zeros(2, dtype=np.float32))
            digest = sha256_file(weights)
            models = [
                _model("P-center", 0, "backdoor", weights, digest, passed=True),
                _model("P-center", 1, "backdoor", weights, digest, passed=False),
                _model("clean", 0, "clean", weights, digest, passed=True),
            ]
            manifest = {
                "study": "cnn-phase2",
                "recipe": {"trigger": phase1_trigger().to_dict()},
                "models": models,
                "failures": [],
            }
            (train / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            first = make_phase2_cases(
                train_out=train, blind=root / "blind-a", private=root / "private-a", case_order_seed=20261111
            )
            second = make_phase2_cases(
                train_out=train, blind=root / "blind-b", private=root / "private-b", case_order_seed=20261111
            )
            self.assertEqual(len(first["cases"]), 2)
            self.assertEqual(len(first["excluded"]), 1)
            self.assertEqual(first["excluded"][0]["seed"], 1)
            order_a = [(row["arm"], row["seed"]) for row in first["cases"].values()]
            order_b = [(row["arm"], row["seed"]) for row in second["cases"].values()]
            self.assertEqual(order_a, order_b)
            self.assertTrue(all(row["trigger"] for row in first["cases"].values()))


def _model(arm, seed, role, weights: Path, digest: str, *, passed: bool) -> dict:
    trigger = trigger_for(arm, seed).to_dict() if arm != "clean" else phase1_trigger().to_dict()
    return {
        "arm": arm,
        "seed": seed,
        "role": role,
        "checkpoint": str(weights),
        "weights": str(weights),
        "checkpoint_sha256": digest,
        "weights_sha256": digest,
        "init_sha256": "init",
        "order_sha256": "order",
        "clean_acc": 0.99,
        "asr": 0.5 if role == "clean" else 0.99,
        "passed_thresholds": passed,
        "trigger": trigger,
    }


def _audit_row(arm, seed, *, detect, iou_value, target=True):
    return {
        "arm": arm,
        "seed": seed,
        "source_role": "backdoor",
        "status": "valid",
        "detection_positive": detect,
        "target_correct": target,
        "trigger_iou": iou_value,
        "functional_success": 0.2 if detect else 0.0,
        "weight_precision": 0.4,
        "weight_recall": 0.2,
        "edit_status": "applied",
        "asr_before": 1.0,
        "asr_after": 0.2,
        "acc_before": 0.99,
        "acc_after": 0.98,
        "random_asr": [0.95] * 20,
        "predicted_support": [[0, 0]] if detect else [],
        "edited_fc1_columns": [seed],
    }


class AggregateTests(unittest.TestCase):
    def test_center_and_rare_arms_lose_to_the_anchor(self):
        records = []
        for seed in range(10):
            records.append(_audit_row("P-center", seed, detect=False, iou_value=0.05))
            records.append(_audit_row("R-0.3pct", seed, detect=False, iou_value=0.05))
            records.append(_audit_row("T-vary", seed, detect=True, iou_value=0.2, target=True))
        for seed in range(20):
            records.append(
                {
                    "arm": "clean",
                    "seed": seed,
                    "source_role": "clean",
                    "status": "valid",
                    "detection_positive": False,
                    "label_backdoor": False,
                }
            )
        anchor = {
            "records": [
                {
                    "condition": "real",
                    "source_role": "backdoor",
                    "status": "valid",
                    "seed": seed,
                    "detection_positive": True,
                    "trigger_iou": 0.8,
                }
                for seed in range(10)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval.json"
            anchor_path = root / "anchor.json"
            eval_path.write_text(json.dumps({"records": records, "excluded": []}), encoding="utf-8")
            anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
            result = aggregate_phase2(eval_path=eval_path, anchor_path=anchor_path, baselines_path=None, out=root / "out")
        self.assertTrue(result["holm"]["confirmatory"])
        self.assertTrue(result["pass_gates"]["h1_both_reject"])
        self.assertTrue(result["pass_gates"]["h2_either_rejects"])
        self.assertEqual(result["clean_fpr"]["k"], 0)
        self.assertEqual(result["clean_fpr"]["n"], 20)
        self.assertTrue(result["pass_gates"]["fpr_point_le_0.10"])
        self.assertIn("T-vary", result["exploratory"]["other_arms_vs_anchor"])
        self.assertEqual(result["arms"]["P-center"]["tpr"]["k"], 0)
        self.assertEqual(result["arms"]["P-center"]["tpr"]["n"], 10)
