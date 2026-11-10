"""Wilson, Holm, and the phase-1 aggregate on a synthetic frozen table."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cnn_phase1.aggregate import aggregate
from cnn_phase1.stats import holm, mcnemar_greater, signflip_p, wilson_interval


def _row(seed, role, condition, *, detect=False, asr_after=1.0, functional=0.0, iou=1.0):
    return {
        "case_id": f"{condition}-{role}-{seed}",
        "seed": seed,
        "source_role": role,
        "condition": condition,
        "label_backdoor": role == "backdoor",
        "status": "valid",
        "invalid_reason": None,
        "edit_status": "applied",
        "verdict": "backdoor" if detect else "clean",
        "probability": 0.9 if detect else 0.2,
        "target_class": 0 if detect else None,
        "target_correct": bool(detect and role == "backdoor"),
        "detection_positive": detect,
        "functional_success": functional,
        "trigger_iou": iou,
        "weight_precision": 1.0 if detect else None,
        "weight_recall": 0.5 if detect else 0.0,
        "asr_before": 1.0 if role == "backdoor" else 0.01,
        "acc_before": 0.99,
        "asr_after": asr_after,
        "acc_after": 0.989,
        "random_asr": [0.9 if role == "backdoor" else 0.01] * 20,
        "random_acc": [0.98] * 20,
    }


def _table(detect_real_backdoor=True, fpr=0):
    records = []
    for seed in range(10):
        for condition in ("real", "no_weights", "random_init"):
            for role in ("backdoor", "clean"):
                detect = False
                functional = 0.0
                asr_after = 1.0 if role == "backdoor" else 0.01
                iou = 0.0 if role == "backdoor" else 1.0
                if condition == "real" and role == "backdoor" and detect_real_backdoor:
                    detect = True
                    functional = 0.9
                    asr_after = 0.05
                    iou = 1.0
                if condition == "real" and role == "clean" and seed < fpr:
                    detect = True
                    iou = 0.0
                records.append(
                    _row(seed, role, condition, detect=detect, asr_after=asr_after, functional=functional, iou=iou)
                )
    return records


class StatsTests(unittest.TestCase):
    def test_wilson_zero_of_ten_stays_below_one_third(self):
        interval = wilson_interval(0, 10)
        self.assertEqual(interval["estimate"], 0.0)
        self.assertEqual(interval["low"], 0.0)
        self.assertGreater(interval["high"], 0.25)
        self.assertLess(interval["high"], 0.31)

    def test_mcnemar_and_signflip_are_one_over_two_to_the_n(self):
        paired = mcnemar_greater([1, 1, 1], [0, 0, 0])
        self.assertAlmostEqual(paired["p"], 1 / 8)
        flipped = signflip_p([-1.0, -1.0, -1.0], alternative="less")
        self.assertAlmostEqual(flipped["p"], 1 / 8)

    def test_holm_stops_after_the_first_miss(self):
        decision = holm(
            {
                "tpr_real_gt_no_weights": 0.01,
                "tpr_real_gt_random_init": 0.03,
                "asr_astra_lt_random": 0.04,
                "functional_backdoor_gt_clean": 0.20,
            }
        )
        self.assertTrue(decision["tests"]["tpr_real_gt_no_weights"]["reject"])
        self.assertFalse(decision["tests"]["tpr_real_gt_random_init"]["reject"])
        self.assertFalse(decision["tests"]["asr_astra_lt_random"]["reject"])


class AggregateTests(unittest.TestCase):
    def test_separated_rates_and_the_claim_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval_records.json"
            eval_path.write_text(json.dumps({"records": _table()}), encoding="utf-8")
            result = aggregate(eval_path=eval_path, baselines_path=None, out=root / "agg")
        real = result["by_condition"]["real"]
        self.assertEqual(real["tpr"]["k"], 10)
        self.assertEqual(real["tpr"]["n"], 10)
        self.assertEqual(real["fpr"]["k"], 0)
        self.assertEqual(real["fpr"]["n"], 10)
        self.assertEqual(result["by_condition"]["no_weights"]["tpr"]["k"], 0)
        self.assertEqual(result["by_condition"]["random_init"]["fpr"]["k"], 0)
        self.assertTrue(result["holm"]["confirmatory"])
        self.assertTrue(all(result["holm"]["tests"][name]["reject"] for name in result["holm"]["family"]))
        self.assertTrue(result["pass_gates"]["claim_allowed"])
        self.assertLess(real["repair"]["asr_drop_astra"]["mean"], 1.0)
        self.assertGreater(real["repair"]["asr_drop_advantage"]["mean"], 0.5)

    def test_fpr_and_a_usage_limit_close_the_claim(self):
        records = _table(fpr=2)
        records[0]["status"] = "unevaluable"
        records[0]["invalid_reason"] = "usage_limit"
        records[0]["detection_positive"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "eval_records.json"
            eval_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            result = aggregate(eval_path=eval_path, baselines_path=None, out=root / "agg")
        self.assertFalse(result["pass_gates"]["claim_allowed"])
        self.assertGreaterEqual(result["by_condition"]["real"]["fpr"]["estimate"], 0.2)
        self.assertEqual(result["counts"]["real"]["backdoor"]["unevaluable"] + result["counts"]["real"]["clean"]["unevaluable"], 1)
        self.assertFalse(result["holm"]["confirmatory"])
