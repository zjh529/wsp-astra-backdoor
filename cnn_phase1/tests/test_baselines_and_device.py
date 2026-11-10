"""Baseline scores stay finite, and the device helper does not edit the environment."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from cnn_phase1.baselines import neural_cleanse_score, spectral_score, weight_stat_score
from cnn_phase1.device import pick_device
from cnn_phase1.model import TENSOR_SPEC, TinyCNN


def _weights(fill=0.0):
    return {name: np.full(shape, fill, dtype=np.float32) for name, shape in TENSOR_SPEC}


class BaselineTests(unittest.TestCase):
    def test_equal_filters_are_not_outliers_and_a_spike_is(self):
        quiet = weight_stat_score(_weights(0.01))
        self.assertFalse(quiet["detected"])
        spiked = _weights(0.01)
        spiked["conv1.weight"][0] = 50
        loud = weight_stat_score(spiked)
        self.assertTrue(loud["detected"])
        self.assertGreaterEqual(loud["max_abs_z"], 3.5)

    def test_spectral_result_is_json_safe(self):
        weights = _weights(0.0)
        weights["fc2.weight"][:] = 1
        result = spectral_score(weights)
        self.assertTrue(result["detected"])
        json.dumps(result)

    def test_neural_cleanse_one_step_returns_ten_scores(self):
        model = TinyCNN()
        batch = torch.rand(4, 1, 28, 28)
        images = torch.rand(6, 1, 28, 28)
        labels = torch.randint(0, 10, (6,))
        result = neural_cleanse_score(
            model, batch, images, labels, torch.device("cpu"), steps=1
        )
        self.assertEqual(len(result["anomaly"]), 10)
        self.assertIsInstance(result["detected"], bool)
        json.dumps(result)


class DeviceTests(unittest.TestCase):
    def test_empty_visible_devices_force_cpu_without_writing_the_variable(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('os.environ["CUDA_VISIBLE_DEVICES"] =', text)
            self.assertNotIn("os.environ['CUDA_VISIBLE_DEVICES'] =", text)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            with mock.patch("torch.cuda.is_available", return_value=True):
                self.assertEqual(pick_device().type, "cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            with mock.patch("torch.cuda.is_available", return_value=True):
                self.assertEqual(pick_device().type, "cuda")
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
