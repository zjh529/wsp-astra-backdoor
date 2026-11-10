"""Delta application, matched random edits, and receptive-field overlap."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from cnn_phase1.edit import N_RANDOM, apply_delta, delta_l2, delta_nnz, matched_random_edits
from cnn_phase1.localize import fc1_input_columns, iou, overlapping_cells
from cnn_phase1.measure import with_delta
from cnn_phase1.model import TENSOR_SPEC, TinyCNN
from cnn_phase1.spec import make_trigger, phase1_trigger


def _zeros():
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in TENSOR_SPEC}


class EditTests(unittest.TestCase):
    def test_apply_delta_adds_and_leaves_other_tensors(self):
        weights = _zeros()
        weights["fc2.bias"][0] = 1.0
        delta = {"fc2.bias": np.zeros(10, dtype=np.float32)}
        delta["fc2.bias"][0] = np.float32(0.25)
        edited = apply_delta(weights, delta)
        self.assertAlmostEqual(float(edited["fc2.bias"][0]), 1.25, places=6)
        self.assertEqual(float(edited["fc2.weight"][0, 0]), 0.0)
        self.assertTrue(np.array_equal(edited["conv1.weight"], weights["conv1.weight"]))

    def test_random_edits_match_count_and_l2(self):
        delta = _zeros()
        delta["fc1.weight"].reshape(-1)[:4] = np.array([0.5, -0.25, 0.25, 0.1], dtype=np.float32)
        delta["fc2.bias"][3] = np.float32(0.4)
        edits = matched_random_edits(delta, n=N_RANDOM, seed=7)
        again = matched_random_edits(delta, n=N_RANDOM, seed=7)
        self.assertEqual(len(edits), 20)
        expected = {name: count for name, count in delta_nnz(delta).items() if count}
        for edit, replay in zip(edits, again):
            self.assertEqual(set(edit), set(expected))
            for name, count in expected.items():
                self.assertEqual(int(np.count_nonzero(edit[name])), count)
                self.assertAlmostEqual(delta_l2({name: edit[name]}), delta_l2({name: delta[name]}), places=4)
            self.assertTrue(np.allclose(edit["fc1.weight"], replay["fc1.weight"]))
            self.assertTrue(np.allclose(edit["fc2.bias"], replay["fc2.bias"]))

    def test_torch_delta_matches_numpy(self):
        model = TinyCNN()
        delta = {"fc2.bias": np.zeros(10, dtype=np.float32)}
        delta["fc2.bias"][1] = 0.5
        edited = with_delta(model, delta, torch.device("cpu"))
        delta_value = float((edited.fc2.bias[1] - model.fc2.bias[1]).detach())
        self.assertAlmostEqual(delta_value, 0.5, places=5)
        self.assertTrue(torch.equal(edited.conv1.weight, model.conv1.weight))


class LocalizeTests(unittest.TestCase):
    def test_top_left_and_bottom_right_cells(self):
        mask, _pattern = phase1_trigger().mask_and_pattern()
        self.assertEqual(set(overlapping_cells(mask)), {(0, 0), (0, 1), (1, 0), (1, 1)})
        corner, _pattern = make_trigger(
            position="bottom_right", shape="square", size=3, target_class=0, poison_rate=0.08
        ).mask_and_pattern()
        self.assertEqual(set(overlapping_cells(corner)), {(5, 5), (5, 6), (6, 5), (6, 6)})
        pixel = np.zeros((28, 28), dtype=np.float32)
        pixel[0, 0] = 1
        self.assertEqual(overlapping_cells(pixel), [(0, 0)])
        columns = fc1_input_columns(pixel)
        self.assertEqual(len(columns), 32)
        self.assertIn(0, columns)
        self.assertEqual(iou(mask, mask), 1.0)
        self.assertEqual(iou(np.zeros_like(mask), np.zeros_like(mask)), 1.0)
