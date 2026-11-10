"""Clean and backdoor pairs share init and batch order."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from cnn_phase1.plan import gather_batch, initial_state_dict, make_plan
from cnn_phase1.spec import phase1_recipe, phase1_trigger, Recipe
from cnn_phase1.train import state_dict_sha256, train_study


class PairingTests(unittest.TestCase):
    def test_to_tensor_scaling(self):
        image = Image.fromarray(np.arange(28 * 28, dtype=np.uint8).reshape(28, 28), mode="L")
        manual = torch.from_numpy(np.asarray(image)).unsqueeze(0).to(dtype=torch.float32).div(255)
        self.assertTrue(torch.allclose(ToTensor()(image), manual))

    def test_plan_and_init_are_deterministic_and_poison_is_the_only_batch_change(self):
        left = make_plan(seed=0, n_examples=40, poison_rate=0.08, holdout_size=8, epochs=2, batch_size=8)
        right = make_plan(seed=0, n_examples=40, poison_rate=0.08, holdout_size=8, epochs=2, batch_size=8)
        self.assertEqual(left["order_sha256"], right["order_sha256"])
        self.assertEqual(left["epoch_orders"], right["epoch_orders"])
        self.assertEqual(left["poison_indices"], right["poison_indices"])
        self.assertEqual(state_dict_sha256(initial_state_dict(0)), state_dict_sha256(initial_state_dict(0)))
        spec = phase1_trigger()
        images = torch.zeros(40, 1, 28, 28)
        labels = torch.arange(40) % 10
        poison = set(left["poison_indices"])
        indices = [left["train_idx"][i] for i in left["epoch_orders"][0][:8]]
        clean_x, clean_y = gather_batch(images, labels, indices, poison, spec, poison=False)
        back_x, back_y = gather_batch(images, labels, indices, poison, spec, poison=True)
        self.assertEqual(len(indices), 8)
        for position, index in enumerate(indices):
            if index in poison:
                self.assertTrue(torch.all(back_x[position, :, :3, :3] == 1))
                self.assertEqual(int(back_y[position]), 0)
                self.assertFalse(torch.equal(clean_x[position], back_x[position]))
            else:
                self.assertTrue(torch.equal(clean_x[position], back_x[position]))
                self.assertEqual(int(clean_y[position]), int(back_y[position]))

    def test_short_pair_shares_init_and_order_and_is_not_dropped_on_failure(self):
        images = torch.rand(40, 1, 28, 28)
        labels = torch.randint(0, 10, (40,))
        test_images = torch.rand(12, 1, 28, 28)
        test_labels = torch.randint(0, 10, (12,))
        recipe = Recipe(trigger=phase1_trigger(), epochs=1, batch_size=8, holdout_size=8, seeds=(0,))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train"
            manifest = train_study(
                recipe=recipe,
                train_out=out,
                images=images,
                labels=labels,
                test_images=test_images,
                test_labels=test_labels,
                device=torch.device("cpu"),
            )
            roles = {row["role"]: row for row in manifest["models"] if row["seed"] == 0}
            self.assertEqual(set(roles), {"init", "clean", "backdoor"})
            self.assertEqual(roles["clean"]["init_sha256"], roles["backdoor"]["init_sha256"])
            self.assertEqual(roles["clean"]["init_sha256"], roles["init"]["init_sha256"])
            self.assertEqual(roles["clean"]["order_sha256"], roles["backdoor"]["order_sha256"])
            self.assertNotEqual(roles["clean"]["weights_sha256"], roles["backdoor"]["weights_sha256"])
            self.assertGreater(manifest["n_failures"], 0)
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / roles["backdoor"]["checkpoint"]).is_file())
            with self.assertRaises(SystemExit):
                train_study(
                    recipe=recipe,
                    train_out=out,
                    images=images,
                    labels=labels,
                    test_images=test_images,
                    test_labels=test_labels,
                    device=torch.device("cpu"),
                    resume=False,
                )
            again = train_study(
                recipe=recipe,
                train_out=out,
                images=images,
                labels=labels,
                test_images=test_images,
                test_labels=test_labels,
                device=torch.device("cpu"),
                resume=True,
            )
            before = {row["role"]: row["checkpoint_sha256"] for row in manifest["models"]}
            after = {row["role"]: row["checkpoint_sha256"] for row in again["models"]}
            self.assertEqual(before, after)
        self.assertEqual(phase1_recipe().epochs, 3)
        self.assertEqual(phase1_recipe().trigger.poison_rate, 0.08)
