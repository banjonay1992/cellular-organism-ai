from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.tasks import generate_routing_batch


class RoutingTaskTests(unittest.TestCase):
    def test_generation_is_seed_deterministic(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        first = generate_routing_batch(batch_size=6, grid_size=12, layout=layout, seed=123)
        second = generate_routing_batch(batch_size=6, grid_size=12, layout=layout, seed=123)

        self.assertTrue(torch.equal(first.initial, second.initial))
        self.assertTrue(torch.equal(first.target, second.target))
        self.assertTrue(torch.equal(first.labels, second.labels))

    def test_target_matches_sampled_label_and_sink(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=10, grid_size=12, layout=layout, seed=321)

        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            source_row, source_col = [int(value) for value in batch.source_rc[item]]
            label = int(batch.labels[item])
            source_channel = layout.source_a if label == 0 else layout.source_b

            self.assertEqual(float(batch.initial[item, source_channel, source_row, source_col]), 1.0)
            self.assertEqual(float(batch.initial[item, layout.sink, sink_row, sink_col]), 1.0)
            self.assertEqual(float(batch.target[item, label, sink_row, sink_col]), 1.0)
            self.assertEqual(float(batch.target[item].sum()), 1.0)
            self.assertEqual(float(batch.alive_mask[item, 0, sink_row, sink_col]), 1.0)
            self.assertEqual(float(batch.alive_mask[item, 0, source_row, source_col]), 1.0)

    def test_sink_does_not_leave_cleared_damage_patch(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(
            batch_size=64,
            grid_size=16,
            layout=layout,
            damage_prob=0.5,
            seed=444,
        )

        blocked_neighbors = 0.0
        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            patch = batch.initial[
                item,
                layout.blocked,
                max(0, sink_row - 1) : sink_row + 2,
                max(0, sink_col - 1) : sink_col + 2,
            ]
            blocked_neighbors += float(patch.sum())

        self.assertGreater(blocked_neighbors, 0.0)


if __name__ == "__main__":
    unittest.main()
