from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.controls import erase_sink_from_input, erase_source, swap_source_label
from organism_v01.tasks import generate_memory_batch, generate_multi_pair_batch, generate_routing_batch


class ControlTransformTests(unittest.TestCase):
    def test_erase_source_removes_input_without_changing_target(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=8, grid_size=12, layout=layout, seed=91)
        transformed = erase_source(batch, layout)

        self.assertEqual(float(transformed.initial[:, layout.source_a].sum()), 0.0)
        self.assertEqual(float(transformed.initial[:, layout.source_b].sum()), 0.0)
        self.assertEqual(float(transformed.env[:, layout.source_a].sum()), 0.0)
        self.assertEqual(float(transformed.env[:, layout.source_b].sum()), 0.0)
        self.assertTrue((transformed.target == batch.target).all())
        self.assertTrue((transformed.labels == batch.labels).all())

    def test_swap_source_label_flips_visible_source_channel(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=8, grid_size=12, layout=layout, seed=92)
        transformed = swap_source_label(batch, layout)

        self.assertTrue((transformed.initial[:, layout.source_a] == batch.initial[:, layout.source_b]).all())
        self.assertTrue((transformed.initial[:, layout.source_b] == batch.initial[:, layout.source_a]).all())
        self.assertTrue((transformed.target == batch.target).all())
        self.assertTrue((transformed.labels == batch.labels).all())

    def test_erase_source_removes_memory_input_source(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_memory_batch(batch_size=8, grid_size=12, layout=layout, seed=93)
        transformed = erase_source(batch, layout)

        self.assertIsNotNone(transformed.input_env)
        self.assertEqual(float(transformed.input_env[:, layout.source_a].sum()), 0.0)
        self.assertEqual(float(transformed.input_env[:, layout.source_b].sum()), 0.0)

    def test_erase_sink_removes_sink_route_cues(self) -> None:
        layout = ChannelLayout(hidden_channels=4, route_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            pair_count=3,
            sink_assignment="reverse",
            seed=95,
        )
        transformed = erase_sink_from_input(batch, layout)
        sink_mask = batch.sink_mask.bool().expand(-1, layout.route_channels, -1, -1)

        self.assertEqual(float(transformed.initial[:, layout.sink].sum()), 0.0)
        self.assertEqual(float(transformed.initial[:, layout.route_slice].masked_select(sink_mask).sum()), 0.0)
        self.assertEqual(float(transformed.env[:, layout.route_slice].masked_select(sink_mask).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
