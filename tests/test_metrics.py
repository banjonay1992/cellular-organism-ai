from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.metrics import classification_accuracy, mean_sink_margin, target_peak_accuracy
from organism_v01.tasks import generate_routing_batch


class MetricTests(unittest.TestCase):
    def test_sink_accuracy_reads_only_the_sink_cell(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=8, grid_size=12, layout=layout, seed=77)
        final_state = torch.zeros_like(batch.initial)

        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            label = int(batch.labels[item])
            final_state[item, layout.output_start + label, sink_row, sink_col] = 4.0
            final_state[item, layout.output_start + (1 - label), sink_row, sink_col] = -4.0
            final_state[item, layout.output_start + (1 - label), 1, 1] = 99.0

        self.assertEqual(classification_accuracy(final_state, batch, layout), 1.0)
        self.assertGreater(mean_sink_margin(final_state, batch, layout), 0.0)

    def test_target_peak_accuracy_requires_correct_cell_and_label(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=8, grid_size=12, layout=layout, seed=78)
        final_state = torch.zeros_like(batch.initial)

        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            label = int(batch.labels[item])
            final_state[item, layout.output_start + label, sink_row, sink_col] = 9.0
            final_state[item, layout.output_start + label, sink_row, 1] = 8.0

        self.assertEqual(target_peak_accuracy(final_state, batch, layout), 1.0)

        first_label = int(batch.labels[0])
        final_state[0, layout.output_start + first_label, 1, 1] = 10.0
        self.assertLess(target_peak_accuracy(final_state, batch, layout), 1.0)


if __name__ == "__main__":
    unittest.main()
