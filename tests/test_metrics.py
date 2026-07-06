from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.metrics import (
    binding_contrastive_loss,
    classification_accuracy,
    mean_sink_margin,
    rank_slot_accuracy,
    rank_slot_supervision_loss,
    target_peak_accuracy,
    target_set_accuracy,
)
from organism_v01.tasks import generate_multi_pair_batch, generate_routing_batch


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

    def test_target_set_accuracy_requires_every_sink(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=6, grid_size=12, layout=layout, seed=79)
        final_state = torch.zeros_like(batch.initial)

        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            label = int(batch.labels[item])
            final_state[item, layout.output_start + label, sink_row, sink_col] = 5.0

        self.assertEqual(target_set_accuracy(final_state, batch, layout), 1.0)

        first_label = int(batch.labels[0])
        first_row, first_col = [int(value) for value in batch.sink_rc[0]]
        final_state[0, layout.output_start + first_label, first_row, first_col] = -5.0
        self.assertLess(target_set_accuracy(final_state, batch, layout), 1.0)

    def test_binding_contrastive_loss_rewards_paired_endpoint_codes(self) -> None:
        layout = ChannelLayout(hidden_channels=6)
        batch = generate_multi_pair_batch(
            batch_size=2,
            grid_size=12,
            layout=layout,
            pair_count=3,
            sink_assignment="reverse",
            seed=80,
        )
        final_state = torch.zeros_like(batch.initial)
        basis = torch.eye(layout.hidden_channels)[:3] * 4.0
        assert batch.pair_source_rc is not None
        assert batch.pair_sink_rc is not None

        for item in range(batch.initial.shape[0]):
            for pair_index in range(3):
                source_row, source_col = [int(value) for value in batch.pair_source_rc[item, pair_index]]
                sink_row, sink_col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
                final_state[item, layout.hidden_slice, source_row, source_col] = basis[pair_index]
                final_state[item, layout.hidden_slice, sink_row, sink_col] = basis[pair_index]

        matched_loss = binding_contrastive_loss(final_state, batch, layout, temperature=0.1)

        swapped_state = final_state.clone()
        for item in range(batch.initial.shape[0]):
            sink_row, sink_col = [int(value) for value in batch.pair_sink_rc[item, 0]]
            swapped_state[item, layout.hidden_slice, sink_row, sink_col] = basis[1]
        swapped_loss = binding_contrastive_loss(swapped_state, batch, layout, temperature=0.1)

        self.assertTrue(torch.isfinite(matched_loss))
        self.assertLess(float(matched_loss), float(swapped_loss))

    def test_binding_contrastive_loss_ignores_single_pair_batches(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=2, grid_size=12, layout=layout, seed=81)
        final_state = torch.zeros_like(batch.initial)

        loss = binding_contrastive_loss(final_state, batch, layout)

        self.assertEqual(float(loss), 0.0)

    def test_rank_slot_supervision_loss_rewards_correct_sink_slots(self) -> None:
        layout = ChannelLayout(hidden_channels=20, rule_channels=1)
        batch = generate_multi_pair_batch(
            batch_size=2,
            grid_size=12,
            layout=layout,
            pair_count=3,
            sink_assignment="reverse",
            seed=82,
        )
        final_state = torch.zeros_like(batch.initial)
        slot_slice = slice(layout.hidden_start + 12, layout.hidden_start + 18)
        assert batch.pair_labels is not None
        assert batch.pair_sink_rc is not None

        for item in range(batch.initial.shape[0]):
            target = torch.full((6,), -5.0)
            for rank_index in range(3):
                label = int(batch.pair_labels[item, rank_index])
                target[rank_index * 2 + label] = 5.0
            for sink_index in range(3):
                row, col = [int(value) for value in batch.pair_sink_rc[item, sink_index]]
                final_state[item, slot_slice, row, col] = target

        matched_loss = rank_slot_supervision_loss(final_state, batch, layout)
        matched_accuracy = rank_slot_accuracy(final_state, batch, layout)
        wrong_state = final_state.clone()
        first_row, first_col = [int(value) for value in batch.pair_sink_rc[0, 0]]
        wrong_state[0, slot_slice, first_row, first_col] = -wrong_state[0, slot_slice, first_row, first_col]
        wrong_loss = rank_slot_supervision_loss(wrong_state, batch, layout)
        wrong_accuracy = rank_slot_accuracy(wrong_state, batch, layout)

        self.assertTrue(torch.isfinite(matched_loss))
        self.assertLess(float(matched_loss), float(wrong_loss))
        self.assertEqual(matched_accuracy, 1.0)
        self.assertLess(wrong_accuracy, 1.0)

    def test_rank_slot_supervision_loss_teaches_two_pair_top_and_bottom_slots(self) -> None:
        layout = ChannelLayout(hidden_channels=20, rule_channels=1)
        batch = generate_multi_pair_batch(
            batch_size=2,
            grid_size=12,
            layout=layout,
            pair_count=2,
            sink_assignment="reverse",
            seed=83,
        )
        final_state = torch.zeros_like(batch.initial)
        slot_slice = slice(layout.hidden_start + 12, layout.hidden_start + 18)
        assert batch.pair_labels is not None
        assert batch.pair_sink_rc is not None

        for item in range(batch.initial.shape[0]):
            target = torch.full((6,), -5.0)
            top_label = int(batch.pair_labels[item, 0])
            bottom_label = int(batch.pair_labels[item, 1])
            target[top_label] = 5.0
            target[4 + bottom_label] = 5.0
            for sink_index in range(2):
                row, col = [int(value) for value in batch.pair_sink_rc[item, sink_index]]
                final_state[item, slot_slice, row, col] = target

        matched_loss = rank_slot_supervision_loss(final_state, batch, layout)
        matched_accuracy = rank_slot_accuracy(final_state, batch, layout)
        wrong_state = final_state.clone()
        first_row, first_col = [int(value) for value in batch.pair_sink_rc[0, 0]]
        wrong_state[0, slot_slice, first_row, first_col] = -wrong_state[0, slot_slice, first_row, first_col]
        wrong_loss = rank_slot_supervision_loss(wrong_state, batch, layout)
        wrong_accuracy = rank_slot_accuracy(wrong_state, batch, layout)

        self.assertTrue(torch.isfinite(matched_loss))
        self.assertLess(float(matched_loss), float(wrong_loss))
        self.assertEqual(matched_accuracy, 1.0)
        self.assertLess(wrong_accuracy, 1.0)

    def test_rank_slot_supervision_loss_teaches_single_pair_as_top_and_bottom(self) -> None:
        layout = ChannelLayout(hidden_channels=20, rule_channels=1)
        batch = generate_multi_pair_batch(
            batch_size=2,
            grid_size=12,
            layout=layout,
            pair_count=1,
            sink_assignment="aligned",
            seed=84,
        )
        final_state = torch.zeros_like(batch.initial)
        slot_slice = slice(layout.hidden_start + 12, layout.hidden_start + 18)
        assert batch.pair_labels is not None
        assert batch.pair_sink_rc is not None

        for item in range(batch.initial.shape[0]):
            target = torch.full((6,), -5.0)
            label = int(batch.pair_labels[item, 0])
            target[label] = 5.0
            target[4 + label] = 5.0
            row, col = [int(value) for value in batch.pair_sink_rc[item, 0]]
            final_state[item, slot_slice, row, col] = target

        matched_loss = rank_slot_supervision_loss(final_state, batch, layout)
        matched_accuracy = rank_slot_accuracy(final_state, batch, layout)
        wrong_state = final_state.clone()
        first_row, first_col = [int(value) for value in batch.pair_sink_rc[0, 0]]
        wrong_state[0, slot_slice, first_row, first_col] = -wrong_state[0, slot_slice, first_row, first_col]
        wrong_loss = rank_slot_supervision_loss(wrong_state, batch, layout)
        wrong_accuracy = rank_slot_accuracy(wrong_state, batch, layout)

        self.assertTrue(torch.isfinite(matched_loss))
        self.assertLess(float(matched_loss), float(wrong_loss))
        self.assertEqual(matched_accuracy, 1.0)
        self.assertLess(wrong_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
