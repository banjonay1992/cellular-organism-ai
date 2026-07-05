from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.tasks import generate_memory_batch, generate_multi_pair_batch, generate_routing_batch


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

    def test_maze_task_adds_a_wall_with_a_gap(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(
            batch_size=4,
            grid_size=16,
            layout=layout,
            damage_prob=0.0,
            maze_barrier=True,
            seed=555,
        )

        wall_col = batch.initial.shape[-1] // 2
        wall = batch.initial[:, layout.blocked, 1:-1, wall_col]
        self.assertTrue((wall.sum(dim=1) >= wall.shape[1] - 1).all())
        self.assertEqual(batch.task_name, "maze")

    def test_multi_pair_task_has_multiple_targets(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=5,
            grid_size=14,
            layout=layout,
            pair_count=3,
            seed=808,
        )

        self.assertEqual(batch.task_name, "multi")
        self.assertIsNotNone(batch.pair_labels)
        self.assertEqual(tuple(batch.pair_labels.shape), (5, 3))
        self.assertEqual(float(batch.sink_mask.sum()), 15.0)
        self.assertEqual(float(batch.target.sum()), 15.0)

    def test_multi_pair_spacing_keeps_rows_apart(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=12,
            grid_size=16,
            layout=layout,
            pair_count=3,
            min_pair_spacing=3,
            seed=809,
        )

        self.assertIsNotNone(batch.pair_sink_rc)
        rows = batch.pair_sink_rc[:, :, 0]
        gaps = rows[:, 1:] - rows[:, :-1]
        self.assertTrue((gaps >= 3).all())

    def test_multi_pair_can_generate_single_pair_for_curriculum(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=5,
            grid_size=12,
            layout=layout,
            pair_count=1,
            seed=810,
        )

        self.assertEqual(float(batch.sink_mask.sum()), 5.0)
        self.assertEqual(float(batch.target.sum()), 5.0)

    def test_multi_pair_can_generate_adjacent_rows_without_spacing_crutch(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=8,
            grid_size=10,
            layout=layout,
            pair_count=3,
            min_pair_spacing=1,
            seed=0,
        )

        self.assertIsNotNone(batch.pair_source_rc)
        rows = batch.pair_source_rc[:, :, 0]
        gaps = rows[:, 1:] - rows[:, :-1]
        self.assertTrue((gaps == 1).any())

    def test_reverse_sink_assignment_creates_crossing_targets(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=6,
            grid_size=12,
            layout=layout,
            pair_count=3,
            min_pair_spacing=1,
            sink_assignment="reverse",
            seed=811,
        )

        self.assertEqual(batch.task_name, "multi_cross")
        self.assertIsNotNone(batch.pair_source_rc)
        self.assertIsNotNone(batch.pair_sink_rc)
        source_rows = batch.pair_source_rc[:, :, 0]
        sink_rows = batch.pair_sink_rc[:, :, 0]
        self.assertTrue(torch.equal(sink_rows, source_rows.flip(1)))

        for item in range(batch.initial.shape[0]):
            for pair_index in range(3):
                label = int(batch.pair_labels[item, pair_index])
                sink_row, sink_col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
                self.assertEqual(float(batch.target[item, label, sink_row, sink_col]), 1.0)

    def test_cycle_sink_assignment_rotates_sink_rows(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_multi_pair_batch(
            batch_size=6,
            grid_size=12,
            layout=layout,
            pair_count=3,
            min_pair_spacing=1,
            sink_assignment="cycle",
            seed=814,
        )

        self.assertEqual(batch.task_name, "multi_cross")
        self.assertIsNotNone(batch.pair_source_rc)
        self.assertIsNotNone(batch.pair_sink_rc)
        source_rows = batch.pair_source_rc[:, :, 0]
        sink_rows = batch.pair_sink_rc[:, :, 0]
        self.assertTrue(torch.equal(sink_rows, torch.roll(source_rows, shifts=-1, dims=1)))

    def test_route_cues_mark_matching_source_and_sink(self) -> None:
        layout = ChannelLayout(hidden_channels=4, route_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            pair_count=3,
            min_pair_spacing=1,
            sink_assignment="reverse",
            seed=812,
        )

        for item in range(batch.initial.shape[0]):
            for pair_index in range(3):
                route_channel = layout.route_start + pair_index
                source_row, source_col = [int(value) for value in batch.pair_source_rc[item, pair_index]]
                sink_row, sink_col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
                self.assertEqual(float(batch.initial[item, route_channel, source_row, source_col]), 1.0)
                self.assertEqual(float(batch.initial[item, route_channel, sink_row, sink_col]), 1.0)

    def test_route_cues_require_enough_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=4, route_channels=2)

        with self.assertRaises(ValueError):
            generate_multi_pair_batch(
                batch_size=1,
                grid_size=12,
                layout=layout,
                pair_count=3,
                seed=813,
            )

    def test_memory_task_hides_source_after_input_phase(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_memory_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            input_steps=3,
            seed=909,
        )

        self.assertEqual(batch.task_name, "memory")
        self.assertIsNotNone(batch.input_env)
        self.assertEqual(batch.input_steps, 3)
        self.assertGreater(float(batch.input_env[:, layout.source_a : layout.source_b + 1].sum()), 0.0)
        self.assertEqual(float(batch.env[:, layout.source_a : layout.source_b + 1].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
