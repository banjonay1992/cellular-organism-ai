from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.metrics import compute_loss, rank_claim_supervision_loss, rank_slot_accuracy, rank_slot_routed_accuracy
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import generate_memory_batch, generate_multi_pair_batch, generate_routing_batch


class OrganismTests(unittest.TestCase):
    def test_rollout_preserves_environment_and_blocks_mutable_state(self) -> None:
        torch.manual_seed(5)
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            damage_prob=0.2,
            seed=5,
        )
        model = CellularOrganism(layout=layout, cell_hidden=16)

        rollout = model(batch, steps=3)

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        blocked_mask = 1.0 - batch.alive_mask
        blocked_mutable = (rollout.final_state[:, layout.mutable_slice] * blocked_mask).detach()
        self.assertLess(float(blocked_mutable.abs().max()), 1e-6)

    def test_loss_is_finite_and_backpropagates(self) -> None:
        torch.manual_seed(11)
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=11)
        model = CellularOrganism(layout=layout, cell_hidden=16)

        rollout = model(batch, steps=2)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_gated_message_rollout_backpropagates(self) -> None:
        torch.manual_seed(14)
        layout = ChannelLayout(hidden_channels=6)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=14)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="gated_message",
            message_slots=3,
        )

        rollout = model(batch, steps=2)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_self_tagging_rollout_backpropagates(self) -> None:
        torch.manual_seed(15)
        layout = ChannelLayout(hidden_channels=6)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=15)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="self_tagging",
            tag_slots=3,
        )

        rollout = model(batch, steps=2)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rank_binding_rollout_backpropagates(self) -> None:
        torch.manual_seed(16)
        layout = ChannelLayout(hidden_channels=6)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=16)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_binding",
        )

        rollout = model(batch, steps=3)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        rank_wave_energy = rollout.final_state[:, layout.hidden_start : layout.hidden_start + 4].detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(rank_wave_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_sink_stabilized_rank_anchors_signal_at_sink(self) -> None:
        torch.manual_seed(17)
        layout = ChannelLayout(hidden_channels=10)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=17)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="sink_stabilized_rank",
        )

        rollout = model(batch, steps=12)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        source_at_sink_channels = slice(layout.hidden_start + 4, layout.hidden_start + 6)
        sink_anchor_energy = (rollout.final_state[:, source_at_sink_channels] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(sink_anchor_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_matching_readout_anchors_label_signal_at_sink_and_backpropagates(self) -> None:
        torch.manual_seed(18)
        layout = ChannelLayout(hidden_channels=14)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=18)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="matching_readout",
        )

        rollout = model(batch, steps=12)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        label_wave_channels = slice(layout.hidden_start + 8, layout.hidden_start + 12)
        label_wave_energy = (rollout.final_state[:, label_wave_channels] * batch.sink_mask).detach().abs().sum()
        output_energy = (rollout.final_state[:, layout.output_slice] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(label_wave_energy), 0.0)
        self.assertGreater(float(output_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rule_cued_matching_readout_uses_rule_context_and_backpropagates(self) -> None:
        torch.manual_seed(19)
        layout = ChannelLayout(hidden_channels=14, rule_channels=1)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=19)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rule_cued_matching_readout",
        )

        rollout = model(batch, steps=12)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        output_energy = (rollout.final_state[:, layout.output_slice] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(output_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rule_cued_matching_readout_requires_rule_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=14)

        with self.assertRaises(ValueError):
            CellularOrganism(
                layout=layout,
                cell_hidden=16,
                update_rule="rule_cued_matching_readout",
            )

    def test_rank_slot_rule_cued_readout_moves_slot_signals_to_sink(self) -> None:
        torch.manual_seed(20)
        layout = ChannelLayout(hidden_channels=20, rule_channels=1)
        batch = generate_routing_batch(batch_size=3, grid_size=10, layout=layout, seed=20)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_rule_cued",
        )

        rollout = model(batch, steps=12)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        slot_channels = slice(layout.hidden_start + 12, layout.hidden_start + 18)
        slot_energy = (rollout.final_state[:, slot_channels] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(slot_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rank_slot_rule_cued_forms_clean_three_pair_slot_organ(self) -> None:
        torch.manual_seed(21)
        layout = ChannelLayout(hidden_channels=32, rule_channels=1)
        batch = generate_multi_pair_batch(
            batch_size=16,
            grid_size=12,
            layout=layout,
            pair_count=3,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=125,
        )
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_rule_cued",
        )

        rollout = model(batch, steps=96)

        self.assertGreaterEqual(rank_slot_accuracy(rollout.final_state, batch, layout), 0.70)
        self.assertGreaterEqual(rank_slot_routed_accuracy(rollout.final_state, batch, layout), 0.90)

    def test_rank_slot_rule_cued_one_hot_rule_presence_gates_sink_readout(self) -> None:
        torch.manual_seed(22)
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=2,
            grid_size=12,
            layout=layout,
            pair_count=3,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=126,
        )
        blank_rule_initial = batch.initial.clone()
        blank_rule_initial[:, layout.rule_slice] = 0.0
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_rule_cued",
        )

        valid_delta = model.cell_update(batch.initial)
        blank_delta = model.cell_update(blank_rule_initial)

        valid_output_energy = (valid_delta[:, layout.output_slice] * batch.sink_mask).abs().sum().detach()
        blank_output_energy = (blank_delta[:, layout.output_slice] * batch.sink_mask).abs().sum().detach()
        self.assertGreater(float(valid_output_energy), 0.0)
        self.assertLess(float(blank_output_energy), float(valid_output_energy) * 0.5)

    def test_rank_slot_repair_rule_cued_builds_repair_bus_and_backpropagates(self) -> None:
        torch.manual_seed(24)
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            pair_count=4,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=128,
        )
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_repair_rule_cued",
        )

        rollout = model(batch, steps=32)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        repair_channels = slice(layout.hidden_start + 20, layout.hidden_start + 24)
        repair_energy = rollout.final_state[:, repair_channels].detach().abs().sum()
        sink_repair_energy = (rollout.final_state[:, repair_channels] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(repair_energy), 0.0)
        self.assertGreater(float(sink_repair_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rank_slot_repair_rule_cued_requires_repair_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=20, rule_channels=3)

        with self.assertRaises(ValueError):
            CellularOrganism(
                layout=layout,
                cell_hidden=16,
                update_rule="rank_slot_repair_rule_cued",
            )

    def test_rank_slot_claim_rule_cued_builds_claim_organ_and_backpropagates(self) -> None:
        torch.manual_seed(25)
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=4,
            grid_size=12,
            layout=layout,
            pair_count=4,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=129,
        )
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_claim_rule_cued",
        )

        rollout = model(batch, steps=40)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        claim_loss = rank_claim_supervision_loss(rollout.final_state, batch, layout)
        (losses["total"] + claim_loss * 0.1).backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        source_rank_label_channels = slice(layout.hidden_start + 20, layout.hidden_start + 28)
        claim_channels = slice(layout.hidden_start + 28, layout.hidden_start + 32)
        source_rank_label_energy = (rollout.final_state[:, source_rank_label_channels] * batch.sink_mask).detach().abs().sum()
        claim_energy = (rollout.final_state[:, claim_channels] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertGreater(float(source_rank_label_energy), 0.0)
        self.assertGreater(float(claim_energy), 0.0)
        self.assertTrue(torch.isfinite(claim_loss))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_rank_slot_claim_rule_cued_requires_claim_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=31, rule_channels=3)

        with self.assertRaises(ValueError):
            CellularOrganism(
                layout=layout,
                cell_hidden=16,
                update_rule="rank_slot_claim_rule_cued",
            )

    def test_rank_slot_claim_residual_rule_cued_starts_with_quiet_trainable_gate(self) -> None:
        layout = ChannelLayout(hidden_channels=44, rule_channels=3)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_claim_residual_rule_cued",
        )
        cell_update = model.cell_update
        self.assertTrue(hasattr(cell_update, "claim_gate"))

        gate_input_channels = cell_update.claim_gate[0].in_channels
        generic_output = torch.full((1, 2, 3, 3), 0.5)
        base_output = torch.zeros(1, 2, 3, 3)
        claim_output = torch.full_like(base_output, 10.0)
        claim_context = torch.zeros(1, gate_input_channels, 3, 3)

        gated_output = cell_update._claim_output_delta(generic_output, base_output, claim_output, claim_context)
        gated_output.sum().backward()

        self.assertGreater(float(gated_output.detach().min()), 0.5)
        self.assertLess(float(gated_output.detach().max()), 0.7)
        self.assertIsNotNone(cell_update.claim_gate[-1].bias.grad)
        self.assertGreater(float(cell_update.claim_gate[-1].bias.grad.abs().sum()), 0.0)

    def test_rank_slot_claim_residual_rule_cued_anchors_claim_target_at_sink(self) -> None:
        layout = ChannelLayout(hidden_channels=44, rule_channels=3)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_claim_residual_rule_cued",
        )
        cell_update = model.cell_update
        claim_state = torch.zeros(1, 4, 3, 3)
        claim_seed = torch.zeros_like(claim_state)
        sink_marker = torch.zeros(1, 1, 3, 3)
        claim_seed[:, :, 1, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        sink_marker[:, :, 1, 1] = 1.0

        target = cell_update._claim_target_from_seed(claim_state, claim_seed, sink_marker)

        self.assertTrue(torch.equal(target[:, :, 1, 1], claim_seed[:, :, 1, 1]))

    def test_rank_slot_claim_residual_rule_cued_requires_extra_claim_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=43, rule_channels=3)

        with self.assertRaises(ValueError):
            CellularOrganism(
                layout=layout,
                cell_hidden=16,
                update_rule="rank_slot_claim_residual_rule_cued",
            )

    def test_relative_rank_rule_cued_separates_four_source_ranks(self) -> None:
        torch.manual_seed(23)
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=8,
            grid_size=14,
            layout=layout,
            pair_count=4,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=127,
        )
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="relative_rank_rule_cued",
        )

        rollout = model(batch, steps=96)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
        )
        losses["total"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        assert batch.pair_source_rc is not None
        source_coordinates = []
        for rank_index in range(4):
            values = []
            for item in range(batch.initial.shape[0]):
                row, col = [int(value) for value in batch.pair_source_rc[item, rank_index]]
                down = rollout.final_state[item, layout.hidden_start + 12, row, col]
                up = rollout.final_state[item, layout.hidden_start + 13, row, col]
                values.append((down - up) / (down + up).clamp_min(1.0))
            source_coordinates.append(float(torch.stack(values).mean().detach()))
        moment_channels = slice(layout.hidden_start + 16, layout.hidden_start + 24)
        moment_energy = (rollout.final_state[:, moment_channels] * batch.sink_mask).detach().abs().sum()

        self.assertTrue(torch.allclose(rollout.final_state[:, : layout.env_count], batch.env))
        self.assertLess(source_coordinates[0], source_coordinates[1])
        self.assertLess(source_coordinates[1], source_coordinates[2])
        self.assertLess(source_coordinates[2], source_coordinates[3])
        self.assertGreater(source_coordinates[2] - source_coordinates[1], 0.05)
        self.assertGreater(float(moment_energy), 0.0)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertGreater(grad_norm, 0.0)

    def test_relative_rank_rule_cued_requires_rule_channels(self) -> None:
        layout = ChannelLayout(hidden_channels=32)

        with self.assertRaises(ValueError):
            CellularOrganism(
                layout=layout,
                cell_hidden=16,
                update_rule="relative_rank_rule_cued",
            )

    def test_rollout_returns_frames_and_can_continue(self) -> None:
        torch.manual_seed(12)
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=2, grid_size=10, layout=layout, seed=12)
        model = CellularOrganism(layout=layout, cell_hidden=16)

        first = model(batch, steps=2, return_frames=True)
        second = model(batch, steps=2, start_state=first.final_state, start_step=2)

        self.assertIsNotNone(first.frames)
        self.assertEqual(tuple(first.frames.shape[:2]), (3, 2))
        self.assertEqual(tuple(second.final_state.shape), tuple(batch.initial.shape))

    def test_memory_input_env_is_visible_only_initially(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_memory_batch(
            batch_size=2,
            grid_size=10,
            layout=layout,
            input_steps=1,
            seed=13,
        )
        model = CellularOrganism(layout=layout, cell_hidden=16)

        rollout = model(batch, steps=2, return_frames=True)

        self.assertIsNotNone(rollout.frames)
        first_source = rollout.frames[0, :, layout.source_a : layout.source_b + 1].sum()
        final_source = rollout.frames[-1, :, layout.source_a : layout.source_b + 1].sum()
        self.assertGreater(float(first_source), 0.0)
        self.assertEqual(float(final_source), 0.0)


if __name__ == "__main__":
    unittest.main()
