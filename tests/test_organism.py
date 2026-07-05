from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.metrics import compute_loss
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import generate_memory_batch, generate_routing_batch


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
