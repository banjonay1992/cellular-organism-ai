from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.organism import CellularOrganism
from organism_v01.train import curriculum_batch_params, load_initial_model


class TrainCurriculumTests(unittest.TestCase):
    def test_multi_pair_curriculum_ramps_to_final_task(self) -> None:
        args = argparse.Namespace(
            task="multi",
            curriculum="multi_pair",
            steps=100,
            pair_count=3,
            damage_prob=0.12,
            coordinate_fields=True,
            min_pair_spacing=2,
            sink_assignment="reverse",
            memory_input_steps=4,
        )

        early = curriculum_batch_params(args, 1)
        middle = curriculum_batch_params(args, 60)
        late = curriculum_batch_params(args, 95)

        self.assertEqual(early["pair_count"], 1)
        self.assertEqual(early["damage_prob"], 0.0)
        self.assertEqual(middle["pair_count"], 3)
        self.assertEqual(middle["damage_prob"], 0.0)
        self.assertEqual(late["pair_count"], 3)
        self.assertEqual(late["damage_prob"], 0.12)
        self.assertEqual(late["min_pair_spacing"], 2)
        self.assertEqual(late["sink_assignment"], "reverse")

    def test_multi_pair_curriculum_rejects_other_tasks(self) -> None:
        args = argparse.Namespace(
            task="routing",
            curriculum="multi_pair",
            steps=10,
            pair_count=3,
            damage_prob=0.1,
            coordinate_fields=True,
            min_pair_spacing=1,
            sink_assignment="aligned",
            memory_input_steps=4,
        )

        with self.assertRaises(ValueError):
            curriculum_batch_params(args, 1)

    def test_binding_curriculum_phases_rank_assignments(self) -> None:
        args = argparse.Namespace(
            task="multi",
            curriculum="binding",
            steps=100,
            pair_count=3,
            damage_prob=0.12,
            coordinate_fields=True,
            min_pair_spacing=1,
            sink_assignment="reverse",
            memory_input_steps=4,
        )

        one_pair = curriculum_batch_params(args, 1)
        two_aligned = curriculum_batch_params(args, 20)
        two_reverse = curriculum_batch_params(args, 40)
        three_reverse = curriculum_batch_params(args, 59)
        three_cycle = curriculum_batch_params(args, 80)
        final = curriculum_batch_params(args, 95)

        self.assertEqual((one_pair["pair_count"], one_pair["sink_assignment"]), (1, "aligned"))
        self.assertEqual((two_aligned["pair_count"], two_aligned["sink_assignment"]), (2, "aligned"))
        self.assertEqual((two_reverse["pair_count"], two_reverse["sink_assignment"]), (2, "reverse"))
        self.assertEqual((three_reverse["pair_count"], three_reverse["sink_assignment"]), (3, "reverse"))
        self.assertEqual((three_cycle["pair_count"], three_cycle["sink_assignment"]), (3, "cycle"))
        self.assertEqual((final["pair_count"], final["sink_assignment"]), (3, "reverse"))
        self.assertEqual(final["damage_prob"], 0.12)

    def test_rule_binding_curriculum_requires_rule_channels_and_alternates_final_rules(self) -> None:
        args = argparse.Namespace(
            task="multi",
            curriculum="rule_binding",
            steps=100,
            pair_count=3,
            damage_prob=0.12,
            coordinate_fields=True,
            min_pair_spacing=1,
            sink_assignment="reverse",
            memory_input_steps=4,
            rule_channels=1,
        )

        three_reverse = curriculum_batch_params(args, 59)
        three_cycle = curriculum_batch_params(args, 70)
        final_odd = curriculum_batch_params(args, 95)
        final_even = curriculum_batch_params(args, 96)

        self.assertEqual((three_reverse["pair_count"], three_reverse["sink_assignment"]), (3, "reverse"))
        self.assertEqual((three_cycle["pair_count"], three_cycle["sink_assignment"]), (3, "cycle"))
        self.assertEqual((final_odd["sink_assignment"], final_odd["damage_prob"]), ("reverse", 0.12))
        self.assertEqual((final_even["sink_assignment"], final_even["damage_prob"]), ("cycle", 0.12))

        args.rule_channels = 0
        with self.assertRaises(ValueError):
            curriculum_batch_params(args, 1)

    def test_load_initial_model_restores_weights(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        source = CellularOrganism(layout=layout, cell_hidden=16)
        target = CellularOrganism(layout=layout, cell_hidden=16)

        with torch.no_grad():
            for parameter in source.parameters():
                parameter.add_(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 4, "route_channels": 0},
                    "args": {"cell_hidden": 16},
                },
                path,
            )
            load_initial_model(
                target,
                init_model=str(path),
                device=torch.device("cpu"),
                expected_hidden_channels=4,
                expected_route_channels=0,
                expected_cell_hidden=16,
                expected_update_rule="standard",
                expected_message_slots=8,
                expected_tag_slots=4,
            )

        for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_load_initial_model_rejects_update_rule_mismatch(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        source = CellularOrganism(layout=layout, cell_hidden=16)
        target = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="gated_message",
            message_slots=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 4, "route_channels": 0},
                    "args": {"cell_hidden": 16, "update_rule": "standard"},
                },
                path,
            )

            with self.assertRaises(ValueError):
                load_initial_model(
                    target,
                    init_model=str(path),
                    device=torch.device("cpu"),
                    expected_hidden_channels=4,
                    expected_route_channels=0,
                    expected_cell_hidden=16,
                    expected_update_rule="gated_message",
                    expected_message_slots=3,
                    expected_tag_slots=4,
                )

    def test_load_initial_model_rejects_self_tagging_slot_mismatch(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        source = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="self_tagging",
            tag_slots=2,
        )
        target = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="self_tagging",
            tag_slots=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 4, "route_channels": 0},
                    "args": {"cell_hidden": 16, "update_rule": "self_tagging", "tag_slots": 2},
                },
                path,
            )

            with self.assertRaises(ValueError):
                load_initial_model(
                    target,
                    init_model=str(path),
                    device=torch.device("cpu"),
                    expected_hidden_channels=4,
                    expected_route_channels=0,
                    expected_cell_hidden=16,
                    expected_update_rule="self_tagging",
                    expected_message_slots=8,
                    expected_tag_slots=3,
                )

    def test_load_initial_model_rejects_rule_channel_mismatch(self) -> None:
        source_layout = ChannelLayout(hidden_channels=4)
        target_layout = ChannelLayout(hidden_channels=4, rule_channels=1)
        source = CellularOrganism(layout=source_layout, cell_hidden=16)
        target = CellularOrganism(layout=target_layout, cell_hidden=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 4, "route_channels": 0, "rule_channels": 0},
                    "args": {"cell_hidden": 16, "update_rule": "standard"},
                },
                path,
            )

            with self.assertRaises(ValueError):
                load_initial_model(
                    target,
                    init_model=str(path),
                    device=torch.device("cpu"),
                    expected_hidden_channels=4,
                    expected_route_channels=0,
                    expected_rule_channels=1,
                    expected_cell_hidden=16,
                    expected_update_rule="standard",
                    expected_message_slots=8,
                    expected_tag_slots=4,
                )


if __name__ == "__main__":
    unittest.main()
