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
from organism_v01.tasks import generate_task_batch
from organism_v01.train import (
    build_parser,
    curriculum_batch_params,
    dynamic_injury_steps,
    freeze_non_claim_parameters,
    freeze_non_repair_parameters,
    load_initial_model,
    scale_training_params,
    training_rollout,
)


class TrainCurriculumTests(unittest.TestCase):
    def test_parser_exposes_slot_weight_for_rank_slot_training(self) -> None:
        args = build_parser().parse_args(["--slot-weight", "0.4"])

        self.assertEqual(args.slot_weight, 0.4)

    def test_parser_exposes_dynamic_injury_training_args(self) -> None:
        args = build_parser().parse_args(
            ["--dynamic-injury-prob", "0.2", "--dynamic-injury-pre-steps", "7", "--rollout-steps", "20"]
        )

        self.assertEqual(args.dynamic_injury_prob, 0.2)
        self.assertEqual(dynamic_injury_steps(args), (7, 13))

    def test_parser_exposes_consistency_loss_args(self) -> None:
        args = build_parser().parse_args(["--consistency-weight", "0.3", "--consistency-margin", "1.5"])

        self.assertEqual(args.consistency_weight, 0.3)
        self.assertEqual(args.consistency_margin, 1.5)

    def test_parser_exposes_repair_only_training_flag(self) -> None:
        args = build_parser().parse_args(["--train-repair-only"])

        self.assertTrue(args.train_repair_only)

    def test_parser_exposes_claim_training_args(self) -> None:
        args = build_parser().parse_args(["--claim-weight", "0.4", "--train-claim-only"])

        self.assertEqual(args.claim_weight, 0.4)
        self.assertTrue(args.train_claim_only)

    def test_parser_accepts_relative_rank_update_rule(self) -> None:
        args = build_parser().parse_args(["--update-rule", "relative_rank_rule_cued"])

        self.assertEqual(args.update_rule, "relative_rank_rule_cued")

    def test_parser_accepts_rank_slot_repair_update_rule(self) -> None:
        args = build_parser().parse_args(["--update-rule", "rank_slot_repair_rule_cued"])

        self.assertEqual(args.update_rule, "rank_slot_repair_rule_cued")

    def test_parser_accepts_rank_slot_claim_update_rule(self) -> None:
        args = build_parser().parse_args(["--update-rule", "rank_slot_claim_rule_cued"])

        self.assertEqual(args.update_rule, "rank_slot_claim_rule_cued")

    def test_scale_training_params_cycle_in_two_step_blocks(self) -> None:
        args = build_parser().parse_args(
            [
                "--grid-size",
                "12",
                "--grid-size-choices",
                "12,14",
                "--rollout-steps",
                "96",
                "--rollout-steps-choices",
                "96,112",
                "--dynamic-injury-prob",
                "0.1",
            ]
        )

        first = scale_training_params(args, 1)
        second = scale_training_params(args, 2)
        third = scale_training_params(args, 3)
        fourth = scale_training_params(args, 4)

        self.assertEqual((first["grid_size"], first["rollout_steps"], first["pre_steps"]), (12, 96, 48))
        self.assertEqual((second["grid_size"], second["rollout_steps"], second["pre_steps"]), (12, 96, 48))
        self.assertEqual((third["grid_size"], third["rollout_steps"], third["pre_steps"]), (14, 112, 56))
        self.assertEqual((fourth["grid_size"], fourth["rollout_steps"], fourth["pre_steps"]), (14, 112, 56))

    def test_scale_training_params_rejects_mismatched_rollout_choices(self) -> None:
        args = build_parser().parse_args(
            [
                "--grid-size-choices",
                "12,14,16",
                "--rollout-steps-choices",
                "96,112",
            ]
        )

        with self.assertRaises(ValueError):
            scale_training_params(args, 1)

    def test_training_rollout_applies_mid_rollout_injury_when_enabled(self) -> None:
        layout = ChannelLayout(hidden_channels=8)
        model = CellularOrganism(layout=layout, cell_hidden=8)
        batch = generate_task_batch(
            task="multi",
            batch_size=2,
            grid_size=8,
            layout=layout,
            damage_prob=0.0,
            pair_count=2,
            min_pair_spacing=1,
            sink_assignment="reverse",
            seed=44,
        )
        args = argparse.Namespace(
            rollout_steps=2,
            dynamic_injury_prob=0.25,
            dynamic_injury_pre_steps=1,
            seed=55,
        )

        rollout = training_rollout(
            model,
            batch,
            layout,
            args,
            step=3,
            device=next(model.parameters()).device,
        )

        self.assertTrue(rollout.injury_applied)
        self.assertEqual((rollout.pre_steps, rollout.post_steps), (1, 1))
        self.assertGreaterEqual(
            float(rollout.loss_batch.env[:, layout.blocked].sum()),
            float(batch.env[:, layout.blocked].sum()),
        )

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

    def test_rule_binding_damage_curriculum_ramps_damage_after_clean_organs(self) -> None:
        args = argparse.Namespace(
            task="multi",
            curriculum="rule_binding_damage",
            steps=100,
            pair_count=3,
            damage_prob=0.10,
            coordinate_fields=True,
            min_pair_spacing=1,
            sink_assignment="reverse",
            memory_input_steps=4,
            rule_channels=3,
        )

        one_pair = curriculum_batch_params(args, 1)
        three_reverse_clean = curriculum_batch_params(args, 45)
        three_cycle_clean = curriculum_batch_params(args, 60)
        light_damage = curriculum_batch_params(args, 70)
        medium_damage = curriculum_batch_params(args, 80)
        full_damage = curriculum_batch_params(args, 95)

        self.assertEqual((one_pair["pair_count"], one_pair["sink_assignment"], one_pair["damage_prob"]), (1, "aligned", 0.0))
        self.assertEqual((three_reverse_clean["pair_count"], three_reverse_clean["sink_assignment"]), (3, "reverse"))
        self.assertEqual(three_reverse_clean["damage_prob"], 0.0)
        self.assertEqual((three_cycle_clean["pair_count"], three_cycle_clean["sink_assignment"]), (3, "cycle"))
        self.assertEqual(three_cycle_clean["damage_prob"], 0.0)
        self.assertAlmostEqual(float(light_damage["damage_prob"]), 0.025)
        self.assertAlmostEqual(float(medium_damage["damage_prob"]), 0.05)
        self.assertAlmostEqual(float(full_damage["damage_prob"]), 0.10)

        args.rule_channels = 0
        with self.assertRaises(ValueError):
            curriculum_batch_params(args, 1)

    def test_rule_binding_final_curriculum_alternates_three_pair_rules_immediately(self) -> None:
        args = argparse.Namespace(
            task="multi",
            curriculum="rule_binding_final",
            steps=100,
            pair_count=3,
            damage_prob=0.05,
            coordinate_fields=True,
            min_pair_spacing=1,
            sink_assignment="reverse",
            memory_input_steps=4,
            rule_channels=3,
        )

        odd_step = curriculum_batch_params(args, 1)
        even_step = curriculum_batch_params(args, 2)

        self.assertEqual((odd_step["pair_count"], odd_step["sink_assignment"], odd_step["damage_prob"]), (3, "reverse", 0.05))
        self.assertEqual((even_step["pair_count"], even_step["sink_assignment"], even_step["damage_prob"]), (3, "cycle", 0.05))

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

    def test_load_initial_model_allows_rank_slot_repair_warm_start(self) -> None:
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        source = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_rule_cued",
        )
        target = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_repair_rule_cued",
        )

        with torch.no_grad():
            source.cell_update.local_match[-1].bias.add_(2.0)
            target.cell_update.local_match[-1].bias.zero_()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 32, "route_channels": 0, "rule_channels": 3},
                    "args": {"cell_hidden": 16, "update_rule": "rank_slot_rule_cued"},
                },
                path,
            )
            load_initial_model(
                target,
                init_model=str(path),
                device=torch.device("cpu"),
                expected_hidden_channels=32,
                expected_route_channels=0,
                expected_rule_channels=3,
                expected_cell_hidden=16,
                expected_update_rule="rank_slot_repair_rule_cued",
                expected_message_slots=8,
                expected_tag_slots=4,
            )

        self.assertTrue(torch.equal(source.cell_update.local_match[-1].bias, target.cell_update.local_match[-1].bias))
        self.assertTrue(hasattr(target.cell_update, "repair_match"))

    def test_load_initial_model_allows_rank_slot_claim_warm_start(self) -> None:
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        source = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_rule_cued",
        )
        target = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_claim_rule_cued",
        )

        with torch.no_grad():
            source.cell_update.local_match[-1].bias.add_(3.0)
            target.cell_update.local_match[-1].bias.zero_()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "layout": {"hidden_channels": 32, "route_channels": 0, "rule_channels": 3},
                    "args": {"cell_hidden": 16, "update_rule": "rank_slot_rule_cued"},
                },
                path,
            )
            load_initial_model(
                target,
                init_model=str(path),
                device=torch.device("cpu"),
                expected_hidden_channels=32,
                expected_route_channels=0,
                expected_rule_channels=3,
                expected_cell_hidden=16,
                expected_update_rule="rank_slot_claim_rule_cued",
                expected_message_slots=8,
                expected_tag_slots=4,
            )

        self.assertTrue(torch.equal(source.cell_update.local_match[-1].bias, target.cell_update.local_match[-1].bias))
        self.assertTrue(hasattr(target.cell_update, "claim_match"))

    def test_freeze_non_repair_parameters_leaves_only_repair_trainable(self) -> None:
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_repair_rule_cued",
        )

        trainable_count = freeze_non_repair_parameters(model)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]

        self.assertGreater(trainable_count, 0)
        self.assertTrue(trainable_names)
        self.assertTrue(frozen_names)
        self.assertTrue(all("repair" in name for name in trainable_names))

    def test_freeze_non_claim_parameters_leaves_only_claim_trainable(self) -> None:
        layout = ChannelLayout(hidden_channels=32, rule_channels=3)
        model = CellularOrganism(
            layout=layout,
            cell_hidden=16,
            update_rule="rank_slot_claim_rule_cued",
        )

        trainable_count = freeze_non_claim_parameters(model)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]

        self.assertGreater(trainable_count, 0)
        self.assertTrue(trainable_names)
        self.assertTrue(frozen_names)
        self.assertTrue(all("claim" in name for name in trainable_names))

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
