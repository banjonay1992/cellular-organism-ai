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
            )

        for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
            self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()
