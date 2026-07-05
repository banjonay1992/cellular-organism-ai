from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v10 import load_model, summarize_v10_result
from organism_v01.channels import ChannelLayout
from organism_v01.organism import CellularOrganism


class BenchmarkV10Tests(unittest.TestCase):
    def test_summary_requires_rule_cued_performance_and_clean_controls(self) -> None:
        reverse = {"target_set_accuracy": 0.66}
        injury = {"target_set_accuracy": 0.56}
        cycle = {"target_set_accuracy": 0.46}
        controls = {
            "normal": {"target_set_accuracy": 0.66},
            "erase_source": {"target_set_accuracy": 0.20},
            "erase_sink": {"target_set_accuracy": 0.12},
            "swap_source": {"target_set_accuracy": 0.0},
            "erase_rule": {"target_set_accuracy": 0.40},
        }

        self.assertTrue(summarize_v10_result(reverse, injury, cycle, controls)["passed"])

        controls["erase_rule"] = {"target_set_accuracy": 0.70}

        self.assertFalse(summarize_v10_result(reverse, injury, cycle, controls)["passed"])

    def test_load_model_rejects_missing_rule_cue(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        model = CellularOrganism(layout=layout, cell_hidden=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "layout": {"hidden_channels": 4, "route_channels": 0, "rule_channels": 0},
                    "args": {"cell_hidden": 16, "update_rule": "standard"},
                },
                path,
            )

            with self.assertRaises(ValueError):
                load_model(str(path), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
