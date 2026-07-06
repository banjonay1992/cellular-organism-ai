from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v13 import summarize_v13_result


class BenchmarkV13Tests(unittest.TestCase):
    def test_summary_requires_damaged_three_pair_answers_organs_and_controls(self) -> None:
        reverse = {
            "target_set_accuracy": 0.56,
            "slot_accuracy": 0.66,
            "routed_slot_accuracy": 0.81,
        }
        cycle = {
            "target_set_accuracy": 0.46,
            "slot_accuracy": 0.67,
            "routed_slot_accuracy": 0.82,
        }
        controls = {
            "normal": {"target_set_accuracy": 0.56},
            "erase_source": {"target_set_accuracy": 0.20},
            "erase_sink": {"target_set_accuracy": 0.80, "target_peak_accuracy": 0.12},
            "swap_source": {"target_set_accuracy": 0.0},
        }
        erase_rule = {
            "reverse": {"target_set_accuracy": 0.54},
            "cycle": {"target_set_accuracy": 0.56},
        }

        self.assertTrue(summarize_v13_result(reverse, cycle, controls, erase_rule)["passed"])

        reverse["routed_slot_accuracy"] = 0.79

        summary = summarize_v13_result(reverse, cycle, controls, erase_rule)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["reverse_routed_slot_accuracy"])

        reverse["routed_slot_accuracy"] = 0.81
        controls["erase_sink"] = {"target_set_accuracy": 0.12, "target_peak_accuracy": 0.40}

        summary = summarize_v13_result(reverse, cycle, controls, erase_rule)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["erase_sink_target_peak_accuracy"])


if __name__ == "__main__":
    unittest.main()
