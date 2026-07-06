from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v12 import summarize_v12_result


class BenchmarkV12Tests(unittest.TestCase):
    def test_summary_requires_clean_three_pair_organs_answers_and_controls(self) -> None:
        reverse = {
            "target_set_accuracy": 0.61,
            "slot_accuracy": 0.81,
            "routed_slot_accuracy": 0.91,
        }
        cycle = {
            "target_set_accuracy": 0.51,
            "slot_accuracy": 0.82,
            "routed_slot_accuracy": 0.92,
        }
        controls = {
            "normal": {"target_set_accuracy": 0.61},
            "erase_source": {"target_set_accuracy": 0.20},
            "erase_sink": {"target_set_accuracy": 0.12},
            "swap_source": {"target_set_accuracy": 0.0},
        }
        erase_rule = {
            "reverse": {"target_set_accuracy": 0.54},
            "cycle": {"target_set_accuracy": 0.56},
        }

        self.assertTrue(summarize_v12_result(reverse, cycle, controls, erase_rule)["passed"])

        erase_rule["cycle"] = {"target_set_accuracy": 0.70}

        summary = summarize_v12_result(reverse, cycle, controls, erase_rule)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["erase_rule_balanced_target_set_accuracy"])


if __name__ == "__main__":
    unittest.main()
