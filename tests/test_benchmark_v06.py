from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v06 import summarize_v06_result


class BenchmarkV06Tests(unittest.TestCase):
    def test_summary_requires_reverse_cycle_injury_and_clean_controls(self) -> None:
        reverse = {"target_set_accuracy": 0.66}
        injury = {"target_set_accuracy": 0.56}
        cycle = {"target_set_accuracy": 0.46}
        controls = {
            "normal": {"target_set_accuracy": 0.66},
            "erase_source": {"target_set_accuracy": 0.20},
            "erase_sink": {"target_set_accuracy": 0.12},
            "swap_source": {"target_set_accuracy": 0.0},
        }

        self.assertTrue(summarize_v06_result(reverse, injury, cycle, controls)["passed"])

        weak_cycle = {"target_set_accuracy": 0.30}

        self.assertFalse(summarize_v06_result(reverse, injury, weak_cycle, controls)["passed"])


if __name__ == "__main__":
    unittest.main()
