from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v05 import summarize_v05_result


class BenchmarkV05Tests(unittest.TestCase):
    def test_summary_passes_only_with_good_main_scores_and_failing_controls(self) -> None:
        heldout = {"target_set_accuracy": 0.91}
        injury = {"target_set_accuracy": 0.86}
        controls = {
            "normal": {"target_set_accuracy": 0.92},
            "erase_source": {"target_set_accuracy": 0.20},
            "erase_sink": {"target_set_accuracy": 0.12},
            "swap_source": {"target_set_accuracy": 0.0},
        }

        self.assertTrue(summarize_v05_result(heldout, injury, controls)["passed"])

        leaky_controls = {
            **controls,
            "erase_sink": {"target_set_accuracy": 0.60},
        }

        self.assertFalse(summarize_v05_result(heldout, injury, leaky_controls)["passed"])


if __name__ == "__main__":
    unittest.main()
