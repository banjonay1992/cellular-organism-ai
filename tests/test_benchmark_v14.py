from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v14 import summarize_v14_result


def _dynamic_result(
    *,
    target_set_accuracy: float = 0.56,
    target_peak_accuracy: float = 0.82,
    routed_slot_accuracy: float = 0.80,
    immediate_target_set_accuracy: float = 0.50,
    newly_blocked_fraction: float = 0.06,
) -> dict[str, object]:
    return {
        "final": {
            "target_set_accuracy": target_set_accuracy,
            "target_peak_accuracy": target_peak_accuracy,
            "routed_slot_accuracy": routed_slot_accuracy,
        },
        "recovery": {
            "0": {
                "target_set_accuracy": immediate_target_set_accuracy,
            }
        },
        "injury": {
            "newly_blocked_fraction": newly_blocked_fraction,
        },
    }


class BenchmarkV14Tests(unittest.TestCase):
    def test_summary_requires_static_baseline_dynamic_recovery_and_real_injury(self) -> None:
        static_reverse = {"target_set_accuracy": 0.60}
        static_cycle = {"target_set_accuracy": 0.58}
        dynamic_reverse = _dynamic_result()
        dynamic_cycle = _dynamic_result(target_set_accuracy=0.55)

        summary = summarize_v14_result(static_reverse, static_cycle, dynamic_reverse, dynamic_cycle)

        self.assertTrue(summary["passed"])
        self.assertGreater(summary["reverse_survival_ratio"], 0.8)
        self.assertAlmostEqual(summary["reverse_recovery_target_set_delta"], 0.06)

    def test_summary_fails_weak_dynamic_answer_or_missing_new_damage(self) -> None:
        static_reverse = {"target_set_accuracy": 0.60}
        static_cycle = {"target_set_accuracy": 0.58}

        weak_dynamic = _dynamic_result(target_set_accuracy=0.45)
        summary = summarize_v14_result(static_reverse, static_cycle, weak_dynamic, _dynamic_result())
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["reverse_dynamic_target_set_accuracy"])

        no_real_injury = _dynamic_result(newly_blocked_fraction=0.01)
        summary = summarize_v14_result(static_reverse, static_cycle, _dynamic_result(), no_real_injury)
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["cycle_newly_blocked_fraction"])


if __name__ == "__main__":
    unittest.main()
