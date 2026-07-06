from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v16 import select_scenarios, summarize_v16_result


def _assignment_result(
    *,
    static_target_set_accuracy: float = 0.70,
    dynamic_target_set_accuracy: float = 0.55,
    dynamic_target_peak_accuracy: float = 0.90,
    routed_slot_accuracy: float = 0.78,
    immediate_target_set_accuracy: float = 0.45,
    newly_blocked_fraction: float = 0.05,
) -> dict[str, dict[str, object]]:
    return {
        "static": {"target_set_accuracy": static_target_set_accuracy},
        "dynamic": {
            "final": {
                "target_set_accuracy": dynamic_target_set_accuracy,
                "target_peak_accuracy": dynamic_target_peak_accuracy,
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
        },
    }


class BenchmarkV16Tests(unittest.TestCase):
    def test_select_scenarios_supports_default_and_named_subset(self) -> None:
        all_scenarios = select_scenarios("all", grid_size=12, rollout_steps=96)
        subset = select_scenarios("baseline,larger_grid", grid_size=12, rollout_steps=96)

        self.assertGreaterEqual(len(all_scenarios), 5)
        self.assertEqual([scenario.name for scenario in subset], ["baseline", "larger_grid"])
        self.assertEqual(subset[1].grid_size, 14)

        with self.assertRaises(ValueError):
            select_scenarios("missing", grid_size=12, rollout_steps=96)

    def test_summary_requires_every_scenario_and_assignment_to_pass(self) -> None:
        scenarios = {
            "baseline": {
                "reverse": _assignment_result(),
                "cycle": _assignment_result(dynamic_target_set_accuracy=0.75),
            },
            "early_injury": {
                "reverse": _assignment_result(dynamic_target_set_accuracy=0.51),
                "cycle": _assignment_result(dynamic_target_set_accuracy=0.52),
            },
        }

        summary = summarize_v16_result(scenarios)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["passed_scenario_count"], 2)
        self.assertEqual(summary["worst_dynamic_target_set_accuracy"], 0.51)

        scenarios["early_injury"]["reverse"] = _assignment_result(dynamic_target_set_accuracy=0.49)
        summary = summarize_v16_result(scenarios)

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["passed_scenario_count"], 1)
        self.assertFalse(
            summary["scenarios"]["early_injury"]["assignments"]["reverse"]["checks"][
                "dynamic_target_set_accuracy"
            ]
        )


if __name__ == "__main__":
    unittest.main()
