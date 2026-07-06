from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v18 import build_parser, summarize_v18_result


def _assignment_result(
    *,
    static_target_set_accuracy: float = 0.45,
    dynamic_target_set_accuracy: float = 0.38,
    dynamic_target_peak_accuracy: float = 0.70,
    routed_slot_accuracy: float = 0.0,
    immediate_target_set_accuracy: float = 0.30,
    newly_blocked_fraction: float = 0.05,
) -> dict[str, dict[str, object]]:
    return {
        "static": {
            "target_set_accuracy": static_target_set_accuracy,
        },
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


class BenchmarkV18Tests(unittest.TestCase):
    def test_parser_defaults_to_four_pair_probe(self) -> None:
        args = build_parser().parse_args(["--model", "checkpoint.pt"])

        self.assertEqual(args.pair_count, 4)
        self.assertEqual(args.damage_prob, 0.10)
        self.assertEqual(args.injury_prob, 0.10)
        self.assertEqual(args.seed, 111800)
        self.assertIn("v18", args.report)

    def test_summary_ignores_rank_slot_gate_when_capacity_is_exceeded(self) -> None:
        summary = summarize_v18_result(
            {
                "reverse": _assignment_result(),
                "cycle": _assignment_result(dynamic_target_set_accuracy=0.41),
            },
            pair_count=4,
        )

        self.assertTrue(summary["passed"])
        self.assertFalse(summary["rank_slot_metrics_supported"])
        self.assertTrue(summary["fixed_rank_slot_capacity_exceeded"])
        self.assertEqual(summary["mean_dynamic_routed_slot_accuracy"], 0.0)
        self.assertEqual(summary["worst_dynamic_target_set_accuracy"], 0.38)

    def test_summary_fails_weak_answers_or_missing_real_injury(self) -> None:
        weak_dynamic = summarize_v18_result(
            {
                "reverse": _assignment_result(dynamic_target_set_accuracy=0.34),
                "cycle": _assignment_result(),
            },
            pair_count=4,
        )

        self.assertFalse(weak_dynamic["passed"])
        self.assertFalse(
            weak_dynamic["assignments"]["reverse"]["checks"]["dynamic_target_set_accuracy"]
        )

        missing_injury = summarize_v18_result(
            {
                "reverse": _assignment_result(),
                "cycle": _assignment_result(newly_blocked_fraction=0.0),
            },
            pair_count=4,
        )

        self.assertFalse(missing_injury["passed"])
        self.assertFalse(missing_injury["assignments"]["cycle"]["checks"]["newly_blocked_fraction"])


if __name__ == "__main__":
    unittest.main()
