from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v19 import build_parser, parse_assignments, state_pair_diagnostics, summarize_v19_result
from organism_v01.channels import ChannelLayout
from organism_v01.tasks import generate_multi_pair_batch


class BenchmarkV19Tests(unittest.TestCase):
    def test_parser_defaults_to_four_pair_assignment_map(self) -> None:
        args = build_parser().parse_args(["--model", "checkpoint.pt"])

        self.assertEqual(args.pair_count, 4)
        self.assertEqual(args.damage_prob, 0.10)
        self.assertEqual(args.injury_prob, 0.10)
        self.assertEqual(parse_assignments(args.assignments), ("aligned", "cycle", "reverse"))
        self.assertIn("v19", args.report)

    def test_state_pair_diagnostics_reports_source_and_sink_rank_failures(self) -> None:
        layout = ChannelLayout(hidden_channels=4, rule_channels=3)
        batch = generate_multi_pair_batch(
            batch_size=1,
            grid_size=14,
            layout=layout,
            pair_count=4,
            sink_assignment="reverse",
            damage_prob=0.0,
            seed=19,
        )
        final_state = torch.zeros_like(batch.initial)
        assert batch.pair_labels is not None
        assert batch.pair_sink_rc is not None

        for pair_index in range(4):
            row, col = [int(value) for value in batch.pair_sink_rc[0, pair_index]]
            label = int(batch.pair_labels[0, pair_index])
            if pair_index == 1:
                label = 1 - label
            final_state[0, layout.output_start + label, row, col] = 5.0

        diagnostics = state_pair_diagnostics(final_state, batch, layout)

        self.assertEqual(diagnostics["target_set_accuracy"], 0.0)
        self.assertEqual(diagnostics["per_sink_accuracy"], 0.75)
        self.assertEqual(diagnostics["correct_count_distribution"]["3"], 1.0)
        self.assertEqual(diagnostics["source_rank_accuracy"]["source_rank_1"], 0.0)
        self.assertEqual(diagnostics["sink_rank_accuracy"]["sink_rank_2"], 0.0)

    def test_summary_highlights_cycle_reverse_gap_and_weakest_rank(self) -> None:
        result = {
            "cycle": _result(target_set=0.75, source_values=[0.9, 0.8, 0.7, 0.6]),
            "reverse": _result(target_set=0.30, source_values=[0.7, 0.2, 0.3, 0.8]),
        }

        summary = summarize_v19_result(result)

        self.assertTrue(summary["diagnostic_only"])
        self.assertEqual(summary["best_assignment"]["assignment"], "cycle")
        self.assertAlmostEqual(summary["cycle_minus_reverse_dynamic_target_set_accuracy"], 0.45)
        self.assertEqual(
            summary["assignments"]["reverse"]["weakest_source_rank"]["rank"],
            "source_rank_1",
        )

    def test_parse_assignments_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            parse_assignments("aligned,missing")


def _detail(target_set: float, source_values: list[float]) -> dict[str, object]:
    source_accuracy = {
        ("source_rank_0_top" if index == 0 else "source_rank_3_bottom" if index == 3 else f"source_rank_{index}"): value
        for index, value in enumerate(source_values)
    }
    sink_accuracy = {
        ("sink_rank_0_top" if index == 0 else "sink_rank_3_bottom" if index == 3 else f"sink_rank_{index}"): value
        for index, value in enumerate(source_values)
    }
    return {
        "target_set_accuracy": target_set,
        "per_sink_accuracy": sum(source_values) / len(source_values),
        "correct_count_distribution": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4},
        "source_rank_accuracy": source_accuracy,
        "sink_rank_accuracy": sink_accuracy,
    }


def _result(target_set: float, source_values: list[float]) -> dict[str, object]:
    return {
        "static": _detail(target_set, source_values),
        "dynamic": {
            "recovery": {
                "0": _detail(max(0.0, target_set - 0.1), source_values),
            },
            "final": _detail(target_set, source_values),
            "injury": {
                "newly_blocked_fraction": 0.05,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
