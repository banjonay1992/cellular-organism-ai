from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.ambiguity import audit_assignment_ambiguity


class AmbiguityTests(unittest.TestCase):
    def test_reverse_and_cycle_can_conflict_on_identical_inputs(self) -> None:
        report = audit_assignment_ambiguity(
            assignment_a="reverse",
            assignment_b="cycle",
            seeds=1,
            start_seed=80,
            batch_size=4,
            grid_size=12,
            pair_count=3,
            min_pair_spacing=1,
            damage_prob=0.10,
        )

        self.assertEqual(report["identical_input_items"], report["total_items"])
        self.assertGreater(report["conflicting_target_items"], 0)
        self.assertGreater(report["conflict_rate_given_identical_input"], 0.0)
        self.assertTrue(report["examples"])

    def test_matching_assignments_do_not_conflict(self) -> None:
        report = audit_assignment_ambiguity(
            assignment_a="reverse",
            assignment_b="reverse",
            seeds=2,
            start_seed=80,
            batch_size=4,
            grid_size=12,
            pair_count=3,
            min_pair_spacing=1,
            damage_prob=0.10,
        )

        self.assertEqual(report["identical_input_items"], report["total_items"])
        self.assertEqual(report["conflicting_target_items"], 0)
        self.assertEqual(report["conflict_rate_given_identical_input"], 0.0)


if __name__ == "__main__":
    unittest.main()
