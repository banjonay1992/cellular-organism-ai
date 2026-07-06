from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.benchmark_v15 import build_parser


class BenchmarkV15Tests(unittest.TestCase):
    def test_parser_defaults_to_compounded_damage_condition(self) -> None:
        args = build_parser().parse_args(["--model", "checkpoint.pt"])

        self.assertEqual(args.damage_prob, 0.10)
        self.assertEqual(args.injury_prob, 0.10)
        self.assertEqual(args.seed, 91400)
        self.assertIn("v15", args.report)


if __name__ == "__main__":
    unittest.main()
