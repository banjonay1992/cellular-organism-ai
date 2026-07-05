from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.diagnose_binding import gather_pair_vectors, mean_cosine_matrix, paired_cosines


class DiagnoseBindingTests(unittest.TestCase):
    def test_gather_pair_vectors_reads_requested_cells(self) -> None:
        state = torch.arange(1 * 4 * 5 * 5, dtype=torch.float32).view(1, 4, 5, 5)
        pair_rc = torch.tensor([[[1, 2], [3, 4]]])

        vectors = gather_pair_vectors(state, pair_rc, slice(1, 4))

        self.assertEqual(tuple(vectors.shape), (1, 2, 3))
        self.assertTrue(torch.equal(vectors[0, 0], state[0, 1:4, 1, 2]))
        self.assertTrue(torch.equal(vectors[0, 1], state[0, 1:4, 3, 4]))

    def test_cosine_summaries_preserve_pair_structure(self) -> None:
        source = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        sink = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        paired = paired_cosines(source, sink)
        matrix = mean_cosine_matrix(source, sink)

        self.assertEqual(paired, [1.0, 1.0])
        self.assertEqual(matrix, [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
