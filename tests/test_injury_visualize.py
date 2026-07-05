from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from organism_v01.channels import ChannelLayout
from organism_v01.injury import apply_random_injury
from organism_v01.tasks import generate_routing_batch
from organism_v01.visualize import panel_grid


class InjuryAndVisualTests(unittest.TestCase):
    def test_injury_preserves_source_and_target_cells(self) -> None:
        layout = ChannelLayout(hidden_channels=4)
        batch = generate_routing_batch(batch_size=8, grid_size=12, layout=layout, seed=101)

        injured = apply_random_injury(batch, layout, injury_prob=0.5, seed=202)

        for item in range(batch.initial.shape[0]):
            source_row, source_col = [int(value) for value in batch.source_rc[item]]
            sink_row, sink_col = [int(value) for value in batch.sink_rc[item]]
            self.assertEqual(float(injured.alive_mask[item, 0, source_row, source_col]), 1.0)
            self.assertEqual(float(injured.alive_mask[item, 0, sink_row, sink_col]), 1.0)
        self.assertGreater(float(injured.env[:, layout.blocked].sum()), float(batch.env[:, layout.blocked].sum()))

    def test_panel_grid_saves_image(self) -> None:
        panels = [
            ("a", np.zeros((4, 4), dtype=np.float32)),
            ("b", np.eye(4, dtype=np.float32)),
        ]
        image = panel_grid(panels, scale=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "panel.png"
            image.save(path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

