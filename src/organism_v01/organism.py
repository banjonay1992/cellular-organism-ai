from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from organism_v01.cell import CellUpdate
from organism_v01.channels import ChannelLayout
from organism_v01.tasks import RoutingBatch


@dataclass(frozen=True)
class OrganismRollout:
    final_state: torch.Tensor
    activity_loss: torch.Tensor


def clamp_environment(state: torch.Tensor, batch: RoutingBatch, layout: ChannelLayout) -> torch.Tensor:
    clamped = state.clone()
    clamped[:, : layout.env_count] = batch.env
    clamped[:, layout.mutable_slice] = clamped[:, layout.mutable_slice] * batch.alive_mask
    return clamped


class CellularOrganism(nn.Module):
    """A recurrent grid of tiny shared cell updates."""

    def __init__(self, layout: ChannelLayout, cell_hidden: int = 64, update_scale: float = 1.0) -> None:
        super().__init__()
        self.layout = layout
        self.update_scale = update_scale
        self.cell_update = CellUpdate(layout.total_channels, hidden=cell_hidden)

    def forward(self, batch: RoutingBatch, steps: int = 24) -> OrganismRollout:
        if steps <= 0:
            raise ValueError("steps must be positive")

        state = clamp_environment(batch.initial, batch, self.layout)
        activity_terms: list[torch.Tensor] = []

        for _ in range(steps):
            delta = self.cell_update(state) * self.update_scale
            delta = delta.clone()
            delta[:, : self.layout.env_count] = 0.0
            delta[:, self.layout.mutable_slice] = delta[:, self.layout.mutable_slice] * batch.alive_mask
            state = clamp_environment(state + delta, batch, self.layout)
            activity_terms.append(delta[:, self.layout.mutable_slice].abs().mean())

        activity_loss = torch.stack(activity_terms).mean()
        return OrganismRollout(final_state=state, activity_loss=activity_loss)

