from __future__ import annotations

import torch
from torch import nn


def _largest_group_count(channel_count: int, max_groups: int = 8) -> int:
    groups = min(channel_count, max_groups)
    while groups > 1 and channel_count % groups != 0:
        groups -= 1
    return groups


class CellUpdate(nn.Module):
    """A tiny shared network applied to every cell neighborhood."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        groups = _largest_group_count(hidden)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

