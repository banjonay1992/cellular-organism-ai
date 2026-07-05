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


class GatedMessageCellUpdate(nn.Module):
    """Shared local update with transient message slots and learned gates."""

    def __init__(self, channels: int, hidden: int = 64, message_slots: int = 8) -> None:
        super().__init__()
        if message_slots <= 0:
            raise ValueError("message_slots must be positive for gated_message update")

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.message_write = nn.Conv2d(hidden, message_slots, kernel_size=1)
        self.message_gate = nn.Conv2d(hidden, message_slots, kernel_size=1)
        self.message_mix = nn.Conv2d(message_slots, message_slots, kernel_size=3, padding=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + message_slots, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        perceived = self.perception(state)
        messages = self.message_write(perceived) * torch.sigmoid(self.message_gate(perceived))
        mixed_messages = self.message_mix(messages)
        readout = self.readout(torch.cat([perceived, mixed_messages], dim=1))
        return self.delta(readout) * torch.sigmoid(self.update_gate(readout))


UPDATE_RULES = ("standard", "gated_message")
