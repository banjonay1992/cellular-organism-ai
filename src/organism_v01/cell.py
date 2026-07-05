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


class SelfTaggingCellUpdate(nn.Module):
    """Shared update with persistent internal tag chemistry in hidden channels."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        hidden: int = 64,
        tag_slots: int = 4,
    ) -> None:
        super().__init__()
        if tag_slots <= 0:
            raise ValueError("tag_slots must be positive for self_tagging update")
        if tag_slots > hidden_channels:
            raise ValueError("tag_slots cannot exceed hidden_channels")

        self.tag_start = hidden_start
        self.tag_slots = tag_slots
        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.tag_diffusion = nn.Conv2d(tag_slots, tag_slots, kernel_size=3, padding=1, groups=tag_slots, bias=False)
        self.tag_write = nn.Conv2d(hidden, tag_slots, kernel_size=1)
        self.tag_gate = nn.Conv2d(hidden, tag_slots, kernel_size=1)
        self.tag_read = nn.Conv2d(tag_slots * 2, tag_slots, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + tag_slots, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)

        with torch.no_grad():
            self.tag_diffusion.weight.zero_()
            self.tag_diffusion.weight[:, 0, 1, 1] = 0.60
            self.tag_diffusion.weight[:, 0, 0, 1] = 0.10
            self.tag_diffusion.weight[:, 0, 1, 0] = 0.10
            self.tag_diffusion.weight[:, 0, 1, 2] = 0.10
            self.tag_diffusion.weight[:, 0, 2, 1] = 0.10
        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        tag_slice = slice(self.tag_start, self.tag_start + self.tag_slots)
        tag_state = state[:, tag_slice]
        perceived = self.perception(state)
        diffused_tags = self.tag_diffusion(tag_state)
        candidate_tags = torch.tanh(self.tag_write(perceived) + diffused_tags)
        tag_gate = torch.sigmoid(self.tag_gate(perceived))
        tag_delta = (candidate_tags - tag_state) * tag_gate * 0.25
        tag_context = self.tag_read(torch.cat([tag_state, candidate_tags], dim=1))
        readout = self.readout(torch.cat([perceived, tag_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))
        delta = delta.clone()
        delta[:, tag_slice] = delta[:, tag_slice] + tag_delta
        return delta


class RankBindingCellUpdate(nn.Module):
    """Shared update with internal directional order waves for source/sink ranks."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("rank_binding requires at least 4 hidden channels")

        self.hidden_start = hidden_start
        self.source_a = source_a
        self.source_b = source_b
        self.sink = sink
        self.source_down = hidden_start
        self.source_up = hidden_start + 1
        self.sink_down = hidden_start + 2
        self.sink_up = hidden_start + 3

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.wave_gate = nn.Conv2d(hidden, 4, kernel_size=1)
        self.rank_read = nn.Conv2d(4, 8, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + 8, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)

        down_kernel = torch.zeros(1, 1, 3, 3)
        down_kernel[0, 0, 0, 1] = 1.0
        up_kernel = torch.zeros(1, 1, 3, 3)
        up_kernel[0, 0, 2, 1] = 1.0
        self.register_buffer("down_kernel", down_kernel)
        self.register_buffer("up_kernel", up_kernel)

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

    def _directional_target(self, marker: torch.Tensor, wave: torch.Tensor, *, downward: bool) -> torch.Tensor:
        kernel = self.down_kernel if downward else self.up_kernel
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.94).clamp(-3.0, 3.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_marker = (state[:, self.source_a : self.source_a + 1] + state[:, self.source_b : self.source_b + 1]).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]

        perceived = self.perception(state)
        rank_features = torch.cat(
            [source_down_state, source_up_state, sink_down_state, sink_up_state],
            dim=1,
        )
        rank_context = self.rank_read(rank_features)
        readout = self.readout(torch.cat([perceived, rank_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        wave_targets = torch.cat(
            [
                self._directional_target(source_marker, source_down_state, downward=True),
                self._directional_target(source_marker, source_up_state, downward=False),
                self._directional_target(sink_marker, sink_down_state, downward=True),
                self._directional_target(sink_marker, sink_up_state, downward=False),
            ],
            dim=1,
        )
        wave_states = rank_features
        wave_delta = (wave_targets - wave_states) * torch.sigmoid(self.wave_gate(perceived)) * 0.35

        delta = delta.clone()
        delta[:, self.source_down : self.sink_up + 1] = delta[:, self.source_down : self.sink_up + 1] + wave_delta
        return delta


class SinkStabilizedRankCellUpdate(nn.Module):
    """Rank waves with bidirectional lateral spread and endpoint anchors."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("sink_stabilized_rank requires at least 8 hidden channels")

        self.source_a = source_a
        self.source_b = source_b
        self.sink = sink
        self.rank_start = hidden_start
        self.source_down = hidden_start
        self.source_up = hidden_start + 1
        self.sink_down = hidden_start + 2
        self.sink_up = hidden_start + 3
        self.source_at_sink_down = hidden_start + 4
        self.source_at_sink_up = hidden_start + 5
        self.sink_at_source_down = hidden_start + 6
        self.sink_at_source_up = hidden_start + 7

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.wave_gate = nn.Conv2d(hidden, 8, kernel_size=1)
        self.rank_read = nn.Conv2d(8, 12, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + 12, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)

        self.register_buffer("source_down_kernel", self._make_kernel(vertical="down", horizontal="right"))
        self.register_buffer("source_up_kernel", self._make_kernel(vertical="up", horizontal="right"))
        self.register_buffer("sink_down_kernel", self._make_kernel(vertical="down", horizontal="left"))
        self.register_buffer("sink_up_kernel", self._make_kernel(vertical="up", horizontal="left"))
        self.register_buffer("diffuse_kernel", self._make_diffuse_kernel())

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)

    @staticmethod
    def _make_kernel(*, vertical: str, horizontal: str) -> torch.Tensor:
        kernel = torch.zeros(1, 1, 3, 3)
        vertical_row = 0 if vertical == "down" else 2
        horizontal_col = 0 if horizontal == "right" else 2
        kernel[0, 0, vertical_row, 1] = 0.32
        kernel[0, 0, 1, horizontal_col] = 0.32
        kernel[0, 0, vertical_row, horizontal_col] = 0.18
        kernel[0, 0, 1, 1] = 0.18
        return kernel

    @staticmethod
    def _make_diffuse_kernel() -> torch.Tensor:
        kernel = torch.zeros(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0.52
        kernel[0, 0, 0, 1] = 0.12
        kernel[0, 0, 1, 0] = 0.12
        kernel[0, 0, 1, 2] = 0.12
        kernel[0, 0, 2, 1] = 0.12
        return kernel

    @staticmethod
    def _propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.96).clamp(-4.0, 4.0)

    def _anchor_target(self, anchor: torch.Tensor, incoming: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        diffused_anchor = torch.nn.functional.conv2d(anchor, self.diffuse_kernel, padding=1) * 0.88
        return torch.where(marker.bool(), incoming, diffused_anchor).clamp(-4.0, 4.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_marker = (state[:, self.source_a : self.source_a + 1] + state[:, self.source_b : self.source_b + 1]).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        rank_slice = slice(self.rank_start, self.rank_start + 8)
        rank_features = state[:, rank_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]

        perceived = self.perception(state)
        rank_context = self.rank_read(rank_features)
        readout = self.readout(torch.cat([perceived, rank_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        wave_targets = torch.cat(
            [
                source_down_target,
                source_up_target,
                sink_down_target,
                sink_up_target,
                self._anchor_target(source_at_sink_down_state, source_down_target, sink_marker),
                self._anchor_target(source_at_sink_up_state, source_up_target, sink_marker),
                self._anchor_target(sink_at_source_down_state, sink_down_target, source_marker),
                self._anchor_target(sink_at_source_up_state, sink_up_target, source_marker),
            ],
            dim=1,
        )
        wave_delta = (wave_targets - rank_features) * torch.sigmoid(self.wave_gate(perceived)) * 0.40

        delta = delta.clone()
        delta[:, rank_slice] = wave_delta
        return delta


UPDATE_RULES = ("standard", "gated_message", "self_tagging", "rank_binding", "sink_stabilized_rank")
