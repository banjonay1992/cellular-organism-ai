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


class MatchingReadoutCellUpdate(nn.Module):
    """Sink-stabilized rank waves plus learned local source-label matching."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int | None = None,
        rule_channels: int = 0,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden_channels < 12:
            raise ValueError("matching_readout requires at least 12 hidden channels")
        if rule_channels < 0:
            raise ValueError("rule_channels cannot be negative")
        if rule_channels and rule_start is None:
            raise ValueError("rule_start is required when rule_channels is nonzero")

        self.source_a = source_a
        self.source_b = source_b
        self.sink = sink
        self.output_start = output_start
        self.rule_start = rule_start
        self.rule_channels = rule_channels
        self.match_start = hidden_start
        self.source_down = hidden_start
        self.source_up = hidden_start + 1
        self.sink_down = hidden_start + 2
        self.sink_up = hidden_start + 3
        self.source_at_sink_down = hidden_start + 4
        self.source_at_sink_up = hidden_start + 5
        self.sink_at_source_down = hidden_start + 6
        self.sink_at_source_up = hidden_start + 7
        self.source_a_down = hidden_start + 8
        self.source_a_up = hidden_start + 9
        self.source_b_down = hidden_start + 10
        self.source_b_up = hidden_start + 11

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.wave_gate = nn.Conv2d(hidden, 12, kernel_size=1)
        self.match_read = nn.Conv2d(12, 16, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + 16, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.local_match = nn.Sequential(
            nn.Conv2d(12 + rule_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        self.register_buffer("source_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="right"))
        self.register_buffer("source_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="right"))
        self.register_buffer("sink_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="left"))
        self.register_buffer("sink_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="left"))
        self.register_buffer("diffuse_kernel", SinkStabilizedRankCellUpdate._make_diffuse_kernel())

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        final_match = self.local_match[-1]
        if isinstance(final_match, nn.Conv2d):
            nn.init.normal_(final_match.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final_match.bias)

    @staticmethod
    def _propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.96).clamp(-4.0, 4.0)

    def _anchor_target(self, anchor: torch.Tensor, incoming: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        diffused_anchor = torch.nn.functional.conv2d(anchor, self.diffuse_kernel, padding=1) * 0.88
        return torch.where(marker.bool(), incoming, diffused_anchor).clamp(-4.0, 4.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_a_marker = state[:, self.source_a : self.source_a + 1].clamp(0.0, 1.0)
        source_b_marker = state[:, self.source_b : self.source_b + 1].clamp(0.0, 1.0)
        source_marker = (source_a_marker + source_b_marker).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        match_slice = slice(self.match_start, self.match_start + 12)
        match_features = state[:, match_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]
        source_a_down_state = state[:, self.source_a_down : self.source_a_down + 1]
        source_a_up_state = state[:, self.source_a_up : self.source_a_up + 1]
        source_b_down_state = state[:, self.source_b_down : self.source_b_down + 1]
        source_b_up_state = state[:, self.source_b_up : self.source_b_up + 1]

        perceived = self.perception(state)
        match_context = self.match_read(match_features)
        readout = self.readout(torch.cat([perceived, match_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        source_a_down_target = self._propagate(source_a_down_state, self.source_down_kernel, source_a_marker)
        source_a_up_target = self._propagate(source_a_up_state, self.source_up_kernel, source_a_marker)
        source_b_down_target = self._propagate(source_b_down_state, self.source_down_kernel, source_b_marker)
        source_b_up_target = self._propagate(source_b_up_state, self.source_up_kernel, source_b_marker)
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
                source_a_down_target,
                source_a_up_target,
                source_b_down_target,
                source_b_up_target,
            ],
            dim=1,
        )
        wave_delta = (wave_targets - match_features) * torch.sigmoid(self.wave_gate(perceived)) * 0.40

        local_match_features = wave_targets
        if self.rule_channels:
            assert self.rule_start is not None
            rule_context = state[:, self.rule_start : self.rule_start + self.rule_channels]
            local_match_features = torch.cat([wave_targets, rule_context], dim=1)
        local_output = self.local_match(local_match_features) * sink_marker
        delta = delta.clone()
        delta[:, match_slice] = wave_delta
        delta[:, self.output_start : self.output_start + 2] = delta[:, self.output_start : self.output_start + 2] + local_output
        return delta


class RuleCuedMatchingReadoutCellUpdate(MatchingReadoutCellUpdate):
    """Matching readout whose sink-local decoder receives a global rule cue."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int,
        rule_channels: int,
        hidden: int = 64,
    ) -> None:
        if rule_channels < 1:
            raise ValueError("rule_cued_matching_readout requires at least 1 rule channel")
        super().__init__(
            channels,
            hidden_start=hidden_start,
            hidden_channels=hidden_channels,
            source_a=source_a,
            source_b=source_b,
            sink=sink,
            output_start=output_start,
            rule_start=rule_start,
            rule_channels=rule_channels,
            hidden=hidden,
        )


class RankSlotRuleCuedCellUpdate(nn.Module):
    """Rule-cued readout with explicit internal source-rank label slots."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int,
        rule_channels: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden_channels < 20:
            raise ValueError("rank_slot_rule_cued requires at least 20 hidden channels")
        if rule_channels < 1:
            raise ValueError("rank_slot_rule_cued requires at least 1 rule channel")

        self.source_a = source_a
        self.source_b = source_b
        self.sink = sink
        self.output_start = output_start
        self.rule_start = rule_start
        self.rule_channels = rule_channels
        self.match_start = hidden_start
        self.source_down = hidden_start
        self.source_up = hidden_start + 1
        self.sink_down = hidden_start + 2
        self.sink_up = hidden_start + 3
        self.source_at_sink_down = hidden_start + 4
        self.source_at_sink_up = hidden_start + 5
        self.sink_at_source_down = hidden_start + 6
        self.sink_at_source_up = hidden_start + 7
        self.source_a_down = hidden_start + 8
        self.source_a_up = hidden_start + 9
        self.source_b_down = hidden_start + 10
        self.source_b_up = hidden_start + 11
        self.top_a = hidden_start + 12
        self.top_b = hidden_start + 13
        self.middle_a = hidden_start + 14
        self.middle_b = hidden_start + 15
        self.bottom_a = hidden_start + 16
        self.bottom_b = hidden_start + 17
        self.rank_down = hidden_start + 18
        self.rank_up = hidden_start + 19

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.organ_channels = 20
        self.match_read = nn.Conv2d(self.organ_channels, 16, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + 16, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.local_match = nn.Sequential(
            nn.Conv2d(self.organ_channels + rule_channels, 48, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(48, 2, kernel_size=1),
        )

        self.register_buffer("source_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="right"))
        self.register_buffer("source_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="right"))
        self.register_buffer("sink_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="left"))
        self.register_buffer("sink_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="left"))
        self.register_buffer("diffuse_kernel", SinkStabilizedRankCellUpdate._make_diffuse_kernel())
        vertical_down_kernel = torch.zeros(1, 1, 3, 3)
        vertical_down_kernel[0, 0, 0, 1] = 1.0
        vertical_up_kernel = torch.zeros(1, 1, 3, 3)
        vertical_up_kernel[0, 0, 2, 1] = 1.0
        slot_kernel = torch.zeros(1, 1, 3, 3)
        slot_kernel[0, 0, 1, 0] = 0.76
        slot_kernel[0, 0, 0, 0] = 0.12
        slot_kernel[0, 0, 2, 0] = 0.12
        slot_kernel[0, 0, 1, 1] = 0.08
        self.register_buffer("vertical_down_kernel", vertical_down_kernel)
        self.register_buffer("vertical_up_kernel", vertical_up_kernel)
        self.register_buffer("slot_kernel", slot_kernel)

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        final_match = self.local_match[-1]
        if isinstance(final_match, nn.Conv2d):
            nn.init.normal_(final_match.weight, mean=0.0, std=5e-3)
            nn.init.zeros_(final_match.bias)

    @staticmethod
    def _propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.96).clamp(-4.0, 4.0)

    def _anchor_target(self, anchor: torch.Tensor, incoming: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        diffused_anchor = torch.nn.functional.conv2d(anchor, self.diffuse_kernel, padding=1) * 0.88
        return torch.where(marker.bool(), incoming, diffused_anchor).clamp(-4.0, 4.0)

    def _slot_propagate(self, wave: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        carried = torch.nn.functional.conv2d(wave, self.slot_kernel, padding=1)
        return (seed + carried * 0.98).clamp(0.0, 4.0)

    @staticmethod
    def _vertical_propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.98).clamp(0.0, 4.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_a_marker = state[:, self.source_a : self.source_a + 1].clamp(0.0, 1.0)
        source_b_marker = state[:, self.source_b : self.source_b + 1].clamp(0.0, 1.0)
        source_marker = (source_a_marker + source_b_marker).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        match_slice = slice(self.match_start, self.match_start + self.organ_channels)
        match_features = state[:, match_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]
        source_a_down_state = state[:, self.source_a_down : self.source_a_down + 1]
        source_a_up_state = state[:, self.source_a_up : self.source_a_up + 1]
        source_b_down_state = state[:, self.source_b_down : self.source_b_down + 1]
        source_b_up_state = state[:, self.source_b_up : self.source_b_up + 1]
        rank_down_state = state[:, self.rank_down : self.rank_down + 1]
        rank_up_state = state[:, self.rank_up : self.rank_up + 1]

        perceived = self.perception(state)
        match_context = self.match_read(match_features)
        readout = self.readout(torch.cat([perceived, match_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        source_a_down_target = self._propagate(source_a_down_state, self.source_down_kernel, source_a_marker)
        source_a_up_target = self._propagate(source_a_up_state, self.source_up_kernel, source_a_marker)
        source_b_down_target = self._propagate(source_b_down_state, self.source_down_kernel, source_b_marker)
        source_b_up_target = self._propagate(source_b_up_state, self.source_up_kernel, source_b_marker)
        rank_down_target = self._vertical_propagate(rank_down_state, self.vertical_down_kernel, source_marker)
        rank_up_target = self._vertical_propagate(rank_up_state, self.vertical_up_kernel, source_marker)

        above = torch.nn.functional.conv2d(rank_down_state, self.vertical_down_kernel, padding=1)
        below = torch.nn.functional.conv2d(rank_up_state, self.vertical_up_kernel, padding=1)
        above = above.clamp_min(0.0)
        below = below.clamp_min(0.0)
        has_above = torch.sigmoid((above - 0.12) * 16.0)
        has_below = torch.sigmoid((below - 0.12) * 16.0)
        no_above = 1.0 - has_above
        no_below = 1.0 - has_below
        isolated_seed = no_above * no_below * 0.12
        top_seed = source_marker * no_above * (has_below + isolated_seed)
        bottom_seed = source_marker * no_below * (has_above + isolated_seed)
        middle_seed = source_marker * has_above * has_below

        slot_targets = [
            self._slot_propagate(state[:, self.top_a : self.top_a + 1], top_seed * source_a_marker),
            self._slot_propagate(state[:, self.top_b : self.top_b + 1], top_seed * source_b_marker),
            self._slot_propagate(state[:, self.middle_a : self.middle_a + 1], middle_seed * source_a_marker),
            self._slot_propagate(state[:, self.middle_b : self.middle_b + 1], middle_seed * source_b_marker),
            self._slot_propagate(state[:, self.bottom_a : self.bottom_a + 1], bottom_seed * source_a_marker),
            self._slot_propagate(state[:, self.bottom_b : self.bottom_b + 1], bottom_seed * source_b_marker),
        ]
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
                source_a_down_target,
                source_a_up_target,
                source_b_down_target,
                source_b_up_target,
                *slot_targets,
                rank_down_target,
                rank_up_target,
            ],
            dim=1,
        )
        wave_delta = (wave_targets - match_features) * 0.40

        rule_context = state[:, self.rule_start : self.rule_start + self.rule_channels]
        local_output = self.local_match(torch.cat([wave_targets, rule_context], dim=1)) * sink_marker
        output_delta = local_output
        if self.rule_channels > 1:
            rule_presence = rule_context.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            output_delta = (delta[:, self.output_start : self.output_start + 2] + local_output) * rule_presence
        delta = delta.clone()
        delta[:, match_slice] = wave_delta
        delta[:, self.output_start : self.output_start + 2] = output_delta
        return delta


class RankSlotRepairRuleCuedCellUpdate(RankSlotRuleCuedCellUpdate):
    """Rank-slot organ with a recurrent sink/source repair bus.

    The base rank-slot organ makes each sink decision locally from the rank
    waves. This variant reserves four hidden channels as a tiny repair loop:
    sinks broadcast their current label vote leftward, and sources answer
    rightward with label-carrying repair signals. The learned sink readout can
    then use the shared repair state to resolve item-level inconsistencies.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int,
        rule_channels: int,
        hidden: int = 64,
    ) -> None:
        if hidden_channels < 24:
            raise ValueError("rank_slot_repair_rule_cued requires at least 24 hidden channels")
        super().__init__(
            channels,
            hidden_start=hidden_start,
            hidden_channels=hidden_channels,
            source_a=source_a,
            source_b=source_b,
            sink=sink,
            output_start=output_start,
            rule_start=rule_start,
            rule_channels=rule_channels,
            hidden=hidden,
        )

        self.repair_start = hidden_start + 20
        self.sink_vote_a = hidden_start + 20
        self.sink_vote_b = hidden_start + 21
        self.source_repair_a = hidden_start + 22
        self.source_repair_b = hidden_start + 23
        self.repair_channels = 4
        self.repair_match = nn.Sequential(
            nn.Conv2d(self.organ_channels + self.repair_channels + 2 + rule_channels, 56, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(56, 2, kernel_size=1),
        )

        sink_vote_kernel = torch.zeros(1, 1, 3, 3)
        sink_vote_kernel[0, 0, 1, 2] = 0.56
        sink_vote_kernel[0, 0, 0, 2] = 0.13
        sink_vote_kernel[0, 0, 2, 2] = 0.13
        sink_vote_kernel[0, 0, 0, 1] = 0.06
        sink_vote_kernel[0, 0, 2, 1] = 0.06
        sink_vote_kernel[0, 0, 1, 1] = 0.06
        source_reply_kernel = torch.zeros(1, 1, 3, 3)
        source_reply_kernel[0, 0, 1, 0] = 0.56
        source_reply_kernel[0, 0, 0, 0] = 0.13
        source_reply_kernel[0, 0, 2, 0] = 0.13
        source_reply_kernel[0, 0, 0, 1] = 0.06
        source_reply_kernel[0, 0, 2, 1] = 0.06
        source_reply_kernel[0, 0, 1, 1] = 0.06
        consensus_kernel = torch.zeros(1, 1, 3, 3)
        consensus_kernel[0, 0, 1, 1] = 0.50
        consensus_kernel[0, 0, 0, 1] = 0.20
        consensus_kernel[0, 0, 2, 1] = 0.20
        consensus_kernel[0, 0, 1, 0] = 0.05
        consensus_kernel[0, 0, 1, 2] = 0.05
        self.register_buffer("sink_vote_kernel", sink_vote_kernel)
        self.register_buffer("source_reply_kernel", source_reply_kernel)
        self.register_buffer("repair_consensus_kernel", consensus_kernel)

        final_repair = self.repair_match[-1]
        if isinstance(final_repair, nn.Conv2d):
            nn.init.zeros_(final_repair.weight)
            nn.init.zeros_(final_repair.bias)

    @staticmethod
    def _repair_propagate(
        wave: torch.Tensor,
        seed: torch.Tensor,
        kernel: torch.Tensor,
        consensus_kernel: torch.Tensor,
    ) -> torch.Tensor:
        carried = torch.nn.functional.conv2d(wave, kernel, padding=1)
        consensus = torch.nn.functional.conv2d(wave, consensus_kernel, padding=1)
        return (seed + carried * 0.78 + consensus * 0.20).clamp(-4.0, 4.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_a_marker = state[:, self.source_a : self.source_a + 1].clamp(0.0, 1.0)
        source_b_marker = state[:, self.source_b : self.source_b + 1].clamp(0.0, 1.0)
        source_marker = (source_a_marker + source_b_marker).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        match_slice = slice(self.match_start, self.match_start + self.organ_channels)
        match_features = state[:, match_slice]
        repair_slice = slice(self.repair_start, self.repair_start + self.repair_channels)
        repair_features = state[:, repair_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]
        source_a_down_state = state[:, self.source_a_down : self.source_a_down + 1]
        source_a_up_state = state[:, self.source_a_up : self.source_a_up + 1]
        source_b_down_state = state[:, self.source_b_down : self.source_b_down + 1]
        source_b_up_state = state[:, self.source_b_up : self.source_b_up + 1]
        rank_down_state = state[:, self.rank_down : self.rank_down + 1]
        rank_up_state = state[:, self.rank_up : self.rank_up + 1]

        perceived = self.perception(state)
        match_context = self.match_read(match_features)
        readout = self.readout(torch.cat([perceived, match_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        source_a_down_target = self._propagate(source_a_down_state, self.source_down_kernel, source_a_marker)
        source_a_up_target = self._propagate(source_a_up_state, self.source_up_kernel, source_a_marker)
        source_b_down_target = self._propagate(source_b_down_state, self.source_down_kernel, source_b_marker)
        source_b_up_target = self._propagate(source_b_up_state, self.source_up_kernel, source_b_marker)
        rank_down_target = self._vertical_propagate(rank_down_state, self.vertical_down_kernel, source_marker)
        rank_up_target = self._vertical_propagate(rank_up_state, self.vertical_up_kernel, source_marker)

        above = torch.nn.functional.conv2d(rank_down_state, self.vertical_down_kernel, padding=1).clamp_min(0.0)
        below = torch.nn.functional.conv2d(rank_up_state, self.vertical_up_kernel, padding=1).clamp_min(0.0)
        has_above = torch.sigmoid((above - 0.12) * 16.0)
        has_below = torch.sigmoid((below - 0.12) * 16.0)
        no_above = 1.0 - has_above
        no_below = 1.0 - has_below
        isolated_seed = no_above * no_below * 0.12
        top_seed = source_marker * no_above * (has_below + isolated_seed)
        bottom_seed = source_marker * no_below * (has_above + isolated_seed)
        middle_seed = source_marker * has_above * has_below

        slot_targets = [
            self._slot_propagate(state[:, self.top_a : self.top_a + 1], top_seed * source_a_marker),
            self._slot_propagate(state[:, self.top_b : self.top_b + 1], top_seed * source_b_marker),
            self._slot_propagate(state[:, self.middle_a : self.middle_a + 1], middle_seed * source_a_marker),
            self._slot_propagate(state[:, self.middle_b : self.middle_b + 1], middle_seed * source_b_marker),
            self._slot_propagate(state[:, self.bottom_a : self.bottom_a + 1], bottom_seed * source_a_marker),
            self._slot_propagate(state[:, self.bottom_b : self.bottom_b + 1], bottom_seed * source_b_marker),
        ]
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
                source_a_down_target,
                source_a_up_target,
                source_b_down_target,
                source_b_up_target,
                *slot_targets,
                rank_down_target,
                rank_up_target,
            ],
            dim=1,
        )
        wave_delta = (wave_targets - match_features) * 0.40

        rule_context = state[:, self.rule_start : self.rule_start + self.rule_channels]
        base_local_output = self.local_match(torch.cat([wave_targets, rule_context], dim=1)) * sink_marker
        current_output = torch.tanh(state[:, self.output_start : self.output_start + 2] + base_local_output)
        sink_vote_a_state = state[:, self.sink_vote_a : self.sink_vote_a + 1]
        sink_vote_b_state = state[:, self.sink_vote_b : self.sink_vote_b + 1]
        source_repair_a_state = state[:, self.source_repair_a : self.source_repair_a + 1]
        source_repair_b_state = state[:, self.source_repair_b : self.source_repair_b + 1]
        sink_vote_a_target = self._repair_propagate(
            sink_vote_a_state,
            current_output[:, 0:1] * sink_marker,
            self.sink_vote_kernel,
            self.repair_consensus_kernel,
        )
        sink_vote_b_target = self._repair_propagate(
            sink_vote_b_state,
            current_output[:, 1:2] * sink_marker,
            self.sink_vote_kernel,
            self.repair_consensus_kernel,
        )
        incoming_vote = (sink_vote_a_state.abs() + sink_vote_b_state.abs()).clamp(0.0, 1.0)
        source_repair_a_target = self._repair_propagate(
            source_repair_a_state,
            source_a_marker * incoming_vote,
            self.source_reply_kernel,
            self.repair_consensus_kernel,
        )
        source_repair_b_target = self._repair_propagate(
            source_repair_b_state,
            source_b_marker * incoming_vote,
            self.source_reply_kernel,
            self.repair_consensus_kernel,
        )
        repair_targets = torch.cat(
            [
                sink_vote_a_target,
                sink_vote_b_target,
                source_repair_a_target,
                source_repair_b_target,
            ],
            dim=1,
        )
        repair_delta = (repair_targets - repair_features) * 0.38
        repair_output = self.repair_match(
            torch.cat([wave_targets, repair_features, current_output * sink_marker, rule_context], dim=1)
        ) * sink_marker

        output_delta = base_local_output + repair_output
        if self.rule_channels > 1:
            rule_presence = rule_context.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            output_delta = (delta[:, self.output_start : self.output_start + 2] + output_delta) * rule_presence

        delta = delta.clone()
        delta[:, match_slice] = wave_delta
        delta[:, repair_slice] = repair_delta
        delta[:, self.output_start : self.output_start + 2] = output_delta
        return delta


class RankSlotClaimRuleCuedCellUpdate(RankSlotRuleCuedCellUpdate):
    """Rank-slot organ with explicit sink-to-source rank claims."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int,
        rule_channels: int,
        hidden: int = 64,
    ) -> None:
        if hidden_channels < 32:
            raise ValueError("rank_slot_claim_rule_cued requires at least 32 hidden channels")
        super().__init__(
            channels,
            hidden_start=hidden_start,
            hidden_channels=hidden_channels,
            source_a=source_a,
            source_b=source_b,
            sink=sink,
            output_start=output_start,
            rule_start=rule_start,
            rule_channels=rule_channels,
            hidden=hidden,
        )

        self.source_rank_label_start = hidden_start + 20
        self.source_rank_label_channels = 8
        self.claim_start = hidden_start + 28
        self.claim_channels = 4
        self.claim_seed = nn.Sequential(
            nn.Conv2d(self.claim_channels + rule_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(16, self.claim_channels, kernel_size=1),
        )
        self.claim_match = nn.Sequential(
            nn.Conv2d(
                self.organ_channels + self.source_rank_label_channels + self.claim_channels + 2 + rule_channels,
                64,
                kernel_size=1,
            ),
            nn.SiLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        rank_centers = torch.tensor([-0.72, -0.24, 0.24, 0.72]).view(1, self.claim_channels, 1, 1)
        rank_label_kernel = torch.zeros(1, 1, 3, 3)
        rank_label_kernel[0, 0, 1, 0] = 0.74
        rank_label_kernel[0, 0, 0, 0] = 0.10
        rank_label_kernel[0, 0, 2, 0] = 0.10
        rank_label_kernel[0, 0, 1, 1] = 0.06
        claim_kernel = torch.zeros(1, 1, 3, 3)
        claim_kernel[0, 0, 1, 1] = 0.58
        claim_kernel[0, 0, 0, 1] = 0.16
        claim_kernel[0, 0, 2, 1] = 0.16
        claim_kernel[0, 0, 1, 0] = 0.05
        claim_kernel[0, 0, 1, 2] = 0.05
        self.register_buffer("rank_centers", rank_centers)
        self.register_buffer("rank_label_kernel", rank_label_kernel)
        self.register_buffer("claim_kernel", claim_kernel)

        final_claim = self.claim_match[-1]
        if isinstance(final_claim, nn.Conv2d):
            nn.init.zeros_(final_claim.weight)
            nn.init.zeros_(final_claim.bias)
        final_seed = self.claim_seed[-1]
        if isinstance(final_seed, nn.Conv2d):
            nn.init.normal_(final_seed.weight, mean=0.0, std=5e-3)
            nn.init.zeros_(final_seed.bias)

    def _rank_bins(self, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        coordinate = ((down - up) / (down + up).clamp_min(1.0)).clamp(-1.0, 1.0)
        scores = -18.0 * (coordinate - self.rank_centers).square()
        return torch.softmax(scores, dim=1)

    def _rank_label_target(self, label_state: torch.Tensor, label_seed: torch.Tensor) -> torch.Tensor:
        kernel = self.rank_label_kernel.expand(self.source_rank_label_channels, 1, 3, 3)
        carried = torch.nn.functional.conv2d(
            label_state,
            kernel,
            padding=1,
            groups=self.source_rank_label_channels,
        )
        return (label_seed + carried * 0.98).clamp(0.0, 4.0)

    def _claim_target(self, claim_state: torch.Tensor, claim_seed: torch.Tensor) -> torch.Tensor:
        kernel = self.claim_kernel.expand(self.claim_channels, 1, 3, 3)
        carried = torch.nn.functional.conv2d(
            claim_state,
            kernel,
            padding=1,
            groups=self.claim_channels,
        )
        return (claim_seed + carried * 0.96).clamp(-4.0, 4.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_a_marker = state[:, self.source_a : self.source_a + 1].clamp(0.0, 1.0)
        source_b_marker = state[:, self.source_b : self.source_b + 1].clamp(0.0, 1.0)
        source_marker = (source_a_marker + source_b_marker).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        match_slice = slice(self.match_start, self.match_start + self.organ_channels)
        match_features = state[:, match_slice]
        label_slice = slice(self.source_rank_label_start, self.source_rank_label_start + self.source_rank_label_channels)
        source_rank_label_state = state[:, label_slice]
        claim_slice = slice(self.claim_start, self.claim_start + self.claim_channels)
        claim_state = state[:, claim_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]
        source_a_down_state = state[:, self.source_a_down : self.source_a_down + 1]
        source_a_up_state = state[:, self.source_a_up : self.source_a_up + 1]
        source_b_down_state = state[:, self.source_b_down : self.source_b_down + 1]
        source_b_up_state = state[:, self.source_b_up : self.source_b_up + 1]
        rank_down_state = state[:, self.rank_down : self.rank_down + 1]
        rank_up_state = state[:, self.rank_up : self.rank_up + 1]

        perceived = self.perception(state)
        match_context = self.match_read(match_features)
        readout = self.readout(torch.cat([perceived, match_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        source_a_down_target = self._propagate(source_a_down_state, self.source_down_kernel, source_a_marker)
        source_a_up_target = self._propagate(source_a_up_state, self.source_up_kernel, source_a_marker)
        source_b_down_target = self._propagate(source_b_down_state, self.source_down_kernel, source_b_marker)
        source_b_up_target = self._propagate(source_b_up_state, self.source_up_kernel, source_b_marker)
        rank_down_target = self._vertical_propagate(rank_down_state, self.vertical_down_kernel, source_marker)
        rank_up_target = self._vertical_propagate(rank_up_state, self.vertical_up_kernel, source_marker)

        above = torch.nn.functional.conv2d(rank_down_state, self.vertical_down_kernel, padding=1).clamp_min(0.0)
        below = torch.nn.functional.conv2d(rank_up_state, self.vertical_up_kernel, padding=1).clamp_min(0.0)
        has_above = torch.sigmoid((above - 0.12) * 16.0)
        has_below = torch.sigmoid((below - 0.12) * 16.0)
        no_above = 1.0 - has_above
        no_below = 1.0 - has_below
        isolated_seed = no_above * no_below * 0.12
        top_seed = source_marker * no_above * (has_below + isolated_seed)
        bottom_seed = source_marker * no_below * (has_above + isolated_seed)
        middle_seed = source_marker * has_above * has_below

        slot_targets = [
            self._slot_propagate(state[:, self.top_a : self.top_a + 1], top_seed * source_a_marker),
            self._slot_propagate(state[:, self.top_b : self.top_b + 1], top_seed * source_b_marker),
            self._slot_propagate(state[:, self.middle_a : self.middle_a + 1], middle_seed * source_a_marker),
            self._slot_propagate(state[:, self.middle_b : self.middle_b + 1], middle_seed * source_b_marker),
            self._slot_propagate(state[:, self.bottom_a : self.bottom_a + 1], bottom_seed * source_a_marker),
            self._slot_propagate(state[:, self.bottom_b : self.bottom_b + 1], bottom_seed * source_b_marker),
        ]
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
                source_a_down_target,
                source_a_up_target,
                source_b_down_target,
                source_b_up_target,
                *slot_targets,
                rank_down_target,
                rank_up_target,
            ],
            dim=1,
        )
        wave_delta = (wave_targets - match_features) * 0.40

        source_rank_bins = self._rank_bins(rank_down_state, rank_up_state)
        source_rank_label_seed = torch.cat(
            [
                source_rank_bins * source_a_marker,
                source_rank_bins * source_b_marker,
            ],
            dim=1,
        )
        source_rank_label_target = self._rank_label_target(source_rank_label_state, source_rank_label_seed)
        source_rank_label_delta = (source_rank_label_target - source_rank_label_state) * 0.40

        rule_context = state[:, self.rule_start : self.rule_start + self.rule_channels]
        sink_rank_bins = self._rank_bins(sink_down_state, sink_up_state)
        claim_seed = self.claim_seed(torch.cat([sink_rank_bins, rule_context], dim=1)) * sink_marker
        claim_target = self._claim_target(claim_state, claim_seed)
        claim_delta = (claim_target - claim_state) * 0.36

        base_local_output = self.local_match(torch.cat([wave_targets, rule_context], dim=1)) * sink_marker
        claim_weights = torch.softmax(claim_state, dim=1)
        claimed_label = torch.cat(
            [
                (claim_weights * source_rank_label_state[:, : self.claim_channels]).sum(dim=1, keepdim=True),
                (claim_weights * source_rank_label_state[:, self.claim_channels :]).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        claim_output = self.claim_match(
            torch.cat(
                [
                    wave_targets,
                    source_rank_label_state,
                    claim_state,
                    claimed_label * sink_marker,
                    rule_context,
                ],
                dim=1,
            )
        ) * sink_marker

        output_delta = base_local_output + claim_output
        if self.rule_channels > 1:
            rule_presence = rule_context.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            output_delta = output_delta * rule_presence

        delta = delta.clone()
        delta[:, match_slice] = wave_delta
        delta[:, label_slice] = source_rank_label_delta
        delta[:, claim_slice] = claim_delta
        delta[:, self.output_start : self.output_start + 2] = output_delta
        return delta


class RelativeRankRuleCuedCellUpdate(nn.Module):
    """Rule-cued readout with scalable relative-rank label moments."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_start: int,
        hidden_channels: int,
        source_a: int,
        source_b: int,
        sink: int,
        output_start: int,
        rule_start: int,
        rule_channels: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if hidden_channels < 24:
            raise ValueError("relative_rank_rule_cued requires at least 24 hidden channels")
        if rule_channels < 1:
            raise ValueError("relative_rank_rule_cued requires at least 1 rule channel")

        self.source_a = source_a
        self.source_b = source_b
        self.sink = sink
        self.output_start = output_start
        self.rule_start = rule_start
        self.rule_channels = rule_channels
        self.organ_start = hidden_start
        self.source_down = hidden_start
        self.source_up = hidden_start + 1
        self.sink_down = hidden_start + 2
        self.sink_up = hidden_start + 3
        self.source_at_sink_down = hidden_start + 4
        self.source_at_sink_up = hidden_start + 5
        self.sink_at_source_down = hidden_start + 6
        self.sink_at_source_up = hidden_start + 7
        self.source_a_down = hidden_start + 8
        self.source_a_up = hidden_start + 9
        self.source_b_down = hidden_start + 10
        self.source_b_up = hidden_start + 11
        self.source_count_down = hidden_start + 12
        self.source_count_up = hidden_start + 13
        self.sink_count_down = hidden_start + 14
        self.sink_count_up = hidden_start + 15
        self.moment_a_start = hidden_start + 16
        self.moment_b_start = hidden_start + 20
        self.organ_channels = 24
        self.moment_channels = 8

        groups = _largest_group_count(hidden)
        self.perception = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.organ_read = nn.Conv2d(self.organ_channels, 24, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden + 24, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(hidden, channels, kernel_size=1)
        self.update_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.local_match = nn.Sequential(
            nn.Conv2d(self.organ_channels + rule_channels, 56, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(56, 2, kernel_size=1),
        )

        self.register_buffer("source_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="right"))
        self.register_buffer("source_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="right"))
        self.register_buffer("sink_down_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="down", horizontal="left"))
        self.register_buffer("sink_up_kernel", SinkStabilizedRankCellUpdate._make_kernel(vertical="up", horizontal="left"))
        self.register_buffer("diffuse_kernel", SinkStabilizedRankCellUpdate._make_diffuse_kernel())
        vertical_down_kernel = torch.zeros(1, 1, 3, 3)
        vertical_down_kernel[0, 0, 0, 1] = 1.0
        vertical_up_kernel = torch.zeros(1, 1, 3, 3)
        vertical_up_kernel[0, 0, 2, 1] = 1.0
        moment_kernel = torch.zeros(1, 1, 3, 3)
        moment_kernel[0, 0, 1, 0] = 0.74
        moment_kernel[0, 0, 0, 0] = 0.10
        moment_kernel[0, 0, 2, 0] = 0.10
        moment_kernel[0, 0, 1, 1] = 0.06
        self.register_buffer("vertical_down_kernel", vertical_down_kernel)
        self.register_buffer("vertical_up_kernel", vertical_up_kernel)
        self.register_buffer("moment_kernel", moment_kernel)

        nn.init.normal_(self.delta.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        final_match = self.local_match[-1]
        if isinstance(final_match, nn.Conv2d):
            nn.init.normal_(final_match.weight, mean=0.0, std=5e-3)
            nn.init.zeros_(final_match.bias)

    @staticmethod
    def _propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.96).clamp(-4.0, 4.0)

    @staticmethod
    def _count_propagate(wave: torch.Tensor, kernel: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        propagated = torch.nn.functional.conv2d(wave, kernel, padding=1)
        return (marker + propagated * 0.98).clamp(0.0, 8.0)

    def _anchor_target(self, anchor: torch.Tensor, incoming: torch.Tensor, marker: torch.Tensor) -> torch.Tensor:
        diffused_anchor = torch.nn.functional.conv2d(anchor, self.diffuse_kernel, padding=1) * 0.88
        return torch.where(marker.bool(), incoming, diffused_anchor).clamp(-4.0, 4.0)

    def _rank_basis(self, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        coordinate = ((down - up) / (down + up).clamp_min(1.0)).clamp(-1.0, 1.0)
        return torch.cat(
            [
                torch.ones_like(coordinate),
                coordinate,
                coordinate.square(),
                coordinate * coordinate.square(),
            ],
            dim=1,
        )

    def _moment_target(self, moment_state: torch.Tensor, moment_seed: torch.Tensor) -> torch.Tensor:
        kernel = self.moment_kernel.expand(self.moment_channels, 1, 3, 3)
        carried = torch.nn.functional.conv2d(moment_state, kernel, padding=1, groups=self.moment_channels)
        return (moment_seed + carried * 0.98).clamp(-8.0, 8.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        source_a_marker = state[:, self.source_a : self.source_a + 1].clamp(0.0, 1.0)
        source_b_marker = state[:, self.source_b : self.source_b + 1].clamp(0.0, 1.0)
        source_marker = (source_a_marker + source_b_marker).clamp(0.0, 1.0)
        sink_marker = state[:, self.sink : self.sink + 1].clamp(0.0, 1.0)
        organ_slice = slice(self.organ_start, self.organ_start + self.organ_channels)
        organ_features = state[:, organ_slice]

        source_down_state = state[:, self.source_down : self.source_down + 1]
        source_up_state = state[:, self.source_up : self.source_up + 1]
        sink_down_state = state[:, self.sink_down : self.sink_down + 1]
        sink_up_state = state[:, self.sink_up : self.sink_up + 1]
        source_at_sink_down_state = state[:, self.source_at_sink_down : self.source_at_sink_down + 1]
        source_at_sink_up_state = state[:, self.source_at_sink_up : self.source_at_sink_up + 1]
        sink_at_source_down_state = state[:, self.sink_at_source_down : self.sink_at_source_down + 1]
        sink_at_source_up_state = state[:, self.sink_at_source_up : self.sink_at_source_up + 1]
        source_a_down_state = state[:, self.source_a_down : self.source_a_down + 1]
        source_a_up_state = state[:, self.source_a_up : self.source_a_up + 1]
        source_b_down_state = state[:, self.source_b_down : self.source_b_down + 1]
        source_b_up_state = state[:, self.source_b_up : self.source_b_up + 1]
        source_count_down_state = state[:, self.source_count_down : self.source_count_down + 1]
        source_count_up_state = state[:, self.source_count_up : self.source_count_up + 1]
        sink_count_down_state = state[:, self.sink_count_down : self.sink_count_down + 1]
        sink_count_up_state = state[:, self.sink_count_up : self.sink_count_up + 1]
        moment_state = state[:, self.moment_a_start : self.moment_b_start + 4]

        perceived = self.perception(state)
        organ_context = self.organ_read(organ_features)
        readout = self.readout(torch.cat([perceived, organ_context], dim=1))
        delta = self.delta(readout) * torch.sigmoid(self.update_gate(readout))

        source_down_target = self._propagate(source_down_state, self.source_down_kernel, source_marker)
        source_up_target = self._propagate(source_up_state, self.source_up_kernel, source_marker)
        sink_down_target = self._propagate(sink_down_state, self.sink_down_kernel, sink_marker)
        sink_up_target = self._propagate(sink_up_state, self.sink_up_kernel, sink_marker)
        source_a_down_target = self._propagate(source_a_down_state, self.source_down_kernel, source_a_marker)
        source_a_up_target = self._propagate(source_a_up_state, self.source_up_kernel, source_a_marker)
        source_b_down_target = self._propagate(source_b_down_state, self.source_down_kernel, source_b_marker)
        source_b_up_target = self._propagate(source_b_up_state, self.source_up_kernel, source_b_marker)
        source_count_down_target = self._count_propagate(
            source_count_down_state,
            self.vertical_down_kernel,
            source_marker,
        )
        source_count_up_target = self._count_propagate(
            source_count_up_state,
            self.vertical_up_kernel,
            source_marker,
        )
        sink_count_down_target = self._count_propagate(
            sink_count_down_state,
            self.vertical_down_kernel,
            sink_marker,
        )
        sink_count_up_target = self._count_propagate(
            sink_count_up_state,
            self.vertical_up_kernel,
            sink_marker,
        )
        source_basis = self._rank_basis(source_count_down_state, source_count_up_state)
        moment_seed = torch.cat(
            [
                source_basis * source_a_marker,
                source_basis * source_b_marker,
            ],
            dim=1,
        )
        moment_target = self._moment_target(moment_state, moment_seed)

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
                source_a_down_target,
                source_a_up_target,
                source_b_down_target,
                source_b_up_target,
                source_count_down_target,
                source_count_up_target,
                sink_count_down_target,
                sink_count_up_target,
                moment_target,
            ],
            dim=1,
        )
        wave_delta = (wave_targets - organ_features) * 0.42

        rule_context = state[:, self.rule_start : self.rule_start + self.rule_channels]
        local_output = self.local_match(torch.cat([wave_targets, rule_context], dim=1)) * sink_marker
        output_delta = local_output
        if self.rule_channels > 1:
            rule_presence = rule_context.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            output_delta = (delta[:, self.output_start : self.output_start + 2] + local_output) * rule_presence
        delta = delta.clone()
        delta[:, organ_slice] = wave_delta
        delta[:, self.output_start : self.output_start + 2] = output_delta
        return delta


UPDATE_RULES = (
    "standard",
    "gated_message",
    "self_tagging",
    "rank_binding",
    "sink_stabilized_rank",
    "matching_readout",
    "rule_cued_matching_readout",
    "rank_slot_rule_cued",
    "rank_slot_repair_rule_cued",
    "rank_slot_claim_rule_cued",
    "relative_rank_rule_cued",
)
