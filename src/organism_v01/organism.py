from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from organism_v01.cell import (
    UPDATE_RULES,
    CellUpdate,
    GatedMessageCellUpdate,
    MatchingReadoutCellUpdate,
    RankBindingCellUpdate,
    SelfTaggingCellUpdate,
    SinkStabilizedRankCellUpdate,
)
from organism_v01.channels import ChannelLayout
from organism_v01.tasks import RoutingBatch


@dataclass(frozen=True)
class OrganismRollout:
    final_state: torch.Tensor
    activity_loss: torch.Tensor
    frames: torch.Tensor | None = None


def _active_env(batch: RoutingBatch, step_index: int) -> torch.Tensor:
    if batch.input_env is not None and step_index < batch.input_steps:
        return batch.input_env
    return batch.env


def clamp_environment(
    state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    step_index: int = 0,
) -> torch.Tensor:
    clamped = state.clone()
    clamped[:, : layout.env_count] = _active_env(batch, step_index)
    clamped[:, layout.mutable_slice] = clamped[:, layout.mutable_slice] * batch.alive_mask
    return clamped


class CellularOrganism(nn.Module):
    """A recurrent grid of tiny shared cell updates."""

    def __init__(
        self,
        layout: ChannelLayout,
        cell_hidden: int = 64,
        update_scale: float = 1.0,
        update_rule: str = "standard",
        message_slots: int = 8,
        tag_slots: int = 4,
    ) -> None:
        super().__init__()
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        self.layout = layout
        self.update_scale = update_scale
        self.update_rule = update_rule
        self.message_slots = message_slots
        self.tag_slots = tag_slots
        if update_rule == "standard":
            self.cell_update = CellUpdate(layout.total_channels, hidden=cell_hidden)
        elif update_rule == "gated_message":
            self.cell_update = GatedMessageCellUpdate(
                layout.total_channels,
                hidden=cell_hidden,
                message_slots=message_slots,
            )
        elif update_rule == "self_tagging":
            self.cell_update = SelfTaggingCellUpdate(
                layout.total_channels,
                hidden_start=layout.hidden_start,
                hidden_channels=layout.hidden_channels,
                hidden=cell_hidden,
                tag_slots=tag_slots,
            )
        elif update_rule == "rank_binding":
            self.cell_update = RankBindingCellUpdate(
                layout.total_channels,
                hidden_start=layout.hidden_start,
                hidden_channels=layout.hidden_channels,
                source_a=layout.source_a,
                source_b=layout.source_b,
                sink=layout.sink,
                hidden=cell_hidden,
            )
        elif update_rule == "sink_stabilized_rank":
            self.cell_update = SinkStabilizedRankCellUpdate(
                layout.total_channels,
                hidden_start=layout.hidden_start,
                hidden_channels=layout.hidden_channels,
                source_a=layout.source_a,
                source_b=layout.source_b,
                sink=layout.sink,
                hidden=cell_hidden,
            )
        else:
            self.cell_update = MatchingReadoutCellUpdate(
                layout.total_channels,
                hidden_start=layout.hidden_start,
                hidden_channels=layout.hidden_channels,
                source_a=layout.source_a,
                source_b=layout.source_b,
                sink=layout.sink,
                output_start=layout.output_start,
                hidden=cell_hidden,
            )

    def forward(
        self,
        batch: RoutingBatch,
        steps: int = 24,
        *,
        start_state: torch.Tensor | None = None,
        start_step: int = 0,
        return_frames: bool = False,
    ) -> OrganismRollout:
        if steps <= 0:
            raise ValueError("steps must be positive")

        state = batch.initial if start_state is None else start_state
        state = clamp_environment(state, batch, self.layout, step_index=start_step)
        activity_terms: list[torch.Tensor] = []
        frames: list[torch.Tensor] = []
        if return_frames:
            frames.append(state.detach().cpu())

        for offset in range(steps):
            step_index = start_step + offset
            delta = self.cell_update(state) * self.update_scale
            delta = delta.clone()
            delta[:, : self.layout.env_count] = 0.0
            delta[:, self.layout.mutable_slice] = delta[:, self.layout.mutable_slice] * batch.alive_mask
            state = clamp_environment(state + delta, batch, self.layout, step_index=step_index + 1)
            activity_terms.append(delta[:, self.layout.mutable_slice].abs().mean())
            if return_frames:
                frames.append(state.detach().cpu())

        activity_loss = torch.stack(activity_terms).mean()
        frame_tensor = torch.stack(frames) if return_frames else None
        return OrganismRollout(final_state=state, activity_loss=activity_loss, frames=frame_tensor)
