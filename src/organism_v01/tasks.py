from __future__ import annotations

from dataclasses import dataclass

import torch

from organism_v01.channels import ChannelLayout


@dataclass(frozen=True)
class RoutingBatch:
    initial: torch.Tensor
    env: torch.Tensor
    target: torch.Tensor
    sink_mask: torch.Tensor
    alive_mask: torch.Tensor
    labels: torch.Tensor
    source_rc: torch.Tensor
    sink_rc: torch.Tensor

    def to(self, device: torch.device | str) -> "RoutingBatch":
        return RoutingBatch(
            initial=self.initial.to(device),
            env=self.env.to(device),
            target=self.target.to(device),
            sink_mask=self.sink_mask.to(device),
            alive_mask=self.alive_mask.to(device),
            labels=self.labels.to(device),
            source_rc=self.source_rc.to(device),
            sink_rc=self.sink_rc.to(device),
        )


def _make_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low=low, high=high, size=(1,), generator=generator).item())


def generate_routing_batch(
    *,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    damage_prob: float = 0.12,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    """Generate a routing task.

    A source cell carries label A or B. A separate sink cell marks where the
    answer should appear. The target is computed from the sampled label and sink.
    """

    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if not 0.0 <= damage_prob < 0.6:
        raise ValueError("damage_prob must be in [0.0, 0.6)")

    generator = _make_generator(seed)
    height = width = grid_size
    state = torch.zeros(batch_size, layout.total_channels, height, width)
    target = torch.zeros(batch_size, layout.output_count, height, width)
    labels = torch.zeros(batch_size, dtype=torch.long)
    source_rc = torch.zeros(batch_size, 2, dtype=torch.long)
    sink_rc = torch.zeros(batch_size, 2, dtype=torch.long)

    blocked = torch.rand(batch_size, height, width, generator=generator) < damage_prob
    blocked[:, 0, :] = True
    blocked[:, -1, :] = True
    blocked[:, :, 0] = True
    blocked[:, :, -1] = True

    for item in range(batch_size):
        source_row = _randint(generator, 1, height - 1)
        sink_row = _randint(generator, 1, height - 1)
        source_col = 1
        sink_col = width - 2
        label = _randint(generator, 0, 2)

        blocked[item, source_row, source_col] = False
        blocked[item, sink_row, sink_col] = False

        labels[item] = label
        source_rc[item] = torch.tensor([source_row, source_col])
        sink_rc[item] = torch.tensor([sink_row, sink_col])

        source_channel = layout.source_a if label == 0 else layout.source_b
        state[item, source_channel, source_row, source_col] = 1.0
        state[item, layout.sink, sink_row, sink_col] = 1.0
        target[item, label, sink_row, sink_col] = 1.0

    alive = (~blocked).float()
    state[:, layout.blocked] = blocked.float()
    state[:, layout.alive] = alive

    x_values = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(batch_size, height, width)
    y_values = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(batch_size, height, width)
    state[:, layout.x_field] = x_values
    state[:, layout.y_field] = y_values

    env = state[:, : layout.env_count].clone()
    batch = RoutingBatch(
        initial=state,
        env=env,
        target=target,
        sink_mask=state[:, layout.sink : layout.sink + 1].clone(),
        alive_mask=alive.unsqueeze(1),
        labels=labels,
        source_rc=source_rc,
        sink_rc=sink_rc,
    )
    if device is not None:
        batch = batch.to(device)
    return batch
