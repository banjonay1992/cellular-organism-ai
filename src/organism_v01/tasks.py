from __future__ import annotations

from dataclasses import dataclass, replace

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
    pair_labels: torch.Tensor | None = None
    pair_source_rc: torch.Tensor | None = None
    pair_sink_rc: torch.Tensor | None = None
    input_env: torch.Tensor | None = None
    input_steps: int = 0
    task_name: str = "routing"

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
            pair_labels=None if self.pair_labels is None else self.pair_labels.to(device),
            pair_source_rc=None if self.pair_source_rc is None else self.pair_source_rc.to(device),
            pair_sink_rc=None if self.pair_sink_rc is None else self.pair_sink_rc.to(device),
            input_env=None if self.input_env is None else self.input_env.to(device),
            input_steps=self.input_steps,
            task_name=self.task_name,
        )


def _make_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low=low, high=high, size=(1,), generator=generator).item())


def _validate_common(grid_size: int, damage_prob: float) -> None:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if not 0.0 <= damage_prob < 0.6:
        raise ValueError("damage_prob must be in [0.0, 0.6)")


def _base_state(
    *,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    damage_prob: float,
    coordinate_fields: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    height = width = grid_size
    state = torch.zeros(batch_size, layout.total_channels, height, width)
    blocked = torch.rand(batch_size, height, width, generator=generator) < damage_prob
    blocked[:, 0, :] = True
    blocked[:, -1, :] = True
    blocked[:, :, 0] = True
    blocked[:, :, -1] = True

    if coordinate_fields:
        x_values = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(batch_size, height, width)
        y_values = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(batch_size, height, width)
        state[:, layout.x_field] = x_values
        state[:, layout.y_field] = y_values

    return state, blocked


def _apply_maze_barrier(
    *,
    blocked: torch.Tensor,
    generator: torch.Generator,
    protected: torch.Tensor,
) -> None:
    _, height, width = blocked.shape
    wall_col = width // 2
    for item in range(blocked.shape[0]):
        gap_row = _randint(generator, 1, height - 1)
        blocked[item, 1 : height - 1, wall_col] = True
        blocked[item, gap_row, wall_col] = False
    blocked[protected] = False


def _finalize_batch(
    *,
    state: torch.Tensor,
    blocked: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    source_rc: torch.Tensor,
    sink_rc: torch.Tensor,
    layout: ChannelLayout,
    task_name: str,
    pair_labels: torch.Tensor | None = None,
    pair_source_rc: torch.Tensor | None = None,
    pair_sink_rc: torch.Tensor | None = None,
    input_env: torch.Tensor | None = None,
    input_steps: int = 0,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    alive = (~blocked).float()
    state[:, layout.blocked] = blocked.float()
    state[:, layout.alive] = alive
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
        pair_labels=pair_labels,
        pair_source_rc=pair_source_rc,
        pair_sink_rc=pair_sink_rc,
        input_env=input_env,
        input_steps=input_steps,
        task_name=task_name,
    )
    if device is not None:
        batch = batch.to(device)
    return batch


def generate_routing_batch(
    *,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    damage_prob: float = 0.12,
    coordinate_fields: bool = True,
    maze_barrier: bool = False,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    """Generate a routing task.

    A source cell carries label A or B. A separate sink cell marks where the
    answer should appear. The target is computed from the sampled label and sink.
    """

    _validate_common(grid_size, damage_prob)

    generator = _make_generator(seed)
    height = width = grid_size
    state, blocked = _base_state(
        batch_size=batch_size,
        grid_size=grid_size,
        layout=layout,
        damage_prob=damage_prob,
        coordinate_fields=coordinate_fields,
        generator=generator,
    )
    target = torch.zeros(batch_size, layout.output_count, height, width)
    labels = torch.zeros(batch_size, dtype=torch.long)
    source_rc = torch.zeros(batch_size, 2, dtype=torch.long)
    sink_rc = torch.zeros(batch_size, 2, dtype=torch.long)
    protected = torch.zeros_like(blocked)

    for item in range(batch_size):
        source_row = _randint(generator, 1, height - 1)
        sink_row = _randint(generator, 1, height - 1)
        source_col = 1
        sink_col = width - 2
        label = _randint(generator, 0, 2)

        blocked[item, source_row, source_col] = False
        blocked[item, sink_row, sink_col] = False
        protected[item, source_row, source_col] = True
        protected[item, sink_row, sink_col] = True

        labels[item] = label
        source_rc[item] = torch.tensor([source_row, source_col])
        sink_rc[item] = torch.tensor([sink_row, sink_col])

        source_channel = layout.source_a if label == 0 else layout.source_b
        state[item, source_channel, source_row, source_col] = 1.0
        state[item, layout.sink, sink_row, sink_col] = 1.0
        target[item, label, sink_row, sink_col] = 1.0

    if maze_barrier:
        _apply_maze_barrier(blocked=blocked, generator=generator, protected=protected)

    return _finalize_batch(
        state=state,
        blocked=blocked,
        target=target,
        labels=labels,
        source_rc=source_rc,
        sink_rc=sink_rc,
        layout=layout,
        task_name="maze" if maze_barrier else "routing",
        device=device,
    )


def generate_multi_pair_batch(
    *,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    pair_count: int = 3,
    damage_prob: float = 0.08,
    coordinate_fields: bool = True,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    """Generate several independent row-aligned source/sink pairs per item."""

    _validate_common(grid_size, damage_prob)
    if pair_count < 2:
        raise ValueError("pair_count must be at least 2 for multi-pair tasks")
    if pair_count > grid_size - 2:
        raise ValueError("pair_count cannot exceed the number of interior rows")

    generator = _make_generator(seed)
    height = width = grid_size
    state, blocked = _base_state(
        batch_size=batch_size,
        grid_size=grid_size,
        layout=layout,
        damage_prob=damage_prob,
        coordinate_fields=coordinate_fields,
        generator=generator,
    )
    target = torch.zeros(batch_size, layout.output_count, height, width)
    pair_labels = torch.zeros(batch_size, pair_count, dtype=torch.long)
    pair_source_rc = torch.zeros(batch_size, pair_count, 2, dtype=torch.long)
    pair_sink_rc = torch.zeros(batch_size, pair_count, 2, dtype=torch.long)
    protected = torch.zeros_like(blocked)

    for item in range(batch_size):
        rows = torch.randperm(height - 2, generator=generator)[:pair_count] + 1
        rows, _ = rows.sort()
        for pair_index, row_tensor in enumerate(rows):
            row = int(row_tensor.item())
            label = _randint(generator, 0, 2)
            source_col = 1
            sink_col = width - 2
            source_channel = layout.source_a if label == 0 else layout.source_b

            pair_labels[item, pair_index] = label
            pair_source_rc[item, pair_index] = torch.tensor([row, source_col])
            pair_sink_rc[item, pair_index] = torch.tensor([row, sink_col])
            state[item, source_channel, row, source_col] = 1.0
            state[item, layout.sink, row, sink_col] = 1.0
            target[item, label, row, sink_col] = 1.0
            blocked[item, row, source_col] = False
            blocked[item, row, sink_col] = False
            protected[item, row, source_col] = True
            protected[item, row, sink_col] = True

    labels = pair_labels[:, 0].clone()
    source_rc = pair_source_rc[:, 0].clone()
    sink_rc = pair_sink_rc[:, 0].clone()
    return _finalize_batch(
        state=state,
        blocked=blocked,
        target=target,
        labels=labels,
        source_rc=source_rc,
        sink_rc=sink_rc,
        layout=layout,
        task_name="multi",
        pair_labels=pair_labels,
        pair_source_rc=pair_source_rc,
        pair_sink_rc=pair_sink_rc,
        device=device,
    )


def generate_memory_batch(
    *,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    damage_prob: float = 0.08,
    coordinate_fields: bool = True,
    input_steps: int = 4,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    """Generate a delayed recall task.

    The source label is visible only during the input phase. The organism must
    carry that information in mutable tissue until the final sink readout.
    """

    if input_steps <= 0:
        raise ValueError("input_steps must be positive")

    batch = generate_routing_batch(
        batch_size=batch_size,
        grid_size=grid_size,
        layout=layout,
        damage_prob=damage_prob,
        coordinate_fields=coordinate_fields,
        seed=seed,
        device=None,
    )
    input_env = batch.env.clone()
    env = batch.env.clone()
    env[:, layout.source_a] = 0.0
    env[:, layout.source_b] = 0.0
    initial = batch.initial.clone()
    alive_mask = batch.alive_mask.clone()
    remembered = replace(
        batch,
        initial=initial,
        env=env,
        input_env=input_env,
        input_steps=input_steps,
        alive_mask=alive_mask,
        task_name="memory",
    )
    if device is not None:
        remembered = remembered.to(device)
    return remembered


TASK_NAMES = ("routing", "maze", "memory", "multi")


def generate_task_batch(
    *,
    task: str,
    batch_size: int,
    grid_size: int,
    layout: ChannelLayout,
    damage_prob: float,
    coordinate_fields: bool = True,
    pair_count: int = 3,
    memory_input_steps: int = 4,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> RoutingBatch:
    if task == "routing":
        return generate_routing_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            damage_prob=damage_prob,
            coordinate_fields=coordinate_fields,
            seed=seed,
            device=device,
        )
    if task == "maze":
        return generate_routing_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            damage_prob=damage_prob,
            coordinate_fields=coordinate_fields,
            maze_barrier=True,
            seed=seed,
            device=device,
        )
    if task == "memory":
        return generate_memory_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            damage_prob=damage_prob,
            coordinate_fields=coordinate_fields,
            input_steps=memory_input_steps,
            seed=seed,
            device=device,
        )
    if task == "multi":
        return generate_multi_pair_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            pair_count=pair_count,
            damage_prob=damage_prob,
            coordinate_fields=coordinate_fields,
            seed=seed,
            device=device,
        )
    raise ValueError(f"unknown task: {task}")
