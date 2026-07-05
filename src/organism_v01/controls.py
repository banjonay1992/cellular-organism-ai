from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, save_json_report, set_seed
from organism_v01.metrics import (
    classification_accuracy,
    compute_loss,
    mean_sink_margin,
    target_peak_accuracy,
    target_set_accuracy,
)
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import SINK_ASSIGNMENTS, TASK_NAMES, RoutingBatch, generate_task_batch


ControlTransform = Callable[[RoutingBatch, ChannelLayout], RoutingBatch]


def erase_source(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    input_env = None if batch.input_env is None else batch.input_env.clone()
    initial[:, layout.source_a] = 0.0
    initial[:, layout.source_b] = 0.0
    env[:, layout.source_a] = 0.0
    env[:, layout.source_b] = 0.0
    if input_env is not None:
        input_env[:, layout.source_a] = 0.0
        input_env[:, layout.source_b] = 0.0
    return replace(batch, initial=initial, env=env, input_env=input_env)


def erase_sink_from_input(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    input_env = None if batch.input_env is None else batch.input_env.clone()
    initial[:, layout.sink] = 0.0
    env[:, layout.sink] = 0.0
    if layout.route_channels:
        sink_mask = batch.sink_mask.bool().expand(-1, layout.route_channels, -1, -1)
        initial[:, layout.route_slice] = initial[:, layout.route_slice].masked_fill(sink_mask, 0.0)
        env[:, layout.route_slice] = env[:, layout.route_slice].masked_fill(sink_mask, 0.0)
    if input_env is not None:
        input_env[:, layout.sink] = 0.0
        if layout.route_channels:
            input_env[:, layout.route_slice] = input_env[:, layout.route_slice].masked_fill(sink_mask, 0.0)
    return replace(batch, initial=initial, env=env, input_env=input_env)


def swap_source_label(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    input_env = None if batch.input_env is None else batch.input_env.clone()
    initial_a = initial[:, layout.source_a].clone()
    env_a = env[:, layout.source_a].clone()
    initial[:, layout.source_a] = initial[:, layout.source_b]
    initial[:, layout.source_b] = initial_a
    env[:, layout.source_a] = env[:, layout.source_b]
    env[:, layout.source_b] = env_a
    if input_env is not None:
        input_a = input_env[:, layout.source_a].clone()
        input_env[:, layout.source_a] = input_env[:, layout.source_b]
        input_env[:, layout.source_b] = input_a
    return replace(batch, initial=initial, env=env, input_env=input_env)


CONTROLS: dict[str, ControlTransform] = {
    "normal": lambda batch, layout: batch,
    "erase_source": erase_source,
    "erase_sink": erase_sink_from_input,
    "swap_source": swap_source_label,
}


def evaluate_control(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
    transform: ControlTransform,
    batches: int,
    batch_size: int,
    grid_size: int,
    rollout_steps: int,
    damage_prob: float,
    task: str,
    coordinate_fields: bool,
    pair_count: int,
    min_pair_spacing: int,
    sink_assignment: str,
    memory_input_steps: int,
    seed: int,
    device: torch.device,
    field_weight: float = 0.5,
    localization_weight: float = 1.0,
    localization_margin: float = 1.0,
    activity_weight: float = 1e-3,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "target_peak_accuracy": 0.0,
        "target_set_accuracy": 0.0,
        "sink_margin": 0.0,
    }
    with torch.no_grad():
        for index in range(batches):
            batch = generate_task_batch(
                task=task,
                batch_size=batch_size,
                grid_size=grid_size,
                layout=layout,
                damage_prob=damage_prob,
                coordinate_fields=coordinate_fields,
                pair_count=pair_count,
                min_pair_spacing=min_pair_spacing,
                sink_assignment=sink_assignment,
                memory_input_steps=memory_input_steps,
                seed=seed + index,
                device=device,
            )
            batch = transform(batch, layout)
            rollout = model(batch, steps=rollout_steps)
            losses = compute_loss(
                rollout.final_state,
                batch,
                layout,
                activity_loss=rollout.activity_loss,
                field_weight=field_weight,
                localization_weight=localization_weight,
                localization_margin=localization_margin,
                activity_weight=activity_weight,
            )
            totals["loss"] += float(losses["total"].item())
            totals["accuracy"] += classification_accuracy(rollout.final_state, batch, layout)
            totals["target_peak_accuracy"] += target_peak_accuracy(rollout.final_state, batch, layout)
            totals["target_set_accuracy"] += target_set_accuracy(rollout.final_state, batch, layout)
            totals["sink_margin"] += mean_sink_margin(rollout.final_state, batch, layout)
    return {key: value / batches for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation controls for organism v0.1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--task", choices=TASK_NAMES, default=None)
    parser.add_argument("--damage-prob", type=float, default=None)
    parser.add_argument("--coordinate-fields", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--min-pair-spacing", type=int, default=None)
    parser.add_argument("--sink-assignment", choices=SINK_ASSIGNMENTS, default=None)
    parser.add_argument("--memory-input-steps", type=int, default=None)
    parser.add_argument("--field-weight", type=float, default=None)
    parser.add_argument("--localization-weight", type=float, default=None)
    parser.add_argument("--localization-margin", type=float, default=None)
    parser.add_argument("--activity-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=9500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/controls-v01.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    checkpoint = torch.load(Path(args.model), map_location=device, weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 16}))
    model = CellularOrganism(
        layout=layout,
        cell_hidden=int(checkpoint_args.get("cell_hidden", 64)),
        update_rule=str(checkpoint_args.get("update_rule", "standard")),
        message_slots=int(checkpoint_args.get("message_slots", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 24))
    task = args.task or str(checkpoint_args.get("task", "routing"))
    damage_prob = args.damage_prob if args.damage_prob is not None else float(checkpoint_args.get("damage_prob", 0.12))
    coordinate_fields = args.coordinate_fields if args.coordinate_fields is not None else bool(checkpoint_args.get("coordinate_fields", True))
    pair_count = args.pair_count if args.pair_count is not None else int(checkpoint_args.get("pair_count", 3))
    min_pair_spacing = args.min_pair_spacing if args.min_pair_spacing is not None else int(checkpoint_args.get("min_pair_spacing", 1))
    sink_assignment = args.sink_assignment or str(checkpoint_args.get("sink_assignment", "aligned"))
    memory_input_steps = args.memory_input_steps if args.memory_input_steps is not None else int(checkpoint_args.get("memory_input_steps", 4))
    field_weight = args.field_weight if args.field_weight is not None else float(checkpoint_args.get("field_weight", 0.5))
    localization_weight = args.localization_weight if args.localization_weight is not None else float(checkpoint_args.get("localization_weight", 1.0))
    localization_margin = args.localization_margin if args.localization_margin is not None else float(checkpoint_args.get("localization_margin", 1.0))
    activity_weight = args.activity_weight if args.activity_weight is not None else float(checkpoint_args.get("activity_weight", 1e-3))

    results = {
        name: evaluate_control(
            model,
            layout,
            transform=transform,
            batches=args.batches,
            batch_size=batch_size,
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            damage_prob=damage_prob,
            task=task,
            coordinate_fields=coordinate_fields,
            pair_count=pair_count,
            min_pair_spacing=min_pair_spacing,
            sink_assignment=sink_assignment,
            memory_input_steps=memory_input_steps,
            seed=args.seed + offset * 10_000,
            device=device,
            field_weight=field_weight,
            localization_weight=localization_weight,
            localization_margin=localization_margin,
            activity_weight=activity_weight,
        )
        for offset, (name, transform) in enumerate(CONTROLS.items())
    }
    report = {
        "model": args.model,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": batch_size,
        "grid_size": grid_size,
        "rollout_steps": rollout_steps,
        "task": task,
        "damage_prob": damage_prob,
        "coordinate_fields": coordinate_fields,
        "pair_count": pair_count,
        "min_pair_spacing": min_pair_spacing,
        "sink_assignment": sink_assignment,
        "memory_input_steps": memory_input_steps,
        "field_weight": field_weight,
        "localization_weight": localization_weight,
        "localization_margin": localization_margin,
        "activity_weight": activity_weight,
        "controls": results,
    }
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
