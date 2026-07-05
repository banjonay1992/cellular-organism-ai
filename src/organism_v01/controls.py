from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, save_json_report, set_seed
from organism_v01.metrics import classification_accuracy, compute_loss, mean_sink_margin, target_peak_accuracy
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import RoutingBatch, generate_routing_batch


ControlTransform = Callable[[RoutingBatch, ChannelLayout], RoutingBatch]


def erase_source(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    initial[:, layout.source_a] = 0.0
    initial[:, layout.source_b] = 0.0
    env[:, layout.source_a] = 0.0
    env[:, layout.source_b] = 0.0
    return replace(batch, initial=initial, env=env)


def erase_sink_from_input(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    initial[:, layout.sink] = 0.0
    env[:, layout.sink] = 0.0
    return replace(batch, initial=initial, env=env)


def swap_source_label(batch: RoutingBatch, layout: ChannelLayout) -> RoutingBatch:
    initial = batch.initial.clone()
    env = batch.env.clone()
    initial_a = initial[:, layout.source_a].clone()
    env_a = env[:, layout.source_a].clone()
    initial[:, layout.source_a] = initial[:, layout.source_b]
    initial[:, layout.source_b] = initial_a
    env[:, layout.source_a] = env[:, layout.source_b]
    env[:, layout.source_b] = env_a
    return replace(batch, initial=initial, env=env)


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
        "sink_margin": 0.0,
    }
    with torch.no_grad():
        for index in range(batches):
            batch = generate_routing_batch(
                batch_size=batch_size,
                grid_size=grid_size,
                layout=layout,
                damage_prob=damage_prob,
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
            totals["sink_margin"] += mean_sink_margin(rollout.final_state, batch, layout)
    return {key: value / batches for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation controls for organism v0.1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--damage-prob", type=float, default=None)
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 24))
    damage_prob = args.damage_prob if args.damage_prob is not None else float(checkpoint_args.get("damage_prob", 0.12))
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
        "damage_prob": damage_prob,
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
