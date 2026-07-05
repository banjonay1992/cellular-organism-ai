from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import SINK_ASSIGNMENTS, TASK_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an organism v0.1 checkpoint.")
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
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/eval-v01.json")
    return parser


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("args", {}))


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    checkpoint = torch.load(Path(args.model), map_location=device, weights_only=False)
    checkpoint_args = _checkpoint_args(checkpoint)
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 16}))
    cell_hidden = int(checkpoint_args.get("cell_hidden", 64))
    model = CellularOrganism(
        layout=layout,
        cell_hidden=cell_hidden,
        update_rule=str(checkpoint_args.get("update_rule", "standard")),
        message_slots=int(checkpoint_args.get("message_slots", 8)),
        tag_slots=int(checkpoint_args.get("tag_slots", 4)),
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

    metrics = evaluate_model(
        model,
        layout,
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
        seed=args.seed,
        device=device,
        field_weight=field_weight,
        localization_weight=localization_weight,
        localization_margin=localization_margin,
        activity_weight=activity_weight,
    )
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
        "metrics": metrics,
    }
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
