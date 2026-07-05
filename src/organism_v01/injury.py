from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

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


def apply_random_injury(
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    injury_prob: float,
    seed: int,
) -> RoutingBatch:
    if not 0.0 <= injury_prob < 0.8:
        raise ValueError("injury_prob must be in [0.0, 0.8)")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    env_cpu = batch.env.detach().cpu().clone()
    initial_cpu = batch.initial.detach().cpu().clone()
    blocked = env_cpu[:, layout.blocked].bool()
    random_hits = torch.rand(blocked.shape, generator=generator) < injury_prob
    protected = (
        initial_cpu[:, layout.source_a].bool()
        | initial_cpu[:, layout.source_b].bool()
        | batch.target.detach().cpu().bool().any(dim=1)
        | batch.sink_mask.detach().cpu()[:, 0].bool()
    )
    new_blocked = blocked | random_hits
    new_blocked[:, 0, :] = True
    new_blocked[:, -1, :] = True
    new_blocked[:, :, 0] = True
    new_blocked[:, :, -1] = True
    new_blocked[protected] = False
    alive = (~new_blocked).float()

    env_cpu[:, layout.blocked] = new_blocked.float()
    env_cpu[:, layout.alive] = alive
    initial_cpu[:, layout.blocked] = new_blocked.float()
    initial_cpu[:, layout.alive] = alive

    input_env = None
    if batch.input_env is not None:
        input_env = batch.input_env.detach().cpu().clone()
        input_env[:, layout.blocked] = new_blocked.float()
        input_env[:, layout.alive] = alive

    injured = replace(
        batch,
        initial=initial_cpu.to(batch.initial.device),
        env=env_cpu.to(batch.env.device),
        alive_mask=alive.unsqueeze(1).to(batch.alive_mask.device),
        input_env=None if input_env is None else input_env.to(batch.input_env.device),
    )
    return injured


def evaluate_dynamic_injury(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
    batches: int,
    batch_size: int,
    grid_size: int,
    pre_steps: int,
    post_steps: int,
    damage_prob: float,
    injury_prob: float,
    task: str,
    coordinate_fields: bool,
    pair_count: int,
    min_pair_spacing: int,
    sink_assignment: str,
    memory_input_steps: int,
    seed: int,
    device: torch.device,
    field_weight: float,
    localization_weight: float,
    localization_margin: float,
    activity_weight: float,
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
            first = model(batch, steps=pre_steps)
            injured = apply_random_injury(batch, layout, injury_prob=injury_prob, seed=seed + 50_000 + index).to(device)
            second = model(
                injured,
                steps=post_steps,
                start_state=first.final_state,
                start_step=pre_steps,
            )
            losses = compute_loss(
                second.final_state,
                injured,
                layout,
                activity_loss=second.activity_loss,
                field_weight=field_weight,
                localization_weight=localization_weight,
                localization_margin=localization_margin,
                activity_weight=activity_weight,
            )
            totals["loss"] += float(losses["total"].item())
            totals["accuracy"] += classification_accuracy(second.final_state, injured, layout)
            totals["target_peak_accuracy"] += target_peak_accuracy(second.final_state, injured, layout)
            totals["target_set_accuracy"] += target_set_accuracy(second.final_state, injured, layout)
            totals["sink_margin"] += mean_sink_margin(second.final_state, injured, layout)
    return {key: value / batches for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate recovery after mid-rollout tissue damage.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--post-steps", type=int, default=None)
    parser.add_argument("--task", choices=TASK_NAMES, default=None)
    parser.add_argument("--damage-prob", type=float, default=None)
    parser.add_argument("--injury-prob", type=float, default=0.25)
    parser.add_argument("--coordinate-fields", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--min-pair-spacing", type=int, default=None)
    parser.add_argument("--sink-assignment", choices=SINK_ASSIGNMENTS, default=None)
    parser.add_argument("--memory-input-steps", type=int, default=None)
    parser.add_argument("--field-weight", type=float, default=None)
    parser.add_argument("--localization-weight", type=float, default=None)
    parser.add_argument("--localization-margin", type=float, default=None)
    parser.add_argument("--activity-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/injury-v02.json")
    return parser


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("args", {}))


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    checkpoint = torch.load(Path(args.model), map_location=device, weights_only=False)
    checkpoint_args = _checkpoint_args(checkpoint)
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 8}))
    model = CellularOrganism(
        layout=layout,
        cell_hidden=int(checkpoint_args.get("cell_hidden", 32)),
        update_rule=str(checkpoint_args.get("update_rule", "standard")),
        message_slots=int(checkpoint_args.get("message_slots", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rollout_steps = int(checkpoint_args.get("rollout_steps", 24))
    pre_steps = args.pre_steps if args.pre_steps is not None else max(1, rollout_steps // 2)
    post_steps = args.post_steps if args.post_steps is not None else max(1, rollout_steps - pre_steps)
    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
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

    metrics = evaluate_dynamic_injury(
        model,
        layout,
        batches=args.batches,
        batch_size=batch_size,
        grid_size=grid_size,
        pre_steps=pre_steps,
        post_steps=post_steps,
        damage_prob=damage_prob,
        injury_prob=args.injury_prob,
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
        "pre_steps": pre_steps,
        "post_steps": post_steps,
        "task": task,
        "damage_prob": damage_prob,
        "injury_prob": args.injury_prob,
        "coordinate_fields": coordinate_fields,
        "pair_count": pair_count,
        "min_pair_spacing": min_pair_spacing,
        "sink_assignment": sink_assignment,
        "memory_input_steps": memory_input_steps,
        "metrics": metrics,
    }
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
