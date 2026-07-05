from __future__ import annotations

import argparse
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import save_json_report
from organism_v01.tasks import SINK_ASSIGNMENTS, generate_multi_pair_batch


def audit_assignment_ambiguity(
    *,
    assignment_a: str,
    assignment_b: str,
    seeds: int,
    start_seed: int,
    batch_size: int,
    grid_size: int,
    pair_count: int,
    min_pair_spacing: int,
    damage_prob: float,
    hidden_channels: int = 4,
) -> dict[str, Any]:
    if assignment_a not in SINK_ASSIGNMENTS:
        raise ValueError(f"assignment_a must be one of {SINK_ASSIGNMENTS}")
    if assignment_b not in SINK_ASSIGNMENTS:
        raise ValueError(f"assignment_b must be one of {SINK_ASSIGNMENTS}")
    if seeds <= 0:
        raise ValueError("seeds must be positive")

    layout = ChannelLayout(hidden_channels=hidden_channels)
    total_items = seeds * batch_size
    identical_input_items = 0
    conflicting_target_items = 0
    examples: list[dict[str, Any]] = []

    for offset in range(seeds):
        seed = start_seed + offset
        first = generate_multi_pair_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            pair_count=pair_count,
            min_pair_spacing=min_pair_spacing,
            sink_assignment=assignment_a,
            damage_prob=damage_prob,
            seed=seed,
        )
        second = generate_multi_pair_batch(
            batch_size=batch_size,
            grid_size=grid_size,
            layout=layout,
            pair_count=pair_count,
            min_pair_spacing=min_pair_spacing,
            sink_assignment=assignment_b,
            damage_prob=damage_prob,
            seed=seed,
        )

        same_input = torch.isclose(first.env, second.env).flatten(1).all(dim=1)
        same_target = torch.isclose(first.target, second.target).flatten(1).all(dim=1)
        conflicts = same_input & ~same_target
        identical_input_items += int(same_input.sum().item())
        conflicting_target_items += int(conflicts.sum().item())

        if len(examples) < 5 and bool(conflicts.any()):
            item = int(conflicts.nonzero(as_tuple=False)[0].item())
            examples.append(
                {
                    "seed": seed,
                    "item": item,
                    "pair_labels": None if first.pair_labels is None else first.pair_labels[item].tolist(),
                    "source_rc": None if first.pair_source_rc is None else first.pair_source_rc[item].tolist(),
                    f"{assignment_a}_sink_rc": None
                    if first.pair_sink_rc is None
                    else first.pair_sink_rc[item].tolist(),
                    f"{assignment_b}_sink_rc": None
                    if second.pair_sink_rc is None
                    else second.pair_sink_rc[item].tolist(),
                }
            )

    conflict_rate = conflicting_target_items / max(identical_input_items, 1)
    return {
        "assignment_a": assignment_a,
        "assignment_b": assignment_b,
        "seeds": seeds,
        "start_seed": start_seed,
        "batch_size": batch_size,
        "grid_size": grid_size,
        "pair_count": pair_count,
        "min_pair_spacing": min_pair_spacing,
        "damage_prob": damage_prob,
        "total_items": total_items,
        "identical_input_items": identical_input_items,
        "conflicting_target_items": conflicting_target_items,
        "conflict_rate_given_identical_input": conflict_rate,
        "examples": examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether assignment variants create identical inputs with conflicting targets.")
    parser.add_argument("--assignment-a", choices=SINK_ASSIGNMENTS, default="reverse")
    parser.add_argument("--assignment-b", choices=SINK_ASSIGNMENTS, default="cycle")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--pair-count", type=int, default=3)
    parser.add_argument("--min-pair-spacing", type=int, default=1)
    parser.add_argument("--damage-prob", type=float, default=0.10)
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--report", default="outputs/reports/assignment-ambiguity.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_assignment_ambiguity(
        assignment_a=args.assignment_a,
        assignment_b=args.assignment_b,
        seeds=args.seeds,
        start_seed=args.start_seed,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        pair_count=args.pair_count,
        min_pair_spacing=args.min_pair_spacing,
        damage_prob=args.damage_prob,
        hidden_channels=args.hidden_channels,
    )
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
