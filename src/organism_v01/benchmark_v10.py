from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.controls import CONTROLS, erase_rule_cue, evaluate_control
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import evaluate_dynamic_injury
from organism_v01.organism import CellularOrganism


V10_THRESHOLDS = {
    "reverse_target_set_accuracy": 0.65,
    "injury_target_set_accuracy": 0.55,
    "cycle_target_set_accuracy": 0.45,
    "normal_control_target_set_accuracy": 0.65,
    "erase_source_max_target_set_accuracy": 0.35,
    "erase_sink_max_target_set_accuracy": 0.35,
    "swap_source_max_target_set_accuracy": 0.10,
    "erase_rule_max_target_set_accuracy": 0.55,
}


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[CellularOrganism, ChannelLayout, dict[str, Any]]:
    checkpoint = torch.load(Path(model_path), map_location=device, weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 8}))
    if layout.route_channels != 0:
        raise ValueError("v0.10 benchmark forbids pair route cues: checkpoint route_channels must be 0")
    if layout.rule_channels < 1:
        raise ValueError("v0.10 benchmark requires a global rule cue: checkpoint rule_channels must be at least 1")

    model = CellularOrganism(
        layout=layout,
        cell_hidden=int(checkpoint_args.get("cell_hidden", 32)),
        update_rule=str(checkpoint_args.get("update_rule", "standard")),
        message_slots=int(checkpoint_args.get("message_slots", 8)),
        tag_slots=int(checkpoint_args.get("tag_slots", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, layout, checkpoint_args


def summarize_v10_result(
    reverse: dict[str, float],
    injury: dict[str, float],
    cycle: dict[str, float],
    controls: dict[str, dict[str, float]],
) -> dict[str, Any]:
    checks = {
        "reverse_target_set_accuracy": reverse["target_set_accuracy"]
        >= V10_THRESHOLDS["reverse_target_set_accuracy"],
        "injury_target_set_accuracy": injury["target_set_accuracy"]
        >= V10_THRESHOLDS["injury_target_set_accuracy"],
        "cycle_target_set_accuracy": cycle["target_set_accuracy"]
        >= V10_THRESHOLDS["cycle_target_set_accuracy"],
        "normal_control_target_set_accuracy": controls["normal"]["target_set_accuracy"]
        >= V10_THRESHOLDS["normal_control_target_set_accuracy"],
        "erase_source_target_set_accuracy": controls["erase_source"]["target_set_accuracy"]
        <= V10_THRESHOLDS["erase_source_max_target_set_accuracy"],
        "erase_sink_target_set_accuracy": controls["erase_sink"]["target_set_accuracy"]
        <= V10_THRESHOLDS["erase_sink_max_target_set_accuracy"],
        "swap_source_target_set_accuracy": controls["swap_source"]["target_set_accuracy"]
        <= V10_THRESHOLDS["swap_source_max_target_set_accuracy"],
        "erase_rule_target_set_accuracy": controls["erase_rule"]["target_set_accuracy"]
        <= V10_THRESHOLDS["erase_rule_max_target_set_accuracy"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": V10_THRESHOLDS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.10 rule-cued uncued-pair binding benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--injury-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=16000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v10.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 40))
    field_weight = float(checkpoint_args.get("field_weight", 0.5))
    localization_weight = float(checkpoint_args.get("localization_weight", 1.0))
    localization_margin = float(checkpoint_args.get("localization_margin", 1.0))
    activity_weight = float(checkpoint_args.get("activity_weight", 1e-3))

    common = {
        "batch_size": batch_size,
        "grid_size": grid_size,
        "damage_prob": 0.10,
        "task": "multi",
        "coordinate_fields": True,
        "pair_count": 3,
        "min_pair_spacing": 1,
        "memory_input_steps": 4,
        "field_weight": field_weight,
        "localization_weight": localization_weight,
        "localization_margin": localization_margin,
        "activity_weight": activity_weight,
    }

    reverse = evaluate_model(
        model,
        layout,
        batches=args.batches,
        rollout_steps=rollout_steps,
        sink_assignment="reverse",
        seed=args.seed,
        device=device,
        **common,
    )
    injury = evaluate_dynamic_injury(
        model,
        layout,
        batches=max(8, args.batches // 2),
        pre_steps=max(1, rollout_steps // 2),
        post_steps=max(1, rollout_steps - max(1, rollout_steps // 2)),
        injury_prob=args.injury_prob,
        sink_assignment="reverse",
        seed=args.seed + 20_000,
        device=device,
        **common,
    )
    control_transforms = {**CONTROLS, "erase_rule": erase_rule_cue}
    controls = {
        name: evaluate_control(
            model,
            layout,
            transform=transform,
            batches=max(8, args.batches // 2),
            rollout_steps=rollout_steps,
            sink_assignment="reverse",
            seed=args.seed + 40_000 + index * 10_000,
            device=device,
            **common,
        )
        for index, (name, transform) in enumerate(control_transforms.items())
    }
    cycle = evaluate_model(
        model,
        layout,
        batches=max(8, args.batches // 2),
        rollout_steps=rollout_steps,
        sink_assignment="cycle",
        seed=args.seed + 90_000,
        device=device,
        **common,
    )
    pair_count_2 = evaluate_model(
        model,
        layout,
        batches=max(6, args.batches // 4),
        rollout_steps=rollout_steps,
        sink_assignment="reverse",
        seed=args.seed + 100_000,
        device=device,
        **{**common, "pair_count": 2},
    )
    larger_grid = evaluate_model(
        model,
        layout,
        batches=max(6, args.batches // 4),
        rollout_steps=rollout_steps,
        sink_assignment="reverse",
        seed=args.seed + 110_000,
        device=device,
        **{**common, "grid_size": grid_size + 4},
    )
    summary = summarize_v10_result(reverse, injury, cycle, controls)
    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.10_rule_cued_binding",
        "config": {
            "batches": args.batches,
            "batch_size": batch_size,
            "grid_size": grid_size,
            "rollout_steps": rollout_steps,
            "injury_prob": args.injury_prob,
            **common,
        },
        "reverse": reverse,
        "injury": injury,
        "cycle": cycle,
        "controls": controls,
        "stress": {
            "pair_count_2": pair_count_2,
            "larger_grid": larger_grid,
        },
        "summary": summary,
    }
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
