from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from organism_v01.benchmark_v10 import load_model
from organism_v01.controls import CONTROLS, erase_rule_cue, evaluate_control
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed


V12_THRESHOLDS = {
    "reverse_target_set_accuracy": 0.60,
    "cycle_target_set_accuracy": 0.50,
    "reverse_slot_accuracy": 0.80,
    "cycle_slot_accuracy": 0.80,
    "reverse_routed_slot_accuracy": 0.90,
    "cycle_routed_slot_accuracy": 0.90,
    "normal_control_target_set_accuracy": 0.60,
    "erase_source_max_target_set_accuracy": 0.35,
    "erase_sink_max_target_set_accuracy": 0.35,
    "swap_source_max_target_set_accuracy": 0.10,
    "erase_rule_balanced_max_target_set_accuracy": 0.55,
}


def summarize_v12_result(
    reverse: dict[str, float],
    cycle: dict[str, float],
    controls: dict[str, dict[str, float]],
    erase_rule: dict[str, dict[str, float]],
) -> dict[str, Any]:
    erase_rule_balanced = (
        erase_rule["reverse"]["target_set_accuracy"] + erase_rule["cycle"]["target_set_accuracy"]
    ) * 0.5
    checks = {
        "reverse_target_set_accuracy": reverse["target_set_accuracy"]
        >= V12_THRESHOLDS["reverse_target_set_accuracy"],
        "cycle_target_set_accuracy": cycle["target_set_accuracy"]
        >= V12_THRESHOLDS["cycle_target_set_accuracy"],
        "reverse_slot_accuracy": reverse["slot_accuracy"] >= V12_THRESHOLDS["reverse_slot_accuracy"],
        "cycle_slot_accuracy": cycle["slot_accuracy"] >= V12_THRESHOLDS["cycle_slot_accuracy"],
        "reverse_routed_slot_accuracy": reverse["routed_slot_accuracy"]
        >= V12_THRESHOLDS["reverse_routed_slot_accuracy"],
        "cycle_routed_slot_accuracy": cycle["routed_slot_accuracy"]
        >= V12_THRESHOLDS["cycle_routed_slot_accuracy"],
        "normal_control_target_set_accuracy": controls["normal"]["target_set_accuracy"]
        >= V12_THRESHOLDS["normal_control_target_set_accuracy"],
        "erase_source_target_set_accuracy": controls["erase_source"]["target_set_accuracy"]
        <= V12_THRESHOLDS["erase_source_max_target_set_accuracy"],
        "erase_sink_target_set_accuracy": controls["erase_sink"]["target_set_accuracy"]
        <= V12_THRESHOLDS["erase_sink_max_target_set_accuracy"],
        "swap_source_target_set_accuracy": controls["swap_source"]["target_set_accuracy"]
        <= V12_THRESHOLDS["swap_source_max_target_set_accuracy"],
        "erase_rule_balanced_target_set_accuracy": erase_rule_balanced
        <= V12_THRESHOLDS["erase_rule_balanced_max_target_set_accuracy"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": V12_THRESHOLDS,
        "erase_rule_balanced_target_set_accuracy": erase_rule_balanced,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.12 organ-first clean 3-pair benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--control-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=24000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v12.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)
    if checkpoint_args.get("update_rule") != "rank_slot_rule_cued":
        raise ValueError("v0.12 benchmark requires update_rule=rank_slot_rule_cued")

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 96))
    field_weight = float(checkpoint_args.get("field_weight", 0.5))
    localization_weight = float(checkpoint_args.get("localization_weight", 1.0))
    localization_margin = float(checkpoint_args.get("localization_margin", 1.0))
    activity_weight = float(checkpoint_args.get("activity_weight", 1e-3))

    common = {
        "batch_size": batch_size,
        "grid_size": grid_size,
        "rollout_steps": rollout_steps,
        "damage_prob": 0.0,
        "task": "multi",
        "coordinate_fields": True,
        "pair_count": 3,
        "min_pair_spacing": 1,
        "memory_input_steps": 4,
        "field_weight": field_weight,
        "localization_weight": localization_weight,
        "localization_margin": localization_margin,
        "activity_weight": activity_weight,
        "device": device,
    }

    reverse = evaluate_model(
        model,
        layout,
        batches=args.batches,
        sink_assignment="reverse",
        seed=args.seed,
        **common,
    )
    cycle = evaluate_model(
        model,
        layout,
        batches=args.batches,
        sink_assignment="cycle",
        seed=args.seed + 10_000,
        **common,
    )
    controls = {
        name: evaluate_control(
            model,
            layout,
            transform=transform,
            batches=args.control_batches,
            sink_assignment="reverse",
            seed=args.seed + 20_000 + index * 5_000,
            **common,
        )
        for index, (name, transform) in enumerate(CONTROLS.items())
    }
    erase_rule = {
        assignment: evaluate_control(
            model,
            layout,
            transform=erase_rule_cue,
            batches=args.batches,
            sink_assignment=assignment,
            seed=args.seed + 50_000 + index * 10_000,
            **common,
        )
        for index, assignment in enumerate(("reverse", "cycle"))
    }
    summary = summarize_v12_result(reverse, cycle, controls, erase_rule)
    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.12_organ_first_clean_three_pair",
        "config": {
            "batches": args.batches,
            "control_batches": args.control_batches,
            "batch_size": batch_size,
            "grid_size": grid_size,
            "rollout_steps": rollout_steps,
            **{key: value for key, value in common.items() if key != "device"},
        },
        "reverse": reverse,
        "cycle": cycle,
        "controls": controls,
        "erase_rule": erase_rule,
        "summary": summary,
    }
    save_json_report(Path(args.report), report)
    print(report)


if __name__ == "__main__":
    main()
