from __future__ import annotations

import argparse
from pathlib import Path

from organism_v01.benchmark_v10 import load_model
from organism_v01.benchmark_v14 import _recovery_steps, summarize_v14_result
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import evaluate_dynamic_injury_recovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.15 compounded-damage 3-pair recovery benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--recovery-checkpoints", default=None)
    parser.add_argument("--damage-prob", type=float, default=0.10)
    parser.add_argument("--injury-prob", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=91400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v15-compounded-damage.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.damage_prob < 0.6:
        raise ValueError("--damage-prob must be in [0.0, 0.6)")
    if not 0.0 <= args.injury_prob < 0.8:
        raise ValueError("--injury-prob must be in [0.0, 0.8)")

    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)
    if checkpoint_args.get("update_rule") != "rank_slot_rule_cued":
        raise ValueError("v0.15 benchmark requires update_rule=rank_slot_rule_cued")
    if layout.rule_channels < 3:
        raise ValueError("v0.15 benchmark requires one-hot rule cue with rule_channels >= 3")

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 96))
    pre_steps = args.pre_steps if args.pre_steps is not None else max(1, rollout_steps // 2)
    post_steps = max(1, rollout_steps - pre_steps)
    recovery_steps = _recovery_steps(args.recovery_checkpoints, post_steps)
    field_weight = float(checkpoint_args.get("field_weight", 0.5))
    localization_weight = float(checkpoint_args.get("localization_weight", 1.0))
    localization_margin = float(checkpoint_args.get("localization_margin", 1.0))
    activity_weight = float(checkpoint_args.get("activity_weight", 1e-3))

    common = {
        "batch_size": batch_size,
        "grid_size": grid_size,
        "damage_prob": args.damage_prob,
        "task": "multi",
        "coordinate_fields": True,
        "pair_count": 3,
        "min_pair_spacing": 1,
        "memory_input_steps": 4,
        "device": device,
    }
    static_common = {
        **common,
        "rollout_steps": rollout_steps,
        "field_weight": field_weight,
        "localization_weight": localization_weight,
        "localization_margin": localization_margin,
        "activity_weight": activity_weight,
    }
    dynamic_common = {
        **common,
        "pre_steps": pre_steps,
        "recovery_steps": recovery_steps,
        "injury_prob": args.injury_prob,
    }

    static_reverse = evaluate_model(
        model,
        layout,
        batches=args.batches,
        sink_assignment="reverse",
        seed=args.seed,
        **static_common,
    )
    static_cycle = evaluate_model(
        model,
        layout,
        batches=args.batches,
        sink_assignment="cycle",
        seed=args.seed + 10_000,
        **static_common,
    )
    dynamic_reverse = evaluate_dynamic_injury_recovery(
        model,
        layout,
        batches=args.batches,
        sink_assignment="reverse",
        seed=args.seed + 20_000,
        **dynamic_common,
    )
    dynamic_cycle = evaluate_dynamic_injury_recovery(
        model,
        layout,
        batches=args.batches,
        sink_assignment="cycle",
        seed=args.seed + 30_000,
        **dynamic_common,
    )
    summary = summarize_v14_result(static_reverse, static_cycle, dynamic_reverse, dynamic_cycle)
    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.15_compounded_damage_three_pair_recovery",
        "config": {
            "batches": args.batches,
            "batch_size": batch_size,
            "grid_size": grid_size,
            "rollout_steps": rollout_steps,
            "pre_steps": pre_steps,
            "post_steps": post_steps,
            "recovery_steps": recovery_steps,
            "damage_prob": args.damage_prob,
            "injury_prob": args.injury_prob,
            "task": "multi",
            "coordinate_fields": True,
            "pair_count": 3,
            "min_pair_spacing": 1,
            "memory_input_steps": 4,
            "field_weight": field_weight,
            "localization_weight": localization_weight,
            "localization_margin": localization_margin,
            "activity_weight": activity_weight,
        },
        "static": {
            "reverse": static_reverse,
            "cycle": static_cycle,
        },
        "dynamic": {
            "reverse": dynamic_reverse,
            "cycle": dynamic_cycle,
        },
        "summary": summary,
    }
    save_json_report(Path(args.report), report)
    print(report)


if __name__ == "__main__":
    main()
