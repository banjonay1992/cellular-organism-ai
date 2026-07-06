from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from organism_v01.benchmark_v10 import load_model
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import evaluate_dynamic_injury_recovery


V14_THRESHOLDS = {
    "reverse_static_target_set_accuracy": 0.55,
    "cycle_static_target_set_accuracy": 0.45,
    "reverse_dynamic_target_set_accuracy": 0.50,
    "cycle_dynamic_target_set_accuracy": 0.50,
    "reverse_dynamic_target_peak_accuracy": 0.75,
    "cycle_dynamic_target_peak_accuracy": 0.75,
    "reverse_dynamic_routed_slot_accuracy": 0.75,
    "cycle_dynamic_routed_slot_accuracy": 0.75,
    "reverse_survival_ratio": 0.80,
    "cycle_survival_ratio": 0.80,
    "min_newly_blocked_fraction": 0.03,
}


def _survival_ratio(dynamic_final: dict[str, float], static_metrics: dict[str, float]) -> float:
    static_score = max(static_metrics["target_set_accuracy"], 1e-6)
    return dynamic_final["target_set_accuracy"] / static_score


def _recovery_delta(dynamic: dict[str, Any], metric: str) -> float:
    immediate = dynamic["recovery"]["0"][metric]
    return dynamic["final"][metric] - immediate


def summarize_v14_result(
    static_reverse: dict[str, float],
    static_cycle: dict[str, float],
    dynamic_reverse: dict[str, Any],
    dynamic_cycle: dict[str, Any],
) -> dict[str, Any]:
    reverse_final = dynamic_reverse["final"]
    cycle_final = dynamic_cycle["final"]
    reverse_survival = _survival_ratio(reverse_final, static_reverse)
    cycle_survival = _survival_ratio(cycle_final, static_cycle)
    reverse_new_damage = dynamic_reverse["injury"]["newly_blocked_fraction"]
    cycle_new_damage = dynamic_cycle["injury"]["newly_blocked_fraction"]

    checks = {
        "reverse_static_target_set_accuracy": static_reverse["target_set_accuracy"]
        >= V14_THRESHOLDS["reverse_static_target_set_accuracy"],
        "cycle_static_target_set_accuracy": static_cycle["target_set_accuracy"]
        >= V14_THRESHOLDS["cycle_static_target_set_accuracy"],
        "reverse_dynamic_target_set_accuracy": reverse_final["target_set_accuracy"]
        >= V14_THRESHOLDS["reverse_dynamic_target_set_accuracy"],
        "cycle_dynamic_target_set_accuracy": cycle_final["target_set_accuracy"]
        >= V14_THRESHOLDS["cycle_dynamic_target_set_accuracy"],
        "reverse_dynamic_target_peak_accuracy": reverse_final["target_peak_accuracy"]
        >= V14_THRESHOLDS["reverse_dynamic_target_peak_accuracy"],
        "cycle_dynamic_target_peak_accuracy": cycle_final["target_peak_accuracy"]
        >= V14_THRESHOLDS["cycle_dynamic_target_peak_accuracy"],
        "reverse_dynamic_routed_slot_accuracy": reverse_final["routed_slot_accuracy"]
        >= V14_THRESHOLDS["reverse_dynamic_routed_slot_accuracy"],
        "cycle_dynamic_routed_slot_accuracy": cycle_final["routed_slot_accuracy"]
        >= V14_THRESHOLDS["cycle_dynamic_routed_slot_accuracy"],
        "reverse_survival_ratio": reverse_survival >= V14_THRESHOLDS["reverse_survival_ratio"],
        "cycle_survival_ratio": cycle_survival >= V14_THRESHOLDS["cycle_survival_ratio"],
        "reverse_newly_blocked_fraction": reverse_new_damage >= V14_THRESHOLDS["min_newly_blocked_fraction"],
        "cycle_newly_blocked_fraction": cycle_new_damage >= V14_THRESHOLDS["min_newly_blocked_fraction"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": V14_THRESHOLDS,
        "reverse_survival_ratio": reverse_survival,
        "cycle_survival_ratio": cycle_survival,
        "reverse_recovery_target_set_delta": _recovery_delta(dynamic_reverse, "target_set_accuracy"),
        "cycle_recovery_target_set_delta": _recovery_delta(dynamic_cycle, "target_set_accuracy"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.14 dynamic-injury 3-pair recovery benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--recovery-checkpoints", default=None)
    parser.add_argument("--damage-prob", type=float, default=0.05)
    parser.add_argument("--injury-prob", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=71400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v14.json")
    return parser


def _recovery_steps(text: str | None, post_steps: int) -> tuple[int, ...]:
    if post_steps <= 0:
        raise ValueError("post_steps must be positive")
    if text is None:
        return tuple(sorted({0, max(1, post_steps // 4), max(1, post_steps // 2), post_steps}))
    steps = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    if not steps:
        raise ValueError("--recovery-checkpoints must include at least one integer")
    if steps[0] < 0:
        raise ValueError("--recovery-checkpoints must be non-negative")
    if steps[-1] > post_steps:
        raise ValueError("--recovery-checkpoints cannot exceed post_steps")
    if 0 not in steps:
        steps = (0, *steps)
    if post_steps not in steps:
        steps = tuple(sorted((*steps, post_steps)))
    return steps


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
        raise ValueError("v0.14 benchmark requires update_rule=rank_slot_rule_cued")
    if layout.rule_channels < 3:
        raise ValueError("v0.14 benchmark requires one-hot rule cue with rule_channels >= 3")

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
        "benchmark": "v0.14_dynamic_injury_three_pair_recovery",
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
