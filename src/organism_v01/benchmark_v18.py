from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from organism_v01.benchmark_v10 import load_model
from organism_v01.benchmark_v14 import _recovery_steps, _survival_ratio
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import evaluate_dynamic_injury_recovery


V18_THRESHOLDS = {
    "min_static_target_set_accuracy": 0.40,
    "min_dynamic_target_set_accuracy": 0.35,
    "min_dynamic_target_peak_accuracy": 0.65,
    "min_newly_blocked_fraction": 0.01,
}

ASSIGNMENTS = ("reverse", "cycle")
RANK_SLOT_CAPACITY = 3
SUPPORTED_UPDATE_RULES = {
    "rank_slot_rule_cued",
    "rank_slot_repair_rule_cued",
    "rank_slot_claim_rule_cued",
    "rank_slot_claim_residual_rule_cued",
    "rank_slot_claim_factor_rule_cued",
    "relative_rank_rule_cued",
}


def summarize_assignment(
    static_metrics: dict[str, float],
    dynamic_metrics: dict[str, Any],
) -> dict[str, Any]:
    final = dynamic_metrics["final"]
    newly_blocked = dynamic_metrics["injury"]["newly_blocked_fraction"]
    checks = {
        "static_target_set_accuracy": static_metrics["target_set_accuracy"]
        >= V18_THRESHOLDS["min_static_target_set_accuracy"],
        "dynamic_target_set_accuracy": final["target_set_accuracy"]
        >= V18_THRESHOLDS["min_dynamic_target_set_accuracy"],
        "dynamic_target_peak_accuracy": final["target_peak_accuracy"]
        >= V18_THRESHOLDS["min_dynamic_target_peak_accuracy"],
        "newly_blocked_fraction": newly_blocked >= V18_THRESHOLDS["min_newly_blocked_fraction"],
    }
    return {
        "passed_probe_gate": all(checks.values()),
        "checks": checks,
        "static_target_set_accuracy": static_metrics["target_set_accuracy"],
        "dynamic_target_set_accuracy": final["target_set_accuracy"],
        "dynamic_target_peak_accuracy": final["target_peak_accuracy"],
        "dynamic_routed_slot_accuracy": final["routed_slot_accuracy"],
        "survival_ratio": _survival_ratio(final, static_metrics),
        "recovery_target_set_delta": final["target_set_accuracy"]
        - dynamic_metrics["recovery"]["0"]["target_set_accuracy"],
        "newly_blocked_fraction": newly_blocked,
    }


def summarize_v18_result(
    results: dict[str, dict[str, dict[str, Any]]],
    *,
    pair_count: int,
) -> dict[str, Any]:
    assignment_summaries = {
        assignment: summarize_assignment(result["static"], result["dynamic"])
        for assignment, result in results.items()
    }
    dynamic_scores = [
        float(summary["dynamic_target_set_accuracy"]) for summary in assignment_summaries.values()
    ]
    static_scores = [
        float(summary["static_target_set_accuracy"]) for summary in assignment_summaries.values()
    ]
    routed_scores = [
        float(summary["dynamic_routed_slot_accuracy"]) for summary in assignment_summaries.values()
    ]
    passed = all(summary["passed_probe_gate"] for summary in assignment_summaries.values())
    rank_slot_supported = pair_count <= RANK_SLOT_CAPACITY

    return {
        "passed": passed,
        "passed_probe_gate": passed,
        "thresholds": V18_THRESHOLDS,
        "pair_count": pair_count,
        "rank_slot_capacity": RANK_SLOT_CAPACITY,
        "rank_slot_metrics_supported": rank_slot_supported,
        "fixed_rank_slot_capacity_exceeded": not rank_slot_supported,
        "assignments": assignment_summaries,
        "worst_static_target_set_accuracy": min(static_scores) if static_scores else 0.0,
        "worst_dynamic_target_set_accuracy": min(dynamic_scores) if dynamic_scores else 0.0,
        "mean_dynamic_target_set_accuracy": sum(dynamic_scores) / len(dynamic_scores)
        if dynamic_scores
        else 0.0,
        "mean_dynamic_routed_slot_accuracy": sum(routed_scores) / len(routed_scores)
        if routed_scores
        else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.18 four-pair dynamic-injury probe.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--pre-steps", type=int, default=None)
    parser.add_argument("--recovery-checkpoints", default=None)
    parser.add_argument("--damage-prob", type=float, default=0.10)
    parser.add_argument("--injury-prob", type=float, default=0.10)
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111800)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v18-four-pair.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    if args.pair_count <= 0:
        raise ValueError("--pair-count must be positive")
    if not 0.0 <= args.damage_prob < 0.6:
        raise ValueError("--damage-prob must be in [0.0, 0.6)")
    if not 0.0 <= args.injury_prob < 0.8:
        raise ValueError("--injury-prob must be in [0.0, 0.8)")

    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)
    if checkpoint_args.get("update_rule") not in SUPPORTED_UPDATE_RULES:
        raise ValueError("v0.18 benchmark requires a rule-cued rank update rule")
    if layout.rule_channels < 3:
        raise ValueError("v0.18 benchmark requires one-hot rule cue with rule_channels >= 3")

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    checkpoint_grid_size = int(checkpoint_args.get("grid_size", 14))
    checkpoint_rollout_steps = int(checkpoint_args.get("rollout_steps", 112))
    grid_size = args.grid_size or max(14, checkpoint_grid_size)
    rollout_steps = args.rollout_steps or max(112, checkpoint_rollout_steps)
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
        "pair_count": args.pair_count,
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

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for assignment_index, assignment in enumerate(ASSIGNMENTS):
        seed = args.seed + assignment_index * 20_000
        static_metrics = evaluate_model(
            model,
            layout,
            batches=args.batches,
            sink_assignment=assignment,
            seed=seed,
            **static_common,
        )
        dynamic_metrics = evaluate_dynamic_injury_recovery(
            model,
            layout,
            batches=args.batches,
            sink_assignment=assignment,
            seed=seed + 10_000,
            **dynamic_common,
        )
        results[assignment] = {
            "static": static_metrics,
            "dynamic": dynamic_metrics,
        }

    summary = summarize_v18_result(results, pair_count=args.pair_count)
    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.18_four_pair_dynamic_injury_probe",
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
            "pair_count": args.pair_count,
            "min_pair_spacing": 1,
            "memory_input_steps": 4,
            "field_weight": field_weight,
            "localization_weight": localization_weight,
            "localization_margin": localization_margin,
            "activity_weight": activity_weight,
        },
        "results": results,
        "summary": summary,
    }
    save_json_report(Path(args.report), report)
    print(report)


if __name__ == "__main__":
    main()
