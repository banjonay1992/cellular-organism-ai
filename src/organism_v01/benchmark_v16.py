from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from organism_v01.benchmark_v10 import load_model
from organism_v01.benchmark_v14 import _recovery_steps, _survival_ratio
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import evaluate_dynamic_injury_recovery


V16_THRESHOLDS = {
    "min_dynamic_target_set_accuracy": 0.50,
    "min_dynamic_target_peak_accuracy": 0.75,
    "min_dynamic_routed_slot_accuracy": 0.70,
    "min_newly_blocked_fraction": 0.01,
}


@dataclass(frozen=True)
class GeneralizationScenario:
    name: str
    grid_size: int
    rollout_steps: int
    pre_steps: int
    damage_prob: float
    injury_prob: float
    seed_offset: int


def default_scenarios(*, grid_size: int, rollout_steps: int) -> dict[str, GeneralizationScenario]:
    half = max(1, rollout_steps // 2)
    early = max(1, rollout_steps // 4)
    late = max(1, (rollout_steps * 3) // 4)
    larger_rollout = rollout_steps + 16
    return {
        "baseline": GeneralizationScenario(
            name="baseline",
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            pre_steps=half,
            damage_prob=0.10,
            injury_prob=0.10,
            seed_offset=0,
        ),
        "early_injury": GeneralizationScenario(
            name="early_injury",
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            pre_steps=early,
            damage_prob=0.10,
            injury_prob=0.10,
            seed_offset=100_000,
        ),
        "late_injury": GeneralizationScenario(
            name="late_injury",
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            pre_steps=late,
            damage_prob=0.10,
            injury_prob=0.10,
            seed_offset=200_000,
        ),
        "mild_damage": GeneralizationScenario(
            name="mild_damage",
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            pre_steps=half,
            damage_prob=0.05,
            injury_prob=0.05,
            seed_offset=300_000,
        ),
        "higher_injury": GeneralizationScenario(
            name="higher_injury",
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            pre_steps=half,
            damage_prob=0.10,
            injury_prob=0.15,
            seed_offset=400_000,
        ),
        "larger_grid": GeneralizationScenario(
            name="larger_grid",
            grid_size=grid_size + 2,
            rollout_steps=larger_rollout,
            pre_steps=max(1, larger_rollout // 2),
            damage_prob=0.10,
            injury_prob=0.10,
            seed_offset=500_000,
        ),
    }


def select_scenarios(
    requested: str | None,
    *,
    grid_size: int,
    rollout_steps: int,
) -> list[GeneralizationScenario]:
    scenarios = default_scenarios(grid_size=grid_size, rollout_steps=rollout_steps)
    if requested is None or requested == "all":
        return list(scenarios.values())
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in scenarios]
    if unknown:
        raise ValueError(f"unknown v0.16 scenario(s): {', '.join(unknown)}")
    return [scenarios[name] for name in names]


def summarize_assignment(static_metrics: dict[str, float], dynamic_metrics: dict[str, Any]) -> dict[str, Any]:
    final = dynamic_metrics["final"]
    checks = {
        "dynamic_target_set_accuracy": final["target_set_accuracy"]
        >= V16_THRESHOLDS["min_dynamic_target_set_accuracy"],
        "dynamic_target_peak_accuracy": final["target_peak_accuracy"]
        >= V16_THRESHOLDS["min_dynamic_target_peak_accuracy"],
        "dynamic_routed_slot_accuracy": final["routed_slot_accuracy"]
        >= V16_THRESHOLDS["min_dynamic_routed_slot_accuracy"],
        "newly_blocked_fraction": dynamic_metrics["injury"]["newly_blocked_fraction"]
        >= V16_THRESHOLDS["min_newly_blocked_fraction"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "static_target_set_accuracy": static_metrics["target_set_accuracy"],
        "dynamic_target_set_accuracy": final["target_set_accuracy"],
        "dynamic_target_peak_accuracy": final["target_peak_accuracy"],
        "dynamic_routed_slot_accuracy": final["routed_slot_accuracy"],
        "survival_ratio": _survival_ratio(final, static_metrics),
        "recovery_target_set_delta": final["target_set_accuracy"]
        - dynamic_metrics["recovery"]["0"]["target_set_accuracy"],
        "newly_blocked_fraction": dynamic_metrics["injury"]["newly_blocked_fraction"],
    }


def summarize_v16_result(scenarios: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    scenario_summaries: dict[str, dict[str, Any]] = {}
    dynamic_scores: list[float] = []
    routed_scores: list[float] = []
    survival_ratios: list[float] = []
    for scenario_name, scenario_results in scenarios.items():
        assignment_summaries = {
            assignment: summarize_assignment(
                result["static"],
                result["dynamic"],
            )
            for assignment, result in scenario_results.items()
        }
        scenario_passed = all(summary["passed"] for summary in assignment_summaries.values())
        scenario_dynamic_scores = [
            float(summary["dynamic_target_set_accuracy"]) for summary in assignment_summaries.values()
        ]
        scenario_routed_scores = [
            float(summary["dynamic_routed_slot_accuracy"]) for summary in assignment_summaries.values()
        ]
        dynamic_scores.extend(scenario_dynamic_scores)
        routed_scores.extend(scenario_routed_scores)
        survival_ratios.extend(float(summary["survival_ratio"]) for summary in assignment_summaries.values())
        scenario_summaries[scenario_name] = {
            "passed": scenario_passed,
            "assignments": assignment_summaries,
            "min_dynamic_target_set_accuracy": min(scenario_dynamic_scores),
            "min_dynamic_routed_slot_accuracy": min(scenario_routed_scores),
        }

    passed_scenarios = sum(1 for scenario in scenario_summaries.values() if scenario["passed"])
    return {
        "passed": passed_scenarios == len(scenario_summaries),
        "thresholds": V16_THRESHOLDS,
        "scenario_count": len(scenario_summaries),
        "passed_scenario_count": passed_scenarios,
        "worst_dynamic_target_set_accuracy": min(dynamic_scores) if dynamic_scores else 0.0,
        "mean_dynamic_target_set_accuracy": sum(dynamic_scores) / len(dynamic_scores) if dynamic_scores else 0.0,
        "worst_dynamic_routed_slot_accuracy": min(routed_scores) if routed_scores else 0.0,
        "mean_survival_ratio": sum(survival_ratios) / len(survival_ratios) if survival_ratios else 0.0,
        "scenarios": scenario_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.16 dynamic-injury generalization benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--recovery-checkpoints", default=None)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--seed", type=int, default=101600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v16-generalization.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)
    if checkpoint_args.get("update_rule") != "rank_slot_rule_cued":
        raise ValueError("v0.16 benchmark requires update_rule=rank_slot_rule_cued")
    if layout.rule_channels < 3:
        raise ValueError("v0.16 benchmark requires one-hot rule cue with rule_channels >= 3")

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 96))
    scenarios = select_scenarios(args.scenarios, grid_size=grid_size, rollout_steps=rollout_steps)
    field_weight = float(checkpoint_args.get("field_weight", 0.5))
    localization_weight = float(checkpoint_args.get("localization_weight", 1.0))
    localization_margin = float(checkpoint_args.get("localization_margin", 1.0))
    activity_weight = float(checkpoint_args.get("activity_weight", 1e-3))

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario in scenarios:
        post_steps = max(1, scenario.rollout_steps - scenario.pre_steps)
        recovery_steps = _recovery_steps(args.recovery_checkpoints, post_steps)
        scenario_results: dict[str, dict[str, Any]] = {}
        for assignment_index, assignment in enumerate(("reverse", "cycle")):
            seed = args.seed + scenario.seed_offset + assignment_index * 10_000
            common = {
                "batch_size": batch_size,
                "grid_size": scenario.grid_size,
                "damage_prob": scenario.damage_prob,
                "task": "multi",
                "coordinate_fields": True,
                "pair_count": 3,
                "min_pair_spacing": 1,
                "memory_input_steps": 4,
                "sink_assignment": assignment,
                "device": device,
            }
            static_metrics = evaluate_model(
                model,
                layout,
                batches=args.batches,
                rollout_steps=scenario.rollout_steps,
                seed=seed,
                field_weight=field_weight,
                localization_weight=localization_weight,
                localization_margin=localization_margin,
                activity_weight=activity_weight,
                **common,
            )
            dynamic_metrics = evaluate_dynamic_injury_recovery(
                model,
                layout,
                batches=args.batches,
                pre_steps=scenario.pre_steps,
                recovery_steps=recovery_steps,
                injury_prob=scenario.injury_prob,
                seed=seed + 50_000,
                **common,
            )
            scenario_results[assignment] = {
                "static": static_metrics,
                "dynamic": dynamic_metrics,
            }
        results[scenario.name] = scenario_results

    summary = summarize_v16_result(results)
    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.16_dynamic_injury_generalization",
        "config": {
            "batches": args.batches,
            "batch_size": batch_size,
            "base_grid_size": grid_size,
            "base_rollout_steps": rollout_steps,
            "scenarios": [asdict(scenario) for scenario in scenarios],
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
