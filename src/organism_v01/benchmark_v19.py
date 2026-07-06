from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from organism_v01.benchmark_v10 import load_model
from organism_v01.benchmark_v14 import _recovery_steps
from organism_v01.evaluation import choose_device, save_json_report, set_seed
from organism_v01.injury import apply_random_injury
from organism_v01.organism import CellularOrganism, clamp_environment
from organism_v01.tasks import SINK_ASSIGNMENTS, RoutingBatch, generate_task_batch
from organism_v01.channels import ChannelLayout


DEFAULT_ASSIGNMENTS = ("aligned", "cycle", "reverse")
SUPPORTED_UPDATE_RULES = {"rank_slot_rule_cued", "rank_slot_repair_rule_cued", "relative_rank_rule_cued"}


def _rank_key(prefix: str, rank: int, pair_count: int) -> str:
    if rank == 0:
        return f"{prefix}_rank_{rank}_top"
    if rank == pair_count - 1:
        return f"{prefix}_rank_{rank}_bottom"
    return f"{prefix}_rank_{rank}"


def _empty_detail_totals(pair_count: int) -> dict[str, Any]:
    return {
        "item_count": 0,
        "sink_count": 0,
        "target_set_correct": 0,
        "correct_count_histogram": [0 for _ in range(pair_count + 1)],
        "source_rank_correct": [0 for _ in range(pair_count)],
        "source_rank_count": [0 for _ in range(pair_count)],
        "source_rank_margin_sum": [0.0 for _ in range(pair_count)],
        "sink_rank_correct": [0 for _ in range(pair_count)],
        "sink_rank_count": [0 for _ in range(pair_count)],
        "sink_rank_margin_sum": [0.0 for _ in range(pair_count)],
        "target_label_counts": [0, 0],
        "predicted_label_counts": [0, 0],
        "margin_sum": 0.0,
    }


def _accuracy_dict(correct: list[int], counts: list[int], prefix: str) -> dict[str, float]:
    pair_count = len(correct)
    return {
        _rank_key(prefix, rank, pair_count): correct[rank] / counts[rank] if counts[rank] else 0.0
        for rank in range(pair_count)
    }


def _mean_dict(sums: list[float], counts: list[int], prefix: str) -> dict[str, float]:
    pair_count = len(sums)
    return {
        _rank_key(prefix, rank, pair_count): sums[rank] / counts[rank] if counts[rank] else 0.0
        for rank in range(pair_count)
    }


def _label_fraction(counts: list[int]) -> dict[str, float]:
    total = sum(counts)
    return {str(index): value / total if total else 0.0 for index, value in enumerate(counts)}


def _finalize_detail_totals(totals: dict[str, Any]) -> dict[str, Any]:
    item_count = int(totals["item_count"])
    sink_count = int(totals["sink_count"])
    correct_hist = [int(value) for value in totals["correct_count_histogram"]]
    source_correct = [int(value) for value in totals["source_rank_correct"]]
    source_counts = [int(value) for value in totals["source_rank_count"]]
    sink_correct = [int(value) for value in totals["sink_rank_correct"]]
    sink_counts = [int(value) for value in totals["sink_rank_count"]]
    source_accuracy = _accuracy_dict(source_correct, source_counts, "source")
    sink_accuracy = _accuracy_dict(sink_correct, sink_counts, "sink")

    return {
        "item_count": item_count,
        "sink_count": sink_count,
        "target_set_accuracy": totals["target_set_correct"] / item_count if item_count else 0.0,
        "per_sink_accuracy": sum(source_correct) / sink_count if sink_count else 0.0,
        "correct_count_distribution": {
            str(correct_count): value / item_count if item_count else 0.0
            for correct_count, value in enumerate(correct_hist)
        },
        "source_rank_accuracy": source_accuracy,
        "source_rank_mean_margin": _mean_dict(
            [float(value) for value in totals["source_rank_margin_sum"]],
            source_counts,
            "source",
        ),
        "sink_rank_accuracy": sink_accuracy,
        "sink_rank_mean_margin": _mean_dict(
            [float(value) for value in totals["sink_rank_margin_sum"]],
            sink_counts,
            "sink",
        ),
        "mean_sink_margin": totals["margin_sum"] / sink_count if sink_count else 0.0,
        "target_label_fraction": _label_fraction([int(value) for value in totals["target_label_counts"]]),
        "predicted_label_fraction": _label_fraction(
            [int(value) for value in totals["predicted_label_counts"]]
        ),
    }


def add_state_pair_diagnostics(
    totals: dict[str, Any],
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> None:
    if batch.pair_labels is None or batch.pair_sink_rc is None:
        raise ValueError("pair diagnostics require generated multi-pair metadata")

    outputs = final_state[:, layout.output_slice]
    predictions = outputs.argmax(dim=1)
    pair_labels = batch.pair_labels
    pair_sink_rc = batch.pair_sink_rc
    pair_count = pair_labels.shape[1]

    for item in range(outputs.shape[0]):
        sink_rows = pair_sink_rc[item, :, 0]
        sink_order = torch.argsort(sink_rows)
        sink_rank_by_pair = torch.empty(pair_count, dtype=torch.long, device=sink_order.device)
        for sink_rank, pair_index_tensor in enumerate(sink_order):
            sink_rank_by_pair[int(pair_index_tensor.item())] = sink_rank

        correct_count = 0
        for pair_index in range(pair_count):
            row, col = [int(value) for value in pair_sink_rc[item, pair_index]]
            label = int(pair_labels[item, pair_index])
            prediction = int(predictions[item, row, col])
            correct = int(prediction == label)
            wrong_label = 1 - label
            margin = float(outputs[item, label, row, col] - outputs[item, wrong_label, row, col])
            sink_rank = int(sink_rank_by_pair[pair_index].item())

            correct_count += correct
            totals["sink_count"] += 1
            totals["source_rank_correct"][pair_index] += correct
            totals["source_rank_count"][pair_index] += 1
            totals["source_rank_margin_sum"][pair_index] += margin
            totals["sink_rank_correct"][sink_rank] += correct
            totals["sink_rank_count"][sink_rank] += 1
            totals["sink_rank_margin_sum"][sink_rank] += margin
            totals["target_label_counts"][label] += 1
            totals["predicted_label_counts"][prediction] += 1
            totals["margin_sum"] += margin

        totals["item_count"] += 1
        totals["target_set_correct"] += int(correct_count == pair_count)
        totals["correct_count_histogram"][correct_count] += 1


def state_pair_diagnostics(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> dict[str, Any]:
    if batch.pair_labels is None:
        raise ValueError("pair diagnostics require generated multi-pair metadata")
    totals = _empty_detail_totals(batch.pair_labels.shape[1])
    add_state_pair_diagnostics(totals, final_state, batch, layout)
    return _finalize_detail_totals(totals)


def _injury_extent(before: RoutingBatch, after: RoutingBatch, layout: ChannelLayout) -> dict[str, float]:
    before_blocked = before.env[:, layout.blocked].bool()
    after_blocked = after.env[:, layout.blocked].bool()
    newly_blocked = after_blocked & ~before_blocked
    return {
        "newly_blocked_fraction": float(newly_blocked.float().mean().item()),
        "blocked_fraction_before": float(before_blocked.float().mean().item()),
        "blocked_fraction_after": float(after_blocked.float().mean().item()),
    }


def _average_extents(totals: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / batches for key, value in totals.items()}


def diagnose_static_assignment(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
    batches: int,
    batch_size: int,
    grid_size: int,
    rollout_steps: int,
    damage_prob: float,
    pair_count: int,
    min_pair_spacing: int,
    sink_assignment: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = _empty_detail_totals(pair_count)
    with torch.no_grad():
        for index in range(batches):
            batch = generate_task_batch(
                task="multi",
                batch_size=batch_size,
                grid_size=grid_size,
                layout=layout,
                damage_prob=damage_prob,
                coordinate_fields=True,
                pair_count=pair_count,
                min_pair_spacing=min_pair_spacing,
                sink_assignment=sink_assignment,
                memory_input_steps=4,
                seed=seed + index,
                device=device,
            )
            rollout = model(batch, steps=rollout_steps)
            add_state_pair_diagnostics(totals, rollout.final_state, batch, layout)
    return _finalize_detail_totals(totals)


def diagnose_dynamic_assignment(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
    batches: int,
    batch_size: int,
    grid_size: int,
    pre_steps: int,
    recovery_steps: tuple[int, ...],
    damage_prob: float,
    injury_prob: float,
    pair_count: int,
    min_pair_spacing: int,
    sink_assignment: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if batches <= 0:
        raise ValueError("batches must be positive")
    if pre_steps <= 0:
        raise ValueError("pre_steps must be positive")
    if not recovery_steps:
        raise ValueError("recovery_steps must not be empty")
    ordered_steps = tuple(sorted(set(recovery_steps)))
    if ordered_steps[0] < 0:
        raise ValueError("recovery_steps must be non-negative")

    model.eval()
    pre_totals = _empty_detail_totals(pair_count)
    recovery_totals = {step: _empty_detail_totals(pair_count) for step in ordered_steps}
    extent_totals = {
        "newly_blocked_fraction": 0.0,
        "blocked_fraction_before": 0.0,
        "blocked_fraction_after": 0.0,
    }

    with torch.no_grad():
        for index in range(batches):
            batch = generate_task_batch(
                task="multi",
                batch_size=batch_size,
                grid_size=grid_size,
                layout=layout,
                damage_prob=damage_prob,
                coordinate_fields=True,
                pair_count=pair_count,
                min_pair_spacing=min_pair_spacing,
                sink_assignment=sink_assignment,
                memory_input_steps=4,
                seed=seed + index,
                device=device,
            )
            before_injury = model(batch, steps=pre_steps)
            add_state_pair_diagnostics(pre_totals, before_injury.final_state, batch, layout)

            injured = apply_random_injury(
                batch,
                layout,
                injury_prob=injury_prob,
                seed=seed + 50_000 + index,
            ).to(device)
            for key, value in _injury_extent(batch, injured, layout).items():
                extent_totals[key] += value

            current_state = clamp_environment(
                before_injury.final_state,
                injured,
                layout,
                step_index=pre_steps,
            )
            previous_step = 0
            for step in ordered_steps:
                if step > previous_step:
                    recovered = model(
                        injured,
                        steps=step - previous_step,
                        start_state=current_state,
                        start_step=pre_steps + previous_step,
                    )
                    current_state = recovered.final_state
                    previous_step = step
                add_state_pair_diagnostics(recovery_totals[step], current_state, injured, layout)

    finalized_recovery = {
        str(step): _finalize_detail_totals(totals) for step, totals in recovery_totals.items()
    }
    return {
        "pre_injury": _finalize_detail_totals(pre_totals),
        "recovery": finalized_recovery,
        "final": finalized_recovery[str(ordered_steps[-1])],
        "injury": _average_extents(extent_totals, batches),
    }


def _weakest_rank(accuracies: dict[str, float]) -> dict[str, float | str]:
    if not accuracies:
        return {"rank": "", "accuracy": 0.0}
    rank, value = min(accuracies.items(), key=lambda item: item[1])
    return {"rank": rank, "accuracy": value}


def _rank_spread(accuracies: dict[str, float]) -> float:
    if not accuracies:
        return 0.0
    values = list(accuracies.values())
    return max(values) - min(values)


def summarize_v19_result(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    assignments: dict[str, dict[str, Any]] = {}
    for assignment, result in results.items():
        final = result["dynamic"]["final"]
        immediate = result["dynamic"]["recovery"]["0"]
        assignments[assignment] = {
            "static_target_set_accuracy": result["static"]["target_set_accuracy"],
            "dynamic_target_set_accuracy": final["target_set_accuracy"],
            "dynamic_per_sink_accuracy": final["per_sink_accuracy"],
            "dynamic_correct_count_distribution": final["correct_count_distribution"],
            "dynamic_source_rank_accuracy": final["source_rank_accuracy"],
            "dynamic_sink_rank_accuracy": final["sink_rank_accuracy"],
            "weakest_source_rank": _weakest_rank(final["source_rank_accuracy"]),
            "weakest_sink_rank": _weakest_rank(final["sink_rank_accuracy"]),
            "source_rank_accuracy_spread": _rank_spread(final["source_rank_accuracy"]),
            "sink_rank_accuracy_spread": _rank_spread(final["sink_rank_accuracy"]),
            "recovery_target_set_delta": final["target_set_accuracy"]
            - immediate["target_set_accuracy"],
            "newly_blocked_fraction": result["dynamic"]["injury"]["newly_blocked_fraction"],
        }

    best_assignment = {
        "assignment": max(assignments.items(), key=lambda item: item[1]["dynamic_target_set_accuracy"])[0]
        if assignments
        else "",
        "dynamic_target_set_accuracy": max(
            (summary["dynamic_target_set_accuracy"] for summary in assignments.values()),
            default=0.0,
        ),
    }
    reverse_cycle_gap = 0.0
    if "cycle" in assignments and "reverse" in assignments:
        reverse_cycle_gap = (
            assignments["cycle"]["dynamic_target_set_accuracy"]
            - assignments["reverse"]["dynamic_target_set_accuracy"]
        )

    return {
        "diagnostic_only": True,
        "assignments": assignments,
        "best_assignment": best_assignment,
        "cycle_minus_reverse_dynamic_target_set_accuracy": reverse_cycle_gap,
    }


def parse_assignments(text: str) -> tuple[str, ...]:
    assignments = tuple(part.strip() for part in text.split(",") if part.strip())
    if not assignments:
        raise ValueError("--assignments must include at least one assignment")
    unknown = [assignment for assignment in assignments if assignment not in SINK_ASSIGNMENTS]
    if unknown:
        raise ValueError(f"unknown assignment(s): {', '.join(unknown)}")
    return assignments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the v0.19 four-pair rank diagnostic.")
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
    parser.add_argument("--assignments", default=",".join(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--seed", type=int, default=111900)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/benchmark-v19-four-pair-diagnostics.json")
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
    assignments = parse_assignments(args.assignments)

    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)
    if checkpoint_args.get("update_rule") not in SUPPORTED_UPDATE_RULES:
        raise ValueError("v0.19 benchmark requires a rule-cued rank update rule")
    if layout.rule_channels < 3:
        raise ValueError("v0.19 benchmark requires one-hot rule cue with rule_channels >= 3")

    batch_size = args.batch_size or int(checkpoint_args.get("batch_size", 32))
    checkpoint_grid_size = int(checkpoint_args.get("grid_size", 14))
    checkpoint_rollout_steps = int(checkpoint_args.get("rollout_steps", 112))
    grid_size = args.grid_size or max(14, checkpoint_grid_size)
    rollout_steps = args.rollout_steps or max(112, checkpoint_rollout_steps)
    pre_steps = args.pre_steps if args.pre_steps is not None else max(1, rollout_steps // 2)
    post_steps = max(1, rollout_steps - pre_steps)
    recovery_steps = _recovery_steps(args.recovery_checkpoints, post_steps)

    results: dict[str, dict[str, Any]] = {}
    for assignment_index, assignment in enumerate(assignments):
        seed = args.seed + assignment_index * 20_000
        static = diagnose_static_assignment(
            model,
            layout,
            batches=args.batches,
            batch_size=batch_size,
            grid_size=grid_size,
            rollout_steps=rollout_steps,
            damage_prob=args.damage_prob,
            pair_count=args.pair_count,
            min_pair_spacing=1,
            sink_assignment=assignment,
            seed=seed,
            device=device,
        )
        dynamic = diagnose_dynamic_assignment(
            model,
            layout,
            batches=args.batches,
            batch_size=batch_size,
            grid_size=grid_size,
            pre_steps=pre_steps,
            recovery_steps=recovery_steps,
            damage_prob=args.damage_prob,
            injury_prob=args.injury_prob,
            pair_count=args.pair_count,
            min_pair_spacing=1,
            sink_assignment=assignment,
            seed=seed + 10_000,
            device=device,
        )
        results[assignment] = {
            "static": static,
            "dynamic": dynamic,
        }

    report = {
        "model": args.model,
        "seed": args.seed,
        "benchmark": "v0.19_four_pair_rank_diagnostics",
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
            "pair_count": args.pair_count,
            "assignments": assignments,
            "min_pair_spacing": 1,
        },
        "results": results,
        "summary": summarize_v19_result(results),
    }
    save_json_report(Path(args.report), report)
    print(report)


if __name__ == "__main__":
    main()
