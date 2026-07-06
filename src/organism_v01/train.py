from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.cell import UPDATE_RULES
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.injury import apply_random_injury, evaluate_dynamic_injury
from organism_v01.metrics import (
    binding_contrastive_loss,
    classification_accuracy,
    compute_loss,
    mean_sink_margin,
    rank_slot_accuracy,
    rank_slot_routed_accuracy,
    rank_slot_supervision_loss,
    target_set_accuracy,
)
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import SINK_ASSIGNMENTS, TASK_NAMES, RoutingBatch, generate_task_batch

CURRICULA = ("none", "multi_pair", "binding", "rule_binding", "rule_binding_damage", "rule_binding_final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train organism v0.1 on generated routing tasks.")
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--grid-size-choices", default=None)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--rollout-steps-choices", default=None)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--route-channels", type=int, default=0)
    parser.add_argument("--rule-channels", type=int, default=0)
    parser.add_argument("--cell-hidden", type=int, default=32)
    parser.add_argument("--update-rule", choices=UPDATE_RULES, default="standard")
    parser.add_argument("--message-slots", type=int, default=8)
    parser.add_argument("--tag-slots", type=int, default=4)
    parser.add_argument("--task", choices=TASK_NAMES, default="routing")
    parser.add_argument("--curriculum", choices=CURRICULA, default="none")
    parser.add_argument("--damage-prob", type=float, default=0.12)
    parser.add_argument("--coordinate-fields", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-count", type=int, default=3)
    parser.add_argument("--min-pair-spacing", type=int, default=1)
    parser.add_argument("--sink-assignment", choices=SINK_ASSIGNMENTS, default="aligned")
    parser.add_argument("--memory-input-steps", type=int, default=4)
    parser.add_argument("--field-weight", type=float, default=0.5)
    parser.add_argument("--localization-weight", type=float, default=1.0)
    parser.add_argument("--localization-margin", type=float, default=1.0)
    parser.add_argument("--activity-weight", type=float, default=1e-3)
    parser.add_argument("--binding-weight", type=float, default=0.0)
    parser.add_argument("--binding-temperature", type=float, default=0.2)
    parser.add_argument("--slot-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-injury-prob", type=float, default=0.0)
    parser.add_argument("--dynamic-injury-pre-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--save-model", default="outputs/models/organism-v01.pt")
    parser.add_argument("--report", default="outputs/reports/train-v01.json")
    return parser


@dataclass(frozen=True)
class TrainingRollout:
    final_state: torch.Tensor
    loss_batch: RoutingBatch
    activity_loss: torch.Tensor
    injury_applied: bool
    injury_prob: float
    pre_steps: int
    post_steps: int


def _parse_positive_int_choices(value: str | None, *, flag_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    choices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not choices:
        raise ValueError(f"{flag_name} must include at least one integer")
    if any(choice <= 0 for choice in choices):
        raise ValueError(f"{flag_name} values must be positive")
    return choices


def dynamic_injury_steps(args: argparse.Namespace, *, rollout_steps: int | None = None) -> tuple[int, int]:
    rollout_steps = int(args.rollout_steps if rollout_steps is None else rollout_steps)
    pre_steps = (
        int(args.dynamic_injury_pre_steps)
        if getattr(args, "dynamic_injury_pre_steps", None) is not None
        else max(1, rollout_steps // 2)
    )
    post_steps = rollout_steps - pre_steps
    if pre_steps <= 0:
        raise ValueError("--dynamic-injury-pre-steps must be positive")
    if post_steps <= 0:
        raise ValueError("--dynamic-injury-pre-steps must be less than --rollout-steps")
    return pre_steps, post_steps


def scale_training_params(args: argparse.Namespace, step: int) -> dict[str, int]:
    grid_choices = _parse_positive_int_choices(
        getattr(args, "grid_size_choices", None),
        flag_name="--grid-size-choices",
    ) or (int(args.grid_size),)
    rollout_choices = _parse_positive_int_choices(
        getattr(args, "rollout_steps_choices", None),
        flag_name="--rollout-steps-choices",
    ) or (int(args.rollout_steps),)
    if len(rollout_choices) not in {1, len(grid_choices)}:
        raise ValueError("--rollout-steps-choices must have length 1 or match --grid-size-choices")

    choice_index = ((step - 1) // 2) % len(grid_choices)
    rollout_index = 0 if len(rollout_choices) == 1 else choice_index
    rollout_steps = rollout_choices[rollout_index]
    pre_steps, post_steps = dynamic_injury_steps(args, rollout_steps=rollout_steps)
    return {
        "grid_size": grid_choices[choice_index],
        "rollout_steps": rollout_steps,
        "pre_steps": pre_steps,
        "post_steps": post_steps,
    }


def training_rollout(
    model: CellularOrganism,
    batch: RoutingBatch,
    layout: ChannelLayout,
    args: argparse.Namespace,
    *,
    step: int,
    device: torch.device,
    rollout_steps: int | None = None,
) -> TrainingRollout:
    rollout_steps = int(args.rollout_steps if rollout_steps is None else rollout_steps)
    injury_prob = float(getattr(args, "dynamic_injury_prob", 0.0))
    if injury_prob <= 0.0:
        rollout = model(batch, steps=rollout_steps)
        return TrainingRollout(
            final_state=rollout.final_state,
            loss_batch=batch,
            activity_loss=rollout.activity_loss,
            injury_applied=False,
            injury_prob=0.0,
            pre_steps=rollout_steps,
            post_steps=0,
        )

    if not 0.0 <= injury_prob < 0.8:
        raise ValueError("--dynamic-injury-prob must be in [0.0, 0.8)")

    pre_steps, post_steps = dynamic_injury_steps(args, rollout_steps=rollout_steps)
    before_injury = model(batch, steps=pre_steps)
    injured = apply_random_injury(
        batch,
        layout,
        injury_prob=injury_prob,
        seed=int(args.seed) + 200_000 + step,
    ).to(device)
    after_injury = model(
        injured,
        steps=post_steps,
        start_state=before_injury.final_state,
        start_step=pre_steps,
    )
    activity_loss = (
        before_injury.activity_loss * pre_steps + after_injury.activity_loss * post_steps
    ) / rollout_steps
    return TrainingRollout(
        final_state=after_injury.final_state,
        loss_batch=injured,
        activity_loss=activity_loss,
        injury_applied=True,
        injury_prob=injury_prob,
        pre_steps=pre_steps,
        post_steps=post_steps,
    )


def curriculum_batch_params(args: argparse.Namespace, step: int) -> dict[str, float | int | str | bool]:
    pair_count = args.pair_count
    damage_prob = args.damage_prob
    task = args.task

    sink_assignment = args.sink_assignment

    if args.curriculum in {"multi_pair", "binding", "rule_binding", "rule_binding_damage", "rule_binding_final"}:
        if args.task != "multi":
            raise ValueError(f"--curriculum {args.curriculum} requires --task multi")
        if args.curriculum in {"rule_binding", "rule_binding_damage", "rule_binding_final"} and int(getattr(args, "rule_channels", 0)) < 1:
            raise ValueError(f"--curriculum {args.curriculum} requires --rule-channels >= 1")
        progress = step / max(args.steps, 1)
        task = "multi"
        if args.curriculum == "rule_binding_final":
            pair_count = args.pair_count
            sink_assignment = "reverse" if step % 2 else "cycle"
            damage_prob = args.damage_prob
        elif args.curriculum == "multi_pair":
            if progress < 0.20:
                pair_count = 1
                damage_prob = 0.0
            elif progress < 0.45:
                pair_count = min(2, args.pair_count)
                damage_prob = 0.0
            elif progress < 0.70:
                pair_count = args.pair_count
                damage_prob = 0.0
            elif progress < 0.85:
                pair_count = args.pair_count
                damage_prob = args.damage_prob * 0.5
            else:
                pair_count = args.pair_count
                damage_prob = args.damage_prob
        elif args.curriculum == "binding":
            if progress < 0.15:
                pair_count = 1
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.30:
                pair_count = min(2, args.pair_count)
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.50:
                pair_count = min(2, args.pair_count)
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.70:
                pair_count = args.pair_count
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.85:
                pair_count = args.pair_count
                sink_assignment = "cycle"
                damage_prob = 0.0
            else:
                pair_count = args.pair_count
                sink_assignment = args.sink_assignment
                damage_prob = args.damage_prob
        elif args.curriculum == "rule_binding":
            if progress < 0.15:
                pair_count = 1
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.30:
                pair_count = min(2, args.pair_count)
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.45:
                pair_count = min(2, args.pair_count)
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.60:
                pair_count = args.pair_count
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.75:
                pair_count = args.pair_count
                sink_assignment = "cycle"
                damage_prob = 0.0
            else:
                pair_count = args.pair_count
                sink_assignment = "reverse" if step % 2 else "cycle"
                damage_prob = args.damage_prob
        else:
            if progress < 0.12:
                pair_count = 1
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.24:
                pair_count = min(2, args.pair_count)
                sink_assignment = "aligned"
                damage_prob = 0.0
            elif progress < 0.36:
                pair_count = min(2, args.pair_count)
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.50:
                pair_count = args.pair_count
                sink_assignment = "reverse"
                damage_prob = 0.0
            elif progress < 0.62:
                pair_count = args.pair_count
                sink_assignment = "cycle"
                damage_prob = 0.0
            elif progress < 0.75:
                pair_count = args.pair_count
                sink_assignment = "reverse" if step % 2 else "cycle"
                damage_prob = args.damage_prob * 0.25
            elif progress < 0.88:
                pair_count = args.pair_count
                sink_assignment = "reverse" if step % 2 else "cycle"
                damage_prob = args.damage_prob * 0.50
            else:
                pair_count = args.pair_count
                sink_assignment = "reverse" if step % 2 else "cycle"
                damage_prob = args.damage_prob

    return {
        "task": task,
        "damage_prob": damage_prob,
        "coordinate_fields": args.coordinate_fields,
        "pair_count": pair_count,
        "min_pair_spacing": args.min_pair_spacing,
        "sink_assignment": sink_assignment,
        "memory_input_steps": args.memory_input_steps,
    }


def checkpoint_payload(
    model: CellularOrganism,
    layout: ChannelLayout,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "layout": {
            "hidden_channels": layout.hidden_channels,
            "route_channels": layout.route_channels,
            "rule_channels": layout.rule_channels,
        },
        "args": vars(args),
        "metrics": metrics,
    }


def load_initial_model(
    model: CellularOrganism,
    *,
    init_model: str | None,
    device: torch.device,
    expected_hidden_channels: int,
    expected_route_channels: int,
    expected_cell_hidden: int,
    expected_update_rule: str,
    expected_message_slots: int,
    expected_tag_slots: int,
    expected_rule_channels: int = 0,
) -> None:
    if init_model is None:
        return

    checkpoint = torch.load(Path(init_model), map_location=device, weights_only=False)
    checkpoint_hidden_channels = int(checkpoint.get("layout", {}).get("hidden_channels", expected_hidden_channels))
    checkpoint_route_channels = int(checkpoint.get("layout", {}).get("route_channels", 0))
    checkpoint_rule_channels = int(checkpoint.get("layout", {}).get("rule_channels", 0))
    checkpoint_cell_hidden = int(checkpoint.get("args", {}).get("cell_hidden", expected_cell_hidden))
    checkpoint_update_rule = str(checkpoint.get("args", {}).get("update_rule", "standard"))
    checkpoint_message_slots = int(checkpoint.get("args", {}).get("message_slots", expected_message_slots))
    checkpoint_tag_slots = int(checkpoint.get("args", {}).get("tag_slots", expected_tag_slots))
    if checkpoint_hidden_channels != expected_hidden_channels:
        raise ValueError(
            f"init checkpoint hidden_channels={checkpoint_hidden_channels} "
            f"does not match requested {expected_hidden_channels}"
        )
    if checkpoint_cell_hidden != expected_cell_hidden:
        raise ValueError(
            f"init checkpoint cell_hidden={checkpoint_cell_hidden} "
            f"does not match requested {expected_cell_hidden}"
        )
    if checkpoint_route_channels != expected_route_channels:
        raise ValueError(
            f"init checkpoint route_channels={checkpoint_route_channels} "
            f"does not match requested {expected_route_channels}"
        )
    if checkpoint_rule_channels != expected_rule_channels:
        raise ValueError(
            f"init checkpoint rule_channels={checkpoint_rule_channels} "
            f"does not match requested {expected_rule_channels}"
        )
    if checkpoint_update_rule != expected_update_rule:
        raise ValueError(
            f"init checkpoint update_rule={checkpoint_update_rule} "
            f"does not match requested {expected_update_rule}"
        )
    if expected_update_rule == "gated_message" and checkpoint_message_slots != expected_message_slots:
        raise ValueError(
            f"init checkpoint message_slots={checkpoint_message_slots} "
            f"does not match requested {expected_message_slots}"
        )
    if expected_update_rule == "self_tagging" and checkpoint_tag_slots != expected_tag_slots:
        raise ValueError(
            f"init checkpoint tag_slots={checkpoint_tag_slots} "
            f"does not match requested {expected_tag_slots}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.dynamic_injury_prob < 0.8:
        raise ValueError("--dynamic-injury-prob must be in [0.0, 0.8)")
    if args.dynamic_injury_prob > 0.0:
        dynamic_pre_steps, dynamic_post_steps = dynamic_injury_steps(args)
    else:
        dynamic_pre_steps = args.rollout_steps
        dynamic_post_steps = 0
    scale_training_params(args, 1)

    device = choose_device(args.device)
    set_seed(args.seed)

    layout = ChannelLayout(
        hidden_channels=args.hidden_channels,
        route_channels=args.route_channels,
        rule_channels=args.rule_channels,
    )
    model = CellularOrganism(
        layout=layout,
        cell_hidden=args.cell_hidden,
        update_rule=args.update_rule,
        message_slots=args.message_slots,
        tag_slots=args.tag_slots,
    ).to(device)
    load_initial_model(
        model,
        init_model=args.init_model,
        device=device,
        expected_hidden_channels=args.hidden_channels,
        expected_route_channels=args.route_channels,
        expected_rule_channels=args.rule_channels,
        expected_cell_hidden=args.cell_hidden,
        expected_update_rule=args.update_rule,
        expected_message_slots=args.message_slots,
        expected_tag_slots=args.tag_slots,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    baseline_metrics = evaluate_model(
        model,
        layout,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        rollout_steps=args.rollout_steps,
        damage_prob=args.damage_prob,
        task=args.task,
        coordinate_fields=args.coordinate_fields,
        pair_count=args.pair_count,
        min_pair_spacing=args.min_pair_spacing,
        sink_assignment=args.sink_assignment,
        memory_input_steps=args.memory_input_steps,
        seed=args.seed + 10_000,
        device=device,
        field_weight=args.field_weight,
        localization_weight=args.localization_weight,
        localization_margin=args.localization_margin,
        activity_weight=args.activity_weight,
    )
    baseline_dynamic_metrics = None
    if args.dynamic_injury_prob > 0.0:
        baseline_dynamic_metrics = evaluate_dynamic_injury(
            model,
            layout,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            pre_steps=dynamic_pre_steps,
            post_steps=dynamic_post_steps,
            damage_prob=args.damage_prob,
            injury_prob=args.dynamic_injury_prob,
            task=args.task,
            coordinate_fields=args.coordinate_fields,
            pair_count=args.pair_count,
            min_pair_spacing=args.min_pair_spacing,
            sink_assignment=args.sink_assignment,
            memory_input_steps=args.memory_input_steps,
            seed=args.seed + 15_000,
            device=device,
            field_weight=args.field_weight,
            localization_weight=args.localization_weight,
            localization_margin=args.localization_margin,
            activity_weight=args.activity_weight,
        )

    history: list[dict[str, float | int | str]] = []
    for step in range(1, args.steps + 1):
        model.train()
        batch_params = curriculum_batch_params(args, step)
        scale_params = scale_training_params(args, step)
        batch = generate_task_batch(
            task=str(batch_params["task"]),
            batch_size=args.batch_size,
            grid_size=scale_params["grid_size"],
            layout=layout,
            damage_prob=float(batch_params["damage_prob"]),
            coordinate_fields=bool(batch_params["coordinate_fields"]),
            pair_count=int(batch_params["pair_count"]),
            min_pair_spacing=int(batch_params["min_pair_spacing"]),
            sink_assignment=str(batch_params["sink_assignment"]),
            memory_input_steps=int(batch_params["memory_input_steps"]),
            seed=args.seed + 100_000 + step,
            device=device,
        )
        rollout = training_rollout(
            model,
            batch,
            layout,
            args,
            step=step,
            device=device,
            rollout_steps=scale_params["rollout_steps"],
        )
        losses = compute_loss(
            rollout.final_state,
            rollout.loss_batch,
            layout,
            activity_loss=rollout.activity_loss,
            field_weight=args.field_weight,
            localization_weight=args.localization_weight,
            localization_margin=args.localization_margin,
            activity_weight=args.activity_weight,
        )
        if args.binding_weight:
            losses = dict(losses)
            binding_loss = binding_contrastive_loss(
                rollout.final_state,
                rollout.loss_batch,
                layout,
                temperature=args.binding_temperature,
            )
            losses["binding"] = binding_loss
            losses["total"] = losses["total"] + binding_loss * args.binding_weight
        if args.slot_weight:
            losses = dict(losses)
            slot_loss = rank_slot_supervision_loss(
                rollout.final_state,
                rollout.loss_batch,
                layout,
            )
            losses["slot"] = slot_loss
            losses["total"] = losses["total"] + slot_loss * args.slot_weight

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            accuracy = classification_accuracy(rollout.final_state.detach(), rollout.loss_batch, layout)
            set_accuracy = target_set_accuracy(rollout.final_state.detach(), rollout.loss_batch, layout)
            slot_accuracy = rank_slot_accuracy(rollout.final_state.detach(), rollout.loss_batch, layout)
            routed_slot_accuracy = rank_slot_routed_accuracy(rollout.final_state.detach(), rollout.loss_batch, layout)
            margin = mean_sink_margin(rollout.final_state.detach(), rollout.loss_batch, layout)
            row = {
                "step": step,
                "train_grid_size": scale_params["grid_size"],
                "train_rollout_steps": scale_params["rollout_steps"],
                "train_pair_count": int(batch_params["pair_count"]),
                "train_damage_prob": float(batch_params["damage_prob"]),
                "train_sink_assignment": str(batch_params["sink_assignment"]),
                "train_dynamic_injury": int(rollout.injury_applied),
                "train_injury_prob": rollout.injury_prob,
                "train_pre_steps": rollout.pre_steps,
                "train_post_steps": rollout.post_steps,
                "loss": float(losses["total"].item()),
                "task_loss": float(losses["task"].item()),
                "sink_loss": float(losses["sink"].item()),
                "quiet_loss": float(losses["quiet"].item()),
                "localization_loss": float(losses["localization"].item()),
                "binding_loss": float(losses.get("binding", losses["total"] * 0.0).item()),
                "slot_loss": float(losses.get("slot", losses["total"] * 0.0).item()),
                "accuracy": accuracy,
                "target_set_accuracy": set_accuracy,
                "slot_accuracy": slot_accuracy,
                "routed_slot_accuracy": routed_slot_accuracy,
                "sink_margin": margin,
            }
            history.append(row)
            print(row)

    trained_metrics = evaluate_model(
        model,
        layout,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        rollout_steps=args.rollout_steps,
        damage_prob=args.damage_prob,
        task=args.task,
        coordinate_fields=args.coordinate_fields,
        pair_count=args.pair_count,
        min_pair_spacing=args.min_pair_spacing,
        sink_assignment=args.sink_assignment,
        memory_input_steps=args.memory_input_steps,
        seed=args.seed + 20_000,
        device=device,
        field_weight=args.field_weight,
        localization_weight=args.localization_weight,
        localization_margin=args.localization_margin,
        activity_weight=args.activity_weight,
    )
    trained_dynamic_metrics = None
    if args.dynamic_injury_prob > 0.0:
        trained_dynamic_metrics = evaluate_dynamic_injury(
            model,
            layout,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            pre_steps=dynamic_pre_steps,
            post_steps=dynamic_post_steps,
            damage_prob=args.damage_prob,
            injury_prob=args.dynamic_injury_prob,
            task=args.task,
            coordinate_fields=args.coordinate_fields,
            pair_count=args.pair_count,
            min_pair_spacing=args.min_pair_spacing,
            sink_assignment=args.sink_assignment,
            memory_input_steps=args.memory_input_steps,
            seed=args.seed + 25_000,
            device=device,
            field_weight=args.field_weight,
            localization_weight=args.localization_weight,
            localization_margin=args.localization_margin,
            activity_weight=args.activity_weight,
        )

    report = {
        "version_goal": f"cellular organism generated {args.task} task",
        "device": str(device),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grid_size": args.grid_size,
            "grid_size_choices": args.grid_size_choices,
            "rollout_steps": args.rollout_steps,
            "rollout_steps_choices": args.rollout_steps_choices,
            "hidden_channels": args.hidden_channels,
            "route_channels": args.route_channels,
            "rule_channels": args.rule_channels,
            "cell_hidden": args.cell_hidden,
            "update_rule": args.update_rule,
            "message_slots": args.message_slots,
            "tag_slots": args.tag_slots,
            "task": args.task,
            "curriculum": args.curriculum,
            "damage_prob": args.damage_prob,
            "coordinate_fields": args.coordinate_fields,
            "pair_count": args.pair_count,
            "min_pair_spacing": args.min_pair_spacing,
            "sink_assignment": args.sink_assignment,
            "memory_input_steps": args.memory_input_steps,
            "field_weight": args.field_weight,
            "localization_weight": args.localization_weight,
            "localization_margin": args.localization_margin,
            "activity_weight": args.activity_weight,
            "binding_weight": args.binding_weight,
            "binding_temperature": args.binding_temperature,
            "slot_weight": args.slot_weight,
            "dynamic_injury_prob": args.dynamic_injury_prob,
            "dynamic_injury_pre_steps": args.dynamic_injury_pre_steps,
            "lr": args.lr,
            "seed": args.seed,
            "eval_batches": args.eval_batches,
            "init_model": args.init_model,
        },
        "baseline_untrained": baseline_metrics,
        "baseline_dynamic_injury": baseline_dynamic_metrics,
        "trained": trained_metrics,
        "trained_dynamic_injury": trained_dynamic_metrics,
        "improvement": {
            "accuracy": trained_metrics["accuracy"] - baseline_metrics["accuracy"],
            "target_set_accuracy": trained_metrics["target_set_accuracy"] - baseline_metrics["target_set_accuracy"],
            "loss": baseline_metrics["loss"] - trained_metrics["loss"],
            "sink_margin": trained_metrics["sink_margin"] - baseline_metrics["sink_margin"],
        },
        "history": history,
    }

    model_path = Path(args.save_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, layout, args, report), model_path)
    save_json_report(args.report, report)
    print(f"saved_model={model_path}")
    print(f"saved_report={args.report}")


if __name__ == "__main__":
    main()
