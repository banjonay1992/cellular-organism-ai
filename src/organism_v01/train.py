from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, evaluate_model, save_json_report, set_seed
from organism_v01.metrics import classification_accuracy, compute_loss, mean_sink_margin, target_set_accuracy
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import TASK_NAMES, generate_task_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train organism v0.1 on generated routing tasks.")
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--cell-hidden", type=int, default=32)
    parser.add_argument("--task", choices=TASK_NAMES, default="routing")
    parser.add_argument("--damage-prob", type=float, default=0.12)
    parser.add_argument("--coordinate-fields", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair-count", type=int, default=3)
    parser.add_argument("--memory-input-steps", type=int, default=4)
    parser.add_argument("--field-weight", type=float, default=0.5)
    parser.add_argument("--localization-weight", type=float, default=1.0)
    parser.add_argument("--localization-margin", type=float, default=1.0)
    parser.add_argument("--activity-weight", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-model", default="outputs/models/organism-v01.pt")
    parser.add_argument("--report", default="outputs/reports/train-v01.json")
    return parser


def checkpoint_payload(
    model: CellularOrganism,
    layout: ChannelLayout,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "layout": {"hidden_channels": layout.hidden_channels},
        "args": vars(args),
        "metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    device = choose_device(args.device)
    set_seed(args.seed)

    layout = ChannelLayout(hidden_channels=args.hidden_channels)
    model = CellularOrganism(layout=layout, cell_hidden=args.cell_hidden).to(device)
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
        memory_input_steps=args.memory_input_steps,
        seed=args.seed + 10_000,
        device=device,
        field_weight=args.field_weight,
        localization_weight=args.localization_weight,
        localization_margin=args.localization_margin,
        activity_weight=args.activity_weight,
    )

    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        model.train()
        batch = generate_task_batch(
            task=args.task,
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            layout=layout,
            damage_prob=args.damage_prob,
            coordinate_fields=args.coordinate_fields,
            pair_count=args.pair_count,
            memory_input_steps=args.memory_input_steps,
            seed=args.seed + 100_000 + step,
            device=device,
        )
        rollout = model(batch, steps=args.rollout_steps)
        losses = compute_loss(
            rollout.final_state,
            batch,
            layout,
            activity_loss=rollout.activity_loss,
            field_weight=args.field_weight,
            localization_weight=args.localization_weight,
            localization_margin=args.localization_margin,
            activity_weight=args.activity_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            accuracy = classification_accuracy(rollout.final_state.detach(), batch, layout)
            set_accuracy = target_set_accuracy(rollout.final_state.detach(), batch, layout)
            margin = mean_sink_margin(rollout.final_state.detach(), batch, layout)
            row = {
                "step": step,
                "loss": float(losses["total"].item()),
                "task_loss": float(losses["task"].item()),
                "sink_loss": float(losses["sink"].item()),
                "quiet_loss": float(losses["quiet"].item()),
                "localization_loss": float(losses["localization"].item()),
                "accuracy": accuracy,
                "target_set_accuracy": set_accuracy,
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
        memory_input_steps=args.memory_input_steps,
        seed=args.seed + 20_000,
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
            "rollout_steps": args.rollout_steps,
            "hidden_channels": args.hidden_channels,
            "cell_hidden": args.cell_hidden,
            "task": args.task,
            "damage_prob": args.damage_prob,
            "coordinate_fields": args.coordinate_fields,
            "pair_count": args.pair_count,
            "memory_input_steps": args.memory_input_steps,
            "field_weight": args.field_weight,
            "localization_weight": args.localization_weight,
            "localization_margin": args.localization_margin,
            "activity_weight": args.activity_weight,
            "lr": args.lr,
            "seed": args.seed,
            "eval_batches": args.eval_batches,
        },
        "baseline_untrained": baseline_metrics,
        "trained": trained_metrics,
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
