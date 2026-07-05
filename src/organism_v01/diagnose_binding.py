from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, save_json_report, set_seed
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import SINK_ASSIGNMENTS, generate_task_batch


def gather_pair_vectors(
    state: torch.Tensor,
    pair_rc: torch.Tensor,
    channel_slice: slice,
) -> torch.Tensor:
    vectors = []
    for item in range(state.shape[0]):
        item_vectors = []
        for pair_index in range(pair_rc.shape[1]):
            row, col = [int(value) for value in pair_rc[item, pair_index]]
            item_vectors.append(state[item, channel_slice, row, col])
        vectors.append(torch.stack(item_vectors))
    return torch.stack(vectors)


def mean_cosine_matrix(source_vectors: torch.Tensor, sink_vectors: torch.Tensor) -> list[list[float]]:
    source = F.normalize(source_vectors, dim=-1)
    sink = F.normalize(sink_vectors, dim=-1)
    cosine = torch.einsum("bpc,bqc->bpq", source, sink)
    return cosine.mean(dim=0).detach().cpu().tolist()


def paired_cosines(source_vectors: torch.Tensor, sink_vectors: torch.Tensor) -> list[float]:
    cosine = F.cosine_similarity(source_vectors, sink_vectors, dim=-1)
    return cosine.mean(dim=0).detach().cpu().tolist()


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[CellularOrganism, ChannelLayout, dict[str, Any]]:
    checkpoint = torch.load(Path(model_path), map_location=device, weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 8}))
    model = CellularOrganism(
        layout=layout,
        cell_hidden=int(checkpoint_args.get("cell_hidden", 32)),
        update_rule=str(checkpoint_args.get("update_rule", "standard")),
        message_slots=int(checkpoint_args.get("message_slots", 8)),
        tag_slots=int(checkpoint_args.get("tag_slots", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, layout, checkpoint_args


def diagnose_binding(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
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
        seed=seed,
        device=device,
    )
    if batch.pair_source_rc is None or batch.pair_sink_rc is None:
        raise ValueError("diagnostics require pair coordinates")

    with torch.no_grad():
        rollout = model(batch, steps=rollout_steps)
    hidden_source = gather_pair_vectors(rollout.final_state, batch.pair_source_rc, layout.hidden_slice)
    hidden_sink = gather_pair_vectors(rollout.final_state, batch.pair_sink_rc, layout.hidden_slice)
    report: dict[str, Any] = {
        "batch_size": batch_size,
        "grid_size": grid_size,
        "rollout_steps": rollout_steps,
        "damage_prob": damage_prob,
        "pair_count": pair_count,
        "min_pair_spacing": min_pair_spacing,
        "sink_assignment": sink_assignment,
        "seed": seed,
        "hidden_paired_cosines": paired_cosines(hidden_source, hidden_sink),
        "hidden_cosine_matrix": mean_cosine_matrix(hidden_source, hidden_sink),
    }
    rank_width = min(8, layout.hidden_channels)
    if rank_width:
        rank_slice = slice(layout.hidden_start, layout.hidden_start + rank_width)
        rank_source = gather_pair_vectors(rollout.final_state, batch.pair_source_rc, rank_slice)
        rank_sink = gather_pair_vectors(rollout.final_state, batch.pair_sink_rc, rank_slice)
        report["rank_paired_cosines"] = paired_cosines(rank_source, rank_sink)
        report["rank_cosine_matrix"] = mean_cosine_matrix(rank_source, rank_sink)
        report["rank_source_mean_abs"] = rank_source.abs().mean(dim=(0, 2)).detach().cpu().tolist()
        report["rank_sink_mean_abs"] = rank_sink.abs().mean(dim=(0, 2)).detach().cpu().tolist()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect source/sink hidden binding vectors.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--damage-prob", type=float, default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--min-pair-spacing", type=int, default=None)
    parser.add_argument("--sink-assignment", choices=SINK_ASSIGNMENTS, default=None)
    parser.add_argument("--seed", type=int, default=12700)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", default="outputs/reports/diagnose-binding.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)
    model, layout, checkpoint_args = load_model(args.model, device)

    report = diagnose_binding(
        model,
        layout,
        batch_size=args.batch_size or int(checkpoint_args.get("batch_size", 32)),
        grid_size=args.grid_size or int(checkpoint_args.get("grid_size", 16)),
        rollout_steps=args.rollout_steps or int(checkpoint_args.get("rollout_steps", 40)),
        damage_prob=args.damage_prob if args.damage_prob is not None else float(checkpoint_args.get("damage_prob", 0.1)),
        pair_count=args.pair_count if args.pair_count is not None else int(checkpoint_args.get("pair_count", 3)),
        min_pair_spacing=args.min_pair_spacing if args.min_pair_spacing is not None else int(checkpoint_args.get("min_pair_spacing", 1)),
        sink_assignment=args.sink_assignment or str(checkpoint_args.get("sink_assignment", "reverse")),
        seed=args.seed,
        device=device,
    )
    report["model"] = args.model
    save_json_report(args.report, report)
    print(report)


if __name__ == "__main__":
    main()
