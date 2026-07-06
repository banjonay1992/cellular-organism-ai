from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from organism_v01.channels import ChannelLayout
from organism_v01.metrics import (
    classification_accuracy,
    compute_loss,
    mean_sink_margin,
    output_localization,
    rank_slot_accuracy,
    target_peak_accuracy,
    target_set_accuracy,
)
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import generate_task_batch


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: CellularOrganism,
    layout: ChannelLayout,
    *,
    batches: int,
    batch_size: int,
    grid_size: int,
    rollout_steps: int,
    damage_prob: float,
    seed: int,
    device: torch.device,
    task: str = "routing",
    coordinate_fields: bool = True,
    pair_count: int = 3,
    min_pair_spacing: int = 1,
    sink_assignment: str = "aligned",
    memory_input_steps: int = 4,
    field_weight: float = 0.5,
    localization_weight: float = 1.0,
    localization_margin: float = 1.0,
    activity_weight: float = 1e-3,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "accuracy": 0.0,
        "target_peak_accuracy": 0.0,
        "target_set_accuracy": 0.0,
        "slot_accuracy": 0.0,
        "sink_margin": 0.0,
        "localization": 0.0,
    }
    with torch.no_grad():
        for index in range(batches):
            batch = generate_task_batch(
                task=task,
                batch_size=batch_size,
                grid_size=grid_size,
                layout=layout,
                damage_prob=damage_prob,
                coordinate_fields=coordinate_fields,
                pair_count=pair_count,
                min_pair_spacing=min_pair_spacing,
                sink_assignment=sink_assignment,
                memory_input_steps=memory_input_steps,
                seed=seed + index,
                device=device,
            )
            rollout = model(batch, steps=rollout_steps)
            losses = compute_loss(
                rollout.final_state,
                batch,
                layout,
                activity_loss=rollout.activity_loss,
                field_weight=field_weight,
                localization_weight=localization_weight,
                localization_margin=localization_margin,
                activity_weight=activity_weight,
            )
            totals["loss"] += float(losses["total"].item())
            totals["task_loss"] += float(losses["task"].item())
            totals["accuracy"] += classification_accuracy(rollout.final_state, batch, layout)
            totals["target_peak_accuracy"] += target_peak_accuracy(rollout.final_state, batch, layout)
            totals["target_set_accuracy"] += target_set_accuracy(rollout.final_state, batch, layout)
            totals["slot_accuracy"] += rank_slot_accuracy(rollout.final_state, batch, layout)
            totals["sink_margin"] += mean_sink_margin(rollout.final_state, batch, layout)
            totals["localization"] += output_localization(rollout.final_state, batch, layout)

    return {key: value / batches for key, value in totals.items()}


def save_json_report(path: str | Path, data: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
