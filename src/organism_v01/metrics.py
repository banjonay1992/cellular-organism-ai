from __future__ import annotations

import torch
import torch.nn.functional as F

from organism_v01.channels import ChannelLayout
from organism_v01.tasks import RoutingBatch


def output_logits_at_sink(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> torch.Tensor:
    outputs = final_state[:, layout.output_slice]
    return (outputs * batch.sink_mask).sum(dim=(2, 3))


def classification_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    logits = output_logits_at_sink(final_state, batch, layout)
    predictions = logits.argmax(dim=1)
    return float((predictions == batch.labels).float().mean().item())


def target_peak_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    outputs = final_state[:, layout.output_slice]
    batch_size, _, height, width = outputs.shape
    predicted_flat = outputs.flatten(1).argmax(dim=1)
    target_flat = (
        batch.labels * height * width
        + batch.sink_rc[:, 0].to(batch.labels.device) * width
        + batch.sink_rc[:, 1].to(batch.labels.device)
    )
    if target_flat.shape[0] != batch_size:
        raise ValueError("target size does not match batch size")
    return float((predicted_flat == target_flat).float().mean().item())


def mean_sink_margin(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    logits = output_logits_at_sink(final_state, batch, layout)
    correct = logits.gather(1, batch.labels.view(-1, 1)).squeeze(1)
    wrong_index = 1 - batch.labels
    wrong = logits.gather(1, wrong_index.view(-1, 1)).squeeze(1)
    return float((correct - wrong).mean().item())


def output_localization(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    output_energy = final_state[:, layout.output_slice].sigmoid()
    sink_energy = (output_energy * batch.sink_mask).sum(dim=(1, 2, 3))
    total_energy = output_energy.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return float((sink_energy / total_energy).mean().item())


def compute_loss(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    activity_loss: torch.Tensor,
    field_weight: float = 0.5,
    localization_weight: float = 1.0,
    localization_margin: float = 1.0,
    activity_weight: float = 1e-3,
) -> dict[str, torch.Tensor]:
    outputs = final_state[:, layout.output_slice]
    logits = output_logits_at_sink(final_state, batch, layout)
    sink_loss = F.cross_entropy(logits, batch.labels)

    outside_sink = 1.0 - batch.sink_mask.expand_as(outputs)
    quiet_targets = torch.zeros_like(outputs)
    quiet_bce = F.binary_cross_entropy_with_logits(outputs, quiet_targets, reduction="none")
    quiet_loss = (quiet_bce * outside_sink).sum() / outside_sink.sum().clamp_min(1.0)

    label_index = batch.labels.view(-1, 1, 1, 1).expand(-1, 1, outputs.shape[2], outputs.shape[3])
    correct_map = outputs.gather(1, label_index).squeeze(1)
    sink_correct = (correct_map * batch.sink_mask.squeeze(1)).sum(dim=(1, 2))
    outside_correct = correct_map.masked_fill(batch.sink_mask.squeeze(1).bool(), -1e6)
    outside_peak = outside_correct.flatten(1).amax(dim=1)
    localization_loss = F.softplus(outside_peak - sink_correct + localization_margin).mean()

    activity_term = activity_loss * activity_weight
    localization_term = localization_loss * localization_weight
    task_loss = sink_loss + quiet_loss * field_weight + localization_term
    return {
        "total": task_loss + activity_term,
        "task": task_loss,
        "sink": sink_loss,
        "quiet": quiet_loss,
        "localization": localization_loss,
        "activity": activity_term,
    }
