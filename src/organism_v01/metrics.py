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
    outputs = final_state[:, layout.output_slice]
    sink_mask = batch.sink_mask[:, 0].bool()
    if int(sink_mask.sum().item()) == 0:
        return 0.0
    predictions = outputs.argmax(dim=1)
    targets = batch.target.argmax(dim=1)
    return float((predictions[sink_mask] == targets[sink_mask]).float().mean().item())


def target_peak_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    outputs = final_state[:, layout.output_slice]
    predicted_flat = outputs.flatten(1).argmax(dim=1)
    target_flat = batch.target.flatten(1).bool()
    return float(target_flat.gather(1, predicted_flat.view(-1, 1)).float().mean().item())


def target_set_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    outputs = final_state[:, layout.output_slice]
    predictions = outputs.argmax(dim=1)
    targets = batch.target.argmax(dim=1)
    sink_mask = batch.sink_mask[:, 0].bool()
    if int(sink_mask.sum().item()) == 0:
        return 0.0

    correct = predictions == targets
    per_item: list[torch.Tensor] = []
    for item in range(outputs.shape[0]):
        item_mask = sink_mask[item]
        if int(item_mask.sum().item()) == 0:
            per_item.append(torch.tensor(False, device=outputs.device))
        else:
            per_item.append(correct[item][item_mask].all())
    return float(torch.stack(per_item).float().mean().item())


def mean_sink_margin(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
) -> float:
    outputs = final_state[:, layout.output_slice]
    sink_mask = batch.sink_mask[:, 0].bool()
    if int(sink_mask.sum().item()) == 0:
        return 0.0
    sink_outputs = outputs.permute(0, 2, 3, 1)[sink_mask]
    sink_labels = batch.target.argmax(dim=1)[sink_mask]
    correct = sink_outputs.gather(1, sink_labels.view(-1, 1)).squeeze(1)
    wrong_index = 1 - sink_labels
    wrong = sink_outputs.gather(1, wrong_index.view(-1, 1)).squeeze(1)
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


def _gather_pair_vectors(
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


def binding_contrastive_loss(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Align internal source/sink endpoint codes for generated multi-pair tasks."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if batch.pair_source_rc is None or batch.pair_sink_rc is None:
        return final_state.sum() * 0.0

    pair_count = batch.pair_source_rc.shape[1]
    if pair_count < 2:
        return final_state.sum() * 0.0

    source_vectors = _gather_pair_vectors(final_state, batch.pair_source_rc, layout.hidden_slice)
    sink_vectors = _gather_pair_vectors(final_state, batch.pair_sink_rc, layout.hidden_slice)
    source_vectors = F.normalize(source_vectors, dim=-1)
    sink_vectors = F.normalize(sink_vectors, dim=-1)
    logits = torch.einsum("bpc,bqc->bpq", sink_vectors, source_vectors) / temperature
    targets = torch.arange(pair_count, device=final_state.device).expand(final_state.shape[0], pair_count)
    sink_to_source = F.cross_entropy(logits.reshape(-1, pair_count), targets.reshape(-1))
    source_to_sink = F.cross_entropy(logits.transpose(1, 2).reshape(-1, pair_count), targets.reshape(-1))
    return (sink_to_source + source_to_sink) * 0.5


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
    sink_mask_bool = batch.sink_mask[:, 0].bool()
    sink_logits = outputs.permute(0, 2, 3, 1)[sink_mask_bool]
    sink_labels = batch.target.argmax(dim=1)[sink_mask_bool]
    if sink_logits.numel() == 0:
        sink_loss = outputs.sum() * 0.0
    else:
        sink_loss = F.cross_entropy(sink_logits, sink_labels)

    outside_sink = 1.0 - batch.sink_mask.expand_as(outputs)
    quiet_targets = torch.zeros_like(outputs)
    quiet_bce = F.binary_cross_entropy_with_logits(outputs, quiet_targets, reduction="none")
    quiet_loss = (quiet_bce * outside_sink).sum() / outside_sink.sum().clamp_min(1.0)

    target_mask = batch.target.bool()
    target_scores = outputs.masked_fill(~target_mask, 1e6).flatten(1).amin(dim=1)
    outside_scores = outputs.masked_fill(target_mask, -1e6).flatten(1).amax(dim=1)
    localization_loss = F.softplus(outside_scores - target_scores + localization_margin).mean()

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
