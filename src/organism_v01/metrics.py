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


def worst_sink_consistency_loss(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Penalize the weakest sink margin for each generated item.

    Average sink cross-entropy can look healthy while one sink per item still
    slips. This loss pushes the minimum correct-vs-wrong margin across all
    sinks in an item above a requested margin.
    """

    outputs = final_state[:, layout.output_slice]
    predictions = outputs.permute(0, 2, 3, 1)
    targets = batch.target.argmax(dim=1)
    sink_mask = batch.sink_mask[:, 0].bool()
    per_item_losses: list[torch.Tensor] = []

    for item in range(outputs.shape[0]):
        item_mask = sink_mask[item]
        if int(item_mask.sum().item()) == 0:
            per_item_losses.append(outputs[item].sum() * 0.0)
            continue
        item_logits = predictions[item][item_mask]
        labels = targets[item][item_mask]
        correct = item_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        wrong = item_logits.gather(1, (1 - labels).view(-1, 1)).squeeze(1)
        weakest_margin = (correct - wrong).min()
        per_item_losses.append(F.softplus(margin - weakest_margin))

    if not per_item_losses:
        return outputs.sum() * 0.0
    return torch.stack(per_item_losses).mean()


def rank_claim_supervision_loss(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    claim_offset: int = 28,
    claim_channels: int = 4,
) -> torch.Tensor:
    """Teach sink cells which generated source rank they claim.

    The target is computed from the generated source/sink pairing metadata, not
    from stored answers. For a reverse four-pair item, the bottom sink still
    claims source rank 0 because `pair_sink_rc[:, 0]` is that source's sink.
    """

    if batch.pair_sink_rc is None:
        return final_state.sum() * 0.0
    pair_count = batch.pair_sink_rc.shape[1]
    if pair_count < 1 or pair_count > claim_channels:
        return final_state.sum() * 0.0

    claim_start = layout.hidden_start + claim_offset
    if layout.hidden_channels < claim_offset + claim_channels:
        return final_state.sum() * 0.0

    claim_logits = final_state[:, claim_start : claim_start + claim_channels]
    terms: list[torch.Tensor] = []
    for item in range(final_state.shape[0]):
        for pair_index in range(pair_count):
            row, col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
            logits = claim_logits[item, :, row, col].view(1, -1)
            target = torch.tensor([pair_index], device=final_state.device)
            terms.append(F.cross_entropy(logits, target))

    if not terms:
        return final_state.sum() * 0.0
    return torch.stack(terms).mean()


def rank_claim_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    claim_offset: int = 28,
    claim_channels: int = 4,
) -> float:
    """Per-sink accuracy for the generated source-rank claim channels."""

    if batch.pair_sink_rc is None:
        return 0.0
    pair_count = batch.pair_sink_rc.shape[1]
    if pair_count < 1 or pair_count > claim_channels:
        return 0.0

    claim_start = layout.hidden_start + claim_offset
    if layout.hidden_channels < claim_offset + claim_channels:
        return 0.0

    claim_logits = final_state[:, claim_start : claim_start + claim_channels]
    correct: list[torch.Tensor] = []
    for item in range(final_state.shape[0]):
        for pair_index in range(pair_count):
            row, col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
            predicted = claim_logits[item, :, row, col].argmax(dim=0)
            correct.append(predicted == pair_index)

    if not correct:
        return 0.0
    return float(torch.stack(correct).float().mean().item())


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


def _rank_slot_assignments(pair_count: int, slot_count: int) -> tuple[tuple[int, int], ...]:
    """Map available source ranks onto the fixed top/middle/bottom slot layout."""

    if slot_count != 3:
        if pair_count == slot_count:
            return tuple((rank_index, rank_index) for rank_index in range(slot_count))
        return ()
    if pair_count == 1:
        return ((0, 0), (2, 0))
    if pair_count == 2:
        return ((0, 0), (2, 1))
    if pair_count == 3:
        return ((0, 0), (1, 1), (2, 2))
    return ()


def rank_slot_supervision_loss(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    slot_offset: int = 12,
    slot_count: int = 3,
) -> torch.Tensor:
    """Teach sink cells the generated source-rank label slots.

    The rank-slot update reserves two hidden channels per source rank. Partial
    curriculum batches still carry rank information: one-pair batches supervise
    top and bottom with the same source, two-pair batches supervise top and
    bottom, and three-pair batches supervise top, middle, and bottom.
    """

    if batch.pair_labels is None or batch.pair_sink_rc is None:
        return final_state.sum() * 0.0
    pair_count = batch.pair_labels.shape[1]
    assignments = _rank_slot_assignments(pair_count, slot_count)
    if not assignments:
        return final_state.sum() * 0.0

    slot_width = slot_count * layout.output_count
    slot_start = layout.hidden_start + slot_offset
    if layout.hidden_channels < slot_offset + slot_width:
        return final_state.sum() * 0.0

    slot_slice = slice(slot_start, slot_start + slot_width)
    terms: list[torch.Tensor] = []
    for item in range(final_state.shape[0]):
        for sink_index in range(pair_count):
            row, col = [int(value) for value in batch.pair_sink_rc[item, sink_index]]
            logits = final_state[item, slot_slice, row, col].view(slot_count, layout.output_count)
            for slot_index, pair_index in assignments:
                label = batch.pair_labels[item, pair_index].view(1)
                terms.append(F.cross_entropy(logits[slot_index].view(1, -1), label))

    if not terms:
        return final_state.sum() * 0.0
    return torch.stack(terms).mean()


def rank_slot_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    slot_offset: int = 12,
    slot_count: int = 3,
) -> float:
    """Strict per-sink accuracy for the active top/middle/bottom label slots."""

    if batch.pair_labels is None or batch.pair_sink_rc is None:
        return 0.0
    pair_count = batch.pair_labels.shape[1]
    assignments = _rank_slot_assignments(pair_count, slot_count)
    if not assignments:
        return 0.0

    slot_width = slot_count * layout.output_count
    slot_start = layout.hidden_start + slot_offset
    if layout.hidden_channels < slot_offset + slot_width:
        return 0.0

    slot_slice = slice(slot_start, slot_start + slot_width)
    slot_logits = final_state[:, slot_slice]
    correct_sets: list[torch.Tensor] = []
    for item in range(final_state.shape[0]):
        for sink_index in range(pair_count):
            row, col = [int(value) for value in batch.pair_sink_rc[item, sink_index]]
            logits = slot_logits[item, :, row, col].view(slot_count, layout.output_count)
            slot_correct: list[torch.Tensor] = []
            for slot_index, pair_index in assignments:
                label = batch.pair_labels[item, pair_index]
                slot_correct.append(logits[slot_index].argmax(dim=0) == label)
            correct_sets.append(torch.stack(slot_correct).all())

    if not correct_sets:
        return 0.0
    return float(torch.stack(correct_sets).float().mean().item())


def rank_slot_routed_accuracy(
    final_state: torch.Tensor,
    batch: RoutingBatch,
    layout: ChannelLayout,
    *,
    slot_offset: int = 12,
    slot_count: int = 3,
) -> float:
    """Accuracy for the rank slot that each generated sink actually needs."""

    if batch.pair_labels is None or batch.pair_sink_rc is None:
        return 0.0
    pair_count = batch.pair_labels.shape[1]
    assignments = _rank_slot_assignments(pair_count, slot_count)
    if not assignments:
        return 0.0

    slot_width = slot_count * layout.output_count
    slot_start = layout.hidden_start + slot_offset
    if layout.hidden_channels < slot_offset + slot_width:
        return 0.0

    pair_to_slots: dict[int, list[int]] = {}
    for slot_index, pair_index in assignments:
        pair_to_slots.setdefault(pair_index, []).append(slot_index)

    slot_slice = slice(slot_start, slot_start + slot_width)
    slot_logits = final_state[:, slot_slice]
    correct_slots: list[torch.Tensor] = []
    for item in range(final_state.shape[0]):
        for pair_index, slot_indices in pair_to_slots.items():
            row, col = [int(value) for value in batch.pair_sink_rc[item, pair_index]]
            logits = slot_logits[item, :, row, col].view(slot_count, layout.output_count)
            label = batch.pair_labels[item, pair_index]
            for slot_index in slot_indices:
                correct_slots.append(logits[slot_index].argmax(dim=0) == label)

    if not correct_slots:
        return 0.0
    return float(torch.stack(correct_slots).float().mean().item())


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
