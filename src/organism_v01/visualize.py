from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from organism_v01.channels import ChannelLayout
from organism_v01.evaluation import choose_device, save_json_report, set_seed
from organism_v01.organism import CellularOrganism
from organism_v01.tasks import SINK_ASSIGNMENTS, TASK_NAMES, generate_task_batch


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high - low < 1e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def heatmap(values: np.ndarray) -> Image.Image:
    norm = normalize_map(values)
    red = norm
    blue = 255 - norm
    green = np.minimum(red, blue)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def panel_grid(panels: list[tuple[str, np.ndarray]], *, scale: int = 14) -> Image.Image:
    if not panels:
        raise ValueError("at least one panel is required")
    height, width = panels[0][1].shape
    label_height = 14
    gap = 4
    columns = len(panels)
    canvas = Image.new(
        "RGB",
        (columns * width * scale + (columns - 1) * gap, height * scale + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, values) in enumerate(panels):
        x = index * (width * scale + gap)
        image = heatmap(values).resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
        canvas.paste(image, (x, label_height))
        draw.text((x + 2, 1), label[:18], fill=(0, 0, 0))
    return canvas


def frame_panels(state: torch.Tensor, batch_target: torch.Tensor, layout: ChannelLayout) -> list[tuple[str, np.ndarray]]:
    state_cpu = state.detach().cpu()
    target_cpu = batch_target.detach().cpu()
    hidden = state_cpu[layout.hidden_slice]
    hidden_energy = hidden.pow(2).mean(dim=0).sqrt()
    return [
        ("source A", state_cpu[layout.source_a].numpy()),
        ("source B", state_cpu[layout.source_b].numpy()),
        ("sink", state_cpu[layout.sink].numpy()),
        ("blocked", state_cpu[layout.blocked].numpy()),
        ("hidden energy", hidden_energy.numpy()),
        ("output A", state_cpu[layout.output_a].numpy()),
        ("output B", state_cpu[layout.output_b].numpy()),
        ("target", target_cpu.max(dim=0).values.numpy()),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render organism rollout frames.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=TASK_NAMES, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--damage-prob", type=float, default=None)
    parser.add_argument("--coordinate-fields", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--min-pair-spacing", type=int, default=None)
    parser.add_argument("--sink-assignment", choices=SINK_ASSIGNMENTS, default=None)
    parser.add_argument("--memory-input-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="outputs/reports/visual-v02")
    parser.add_argument("--scale", type=int, default=14)
    return parser


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("args", {}))


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    set_seed(args.seed)

    checkpoint = torch.load(Path(args.model), map_location=device, weights_only=False)
    checkpoint_args = _checkpoint_args(checkpoint)
    layout = ChannelLayout(**checkpoint.get("layout", {"hidden_channels": 8}))
    model = CellularOrganism(
        layout=layout,
        cell_hidden=int(checkpoint_args.get("cell_hidden", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grid_size = args.grid_size or int(checkpoint_args.get("grid_size", 16))
    rollout_steps = args.rollout_steps or int(checkpoint_args.get("rollout_steps", 24))
    task = args.task or str(checkpoint_args.get("task", "routing"))
    damage_prob = args.damage_prob if args.damage_prob is not None else float(checkpoint_args.get("damage_prob", 0.12))
    coordinate_fields = args.coordinate_fields if args.coordinate_fields is not None else bool(checkpoint_args.get("coordinate_fields", True))
    pair_count = args.pair_count if args.pair_count is not None else int(checkpoint_args.get("pair_count", 3))
    min_pair_spacing = args.min_pair_spacing if args.min_pair_spacing is not None else int(checkpoint_args.get("min_pair_spacing", 1))
    sink_assignment = args.sink_assignment or str(checkpoint_args.get("sink_assignment", "aligned"))
    memory_input_steps = args.memory_input_steps if args.memory_input_steps is not None else int(checkpoint_args.get("memory_input_steps", 4))

    batch = generate_task_batch(
        task=task,
        batch_size=1,
        grid_size=grid_size,
        layout=layout,
        damage_prob=damage_prob,
        coordinate_fields=coordinate_fields,
        pair_count=pair_count,
        min_pair_spacing=min_pair_spacing,
        sink_assignment=sink_assignment,
        memory_input_steps=memory_input_steps,
        seed=args.seed,
        device=device,
    )
    with torch.no_grad():
        rollout = model(batch, steps=rollout_steps, return_frames=True)
    if rollout.frames is None:
        raise RuntimeError("rollout did not return frames")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Image.Image] = []
    for step, frame in enumerate(rollout.frames[:, 0]):
        image = panel_grid(frame_panels(frame, batch.target[0].cpu(), layout), scale=args.scale)
        path = out_dir / f"frame_{step:03d}.png"
        image.save(path)
        rendered.append(image)

    gif_path = out_dir / "rollout.gif"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=90,
        loop=0,
    )
    report = {
        "model": args.model,
        "task": task,
        "seed": args.seed,
        "grid_size": grid_size,
        "rollout_steps": rollout_steps,
        "frames": len(rendered),
        "gif": str(gif_path),
        "first_frame": str(out_dir / "frame_000.png"),
    }
    save_json_report(out_dir / "visual-report.json", report)
    print(report)


if __name__ == "__main__":
    main()
