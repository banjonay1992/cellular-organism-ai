# Organism v0.1

This is a first proof for a cellular neural organism: a grid of tiny shared neural cells that communicate locally, preserve environmental signals, and learn generated routing tasks.

The goal is not to fake a good-looking demo. The goal is to measure whether the organism learns a task generated at runtime, and whether a trained checkpoint beats its untrained baseline on fresh random batches.

## v0.1 Goal

Build a tiny artificial tissue that can:

- receive a labeled source signal in one part of the body
- receive an unlabeled sink marker elsewhere
- propagate the source identity through local cell updates
- emit the correct output at the sink
- operate with randomly damaged cells in the grid
- report actual measured metrics, not pre-filled scores

## What Is Deliberately Not Hardcoded

- No fixed answer table is stored.
- Source positions, sink positions, labels, and damage masks are sampled from seeds.
- Training batches and evaluation batches are generated independently.
- Evaluation reports are written from live model outputs.

## Run Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Train A Smoke Model

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --steps 450 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 24 \
  --hidden-channels 8 \
  --cell-hidden 32 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --localization-margin 1.0 \
  --seed 11 \
  --save-model outputs/models/organism-v01.pt \
  --report outputs/reports/train-v01.json
```

## Evaluate A Checkpoint

```bash
PYTHONPATH=src python3 -m organism_v01.evaluate \
  --model outputs/models/organism-v01.pt \
  --batches 20 \
  --seed 9001 \
  --report outputs/reports/eval-v01.json
```

## Run Ablation Controls

```bash
PYTHONPATH=src python3 -m organism_v01.controls \
  --model outputs/models/organism-v01.pt \
  --batches 30 \
  --seed 9500 \
  --report outputs/reports/controls-v01.json
```

The main score is `target_peak_accuracy`: the single highest output across every cell and both labels must be the sampled target label at the sampled sink cell. Plain label accuracy at the sink is tracked too, but it is not enough by itself because a model can weakly broadcast the right label everywhere.

Useful controls:

- `erase_source` should fall near chance.
- `erase_sink` should fail target-peak accuracy.
- `swap_source` should fail.

## Architecture

Each cell shares the same tiny update network. The body is a grid of state vectors:

- source A/B channels
- sink marker
- damage/alive channels
- x/y chemical fields
- hidden tissue channels
- output A/B channels

Cells only see their local 3x3 neighborhood. Immutable environment channels are clamped after each step, and damaged cells cannot carry hidden or output state.
