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

## v0.2 And v0.3 Probes

Train a variant:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task memory \
  --steps 350 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 24 \
  --hidden-channels 8 \
  --cell-hidden 32 \
  --damage-prob 0.08 \
  --memory-input-steps 4 \
  --save-model outputs/models/organism-v02-memory.pt \
  --report outputs/reports/train-v02-memory.json
```

Available tasks:

- `routing`: original single source/sink routing task.
- `maze`: single source/sink task with a wall and a random gap.
- `memory`: source is visible only during the input phase, then removed.
- `multi`: multiple source/sink pairs in the same tissue. Use `--min-pair-spacing 1`
  for adjacent rows and `--sink-assignment reverse` for crossing assignments.

Run mid-rollout injury recovery:

```bash
PYTHONPATH=src python3 -m organism_v01.injury \
  --model outputs/models/organism-v02-memory.pt \
  --injury-prob 0.25 \
  --report outputs/reports/injury-v02-memory.json
```

Run anti-cheat stress checks:

```bash
PYTHONPATH=src python3 -m organism_v01.stress \
  --model outputs/models/organism-v01.pt \
  --report outputs/reports/stress-v02.json
```

Render tissue frames:

```bash
PYTHONPATH=src python3 -m organism_v01.visualize \
  --model outputs/models/organism-v02-memory.pt \
  --out-dir outputs/reports/visual-v02-memory
```

The visualizer writes `frame_*.png`, `rollout.gif`, and `visual-report.json`.

## Latest Observed Results

These are run artifacts, not hardcoded scores:

- Routing v0.1: held-out `target_set_accuracy = 1.0`; 25% mid-run injury recovery stayed at `1.0`.
- Maze v0.2: held-out `target_set_accuracy = 1.0`; 25% mid-run injury recovery stayed at `1.0`.
- Memory v0.3 probe: held-out `target_set_accuracy = 1.0`; erasing the source dropped near chance.
- Multi-pair v0.2: the harder 3-pair damaged setup did not learn in the first run; an easier 2-pair no-damage setup reached held-out `target_set_accuracy = 0.9890625`.
- Multi-pair v0.3: warm-starting from the 2-pair checkpoint and fine-tuning on 3 spaced pairs with 10% static damage reached held-out `target_set_accuracy = 0.99296875`; 20% mid-run injury recovery reached `0.984375`.
- Multi-pair v0.4 adjacent: removing the spacing crutch with 3 adjacent-capable pairs, 10% static damage, and no route cues reached held-out `target_set_accuracy = 0.98125`; 20% mid-run injury recovery reached `0.9708333333333333`.
- Multi-pair v0.4 crossing: uncued reverse/crossing assignments did not clear in this architecture. Adding 3 learned-visible route-cue channels, one per pair, reached held-out `target_set_accuracy = 0.96796875`; 20% mid-run injury recovery reached `0.9583333333333334`. Controls after fixing route-cue sink erasure: normal `0.9609375`, erase-source `0.13802083333333334`, erase-sink `0.07942708333333333`, swap-source `0.0`.
- Multi-pair v0.5 uncued crossing benchmark: the gate is now explicit and rejects route-cued checkpoints. The standard adjacent checkpoint reached held-out `target_set_accuracy = 0.52734375` on uncued reverse crossing, so it fails. A first learned internal message-slot/gated update also failed: small gated checkpoint held out at `0.3385416666666667`; larger gated checkpoint held out at `0.2552083333333333`. Controls still dropped under ablation, so the failure is not a control leak; the missing piece is stable all-pair binding.
- Multi-pair v0.6 self-tagging benchmark: arbitrary random unmarked permutations are intentionally not used because they are not determined by the input. Instead v0.6 adds a deterministic cyclic assignment stress probe and a `self_tagging` update rule with persistent internal tag slots. The old adjacent checkpoint reached reverse `0.51953125` and cycle `0.265625`. The first self-tagging checkpoint reached reverse `0.23697916666666666`, cycle `0.24609375`; after lower-rate continuation it reached reverse `0.2421875`, injury `0.27734375`, cycle `0.2109375`. Result: self-tagging as implemented does not solve uncued binding.
- Multi-pair v0.7 rank-binding benchmark: added internal directional source/sink order waves, a binding curriculum, and hidden-vector diagnostics. The continued rank-binding checkpoint reached reverse `0.2734375`, injury `0.26171875`, cycle `0.2578125`, and 2-pair reverse stress `0.6041666666666666`. Result: rank waves improve 2-pair ordering but still fail stable 3-pair binding. Diagnostics showed strong rank-wave magnitude at sources but much weaker rank-wave magnitude at sinks, suggesting the order signal is not being stabilized across the body.
- Multi-pair v0.8 sink-stabilized rank benchmark: added lateral source/sink order waves plus endpoint anchors, and reserved those rank channels so the learned update cannot overwrite them. The continued checkpoint reached reverse `0.302734375`, injury `0.33984375`, cycle `0.31640625`, and 2-pair reverse stress `0.5`. Diagnostics now show balanced rank magnitude at sources and sinks, so signal delivery improved; stable learned matching across all 3 pairs is still not solved.
- Multi-pair v0.9 matching-readout benchmark: added a sink-local readout over source-label waves and an optional contrastive endpoint binding loss. The first matching-readout checkpoint reached reverse `0.26953125`, injury `0.296875`, cycle `0.265625`, and 2-pair reverse stress `0.5416666666666666`. A binding-loss continuation reached reverse `0.263671875`, injury `0.28515625`, cycle `0.27734375`, and 2-pair reverse stress `0.4270833333333333`. Result: the readout can exploit easier 2-pair structure, but endpoint embeddings stayed near random for 3-pair binding.
- Assignment ambiguity audit: reverse and cycle assignments can present identical inputs with different targets. In `outputs/reports/assignment-ambiguity-reverse-cycle.json`, reverse vs cycle had identical inputs for `2048 / 2048` sampled items and conflicting targets for `1001 / 2048` items. That means one uncued model cannot be expected to satisfy both reverse and cycle rules simultaneously without an assignment/rule cue; the cycle check is useful as a contradiction probe, not as a fair all-in-one pass gate for uncued inputs.
- Multi-pair v0.12 organ-first benchmark: added a clean 3-pair benchmark, strict and routed rank-slot diagnostics, and a rank-slot organ with separate vertical morphogen waves for top/middle/bottom seeding. The first clean checkpoint passed: reverse `target_set_accuracy = 0.6419270833333334`, cycle `0.7903645833333334`, reverse strict/routed slot accuracy `0.8133680547277132 / 0.9605034776031971`, cycle strict/routed slot accuracy `0.8407118084530035 / 0.9201388893028101`, and balanced erase-rule `0.5390625` under the `0.55` gate. Damage is intentionally not part of this pass yet.
- Multi-pair v0.13 damaged survival benchmark: added one-hot rule cues, rule-presence output gating, protected rank-slot organ updates, a final reverse/cycle curriculum, and static-damage gates. The first 5% static-damage checkpoint passed: reverse/cycle `target_set_accuracy = 0.6223958333333334 / 0.6744791666666666`, reverse/cycle routed slot accuracy `0.9192708333333334 / 0.8793402835726738`, and balanced erase-rule `0.125`. A 10% static-damage continuation also passed: reverse/cycle `target_set_accuracy = 0.5703125 / 0.8619791666666666`, reverse/cycle routed slot accuracy `0.8723958233992258 / 0.828125`, and balanced erase-rule `0.14322916666666669`.
- Multi-pair v0.14 dynamic-injury recovery benchmark: promoted mid-rollout injury into a 3-pair reverse/cycle gate with recovery checkpoints and rank-slot diagnostics. Using the v0.13 10% static-damage checkpoint, a 48-step pre-injury / 48-step recovery run with 5% base damage plus 10% new mid-rollout injury passed. Reverse recovered from immediate `target_set_accuracy = 0.5364583333333334` to final `0.6145833333333334`; cycle recovered from `0.7447916666666666` to `0.9296875`. New blocked tissue was real: reverse/cycle newly blocked fractions were `0.06282552052289248 / 0.06150535323346654`.
- Multi-pair v0.15 compounded-damage benchmark: added dynamic-injury training and a harder default benchmark with 10% base damage plus 10% new mid-rollout injury. The v0.13 checkpoint failed this gate at reverse dynamic `target_set_accuracy = 0.4791666666666667`. After a 450-step dynamic-injury continuation, the v0.15 checkpoint passed: reverse/cycle final dynamic `target_set_accuracy = 0.640625 / 0.9609375`, reverse/cycle routed slot accuracy `0.7986111069718996 / 0.79253472139438`, and reverse/cycle recovery deltas `0.07291666666666663 / 0.34635416666666663`.
- Multi-pair v0.16 generalization audit: added a scenario matrix over unseen seeds, injury timing, injury severity, mild damage, and a larger 14x14 grid. The v0.15 checkpoint passed 5 of 6 scenarios. It passed baseline, early injury, late injury, mild damage, and higher injury. The only failure was larger-grid reverse: dynamic `target_set_accuracy = 0.4921875` in the 8-batch matrix and `0.484375` in a 16-batch confirmation run, just under the `0.50` gate. Larger-grid cycle passed at `0.875` / `0.90625`. This makes v0.17's target clear: randomized grid-size / scale-generalization recovery training.
- Multi-pair v0.17 scale-generalized recovery: added scale-aware training choices so dynamic-injury training alternates 12x12/96-step and 14x14/112-step bodies in two-step blocks, ensuring reverse and cycle both see each scale. After a 500-step continuation from v0.15, the larger-grid reverse failure passed: 16-batch confirmation improved from `0.484375` to `0.80859375`. The full v0.16 matrix then passed 6 of 6 scenarios, with worst dynamic `target_set_accuracy = 0.765625`, mean dynamic `0.8951822916666666`, and larger-grid reverse/cycle `0.8125 / 0.953125`.

Example 3-pair damaged training path:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --steps 650 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 28 \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 2 \
  --lr 0.0008 \
  --init-model outputs/models/organism-v02-multi.pt \
  --save-model outputs/models/organism-v03-multi3.pt \
  --report outputs/reports/train-v03-multi3-stage1.json
```

Then continue at lower rate:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --steps 600 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 32 \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 2 \
  --lr 0.00035 \
  --init-model outputs/models/organism-v03-multi3.pt \
  --save-model outputs/models/organism-v03-multi3.pt \
  --report outputs/reports/train-v03-multi3.json
```

Example route-cued crossing run:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --steps 700 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 40 \
  --hidden-channels 12 \
  --route-channels 3 \
  --cell-hidden 48 \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --lr 0.00055 \
  --init-model outputs/models/organism-v04-cross-cued.pt \
  --save-model outputs/models/organism-v04-cross-cued.pt \
  --report outputs/reports/train-v04-cross-cued.json
```

Run the v0.5 uncued crossing benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v05 \
  --model outputs/models/organism-v05-gated.pt \
  --batches 12 \
  --report outputs/reports/benchmark-v05-gated.json
```

Train the first v0.5 gated-message candidate:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum multi_pair \
  --steps 1200 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 40 \
  --hidden-channels 24 \
  --cell-hidden 64 \
  --update-rule gated_message \
  --message-slots 8 \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --lr 0.0012 \
  --save-model outputs/models/organism-v05-gated.pt \
  --report outputs/reports/train-v05-gated.json
```

Run the v0.6 self-tagging benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v06 \
  --model outputs/models/organism-v06-self-tagging.pt \
  --batches 12 \
  --report outputs/reports/benchmark-v06-self-tagging-continued.json
```

Run the v0.12 organ-first clean 3-pair benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v12 \
  --model outputs/models/organism-v12-slot-organ-clean-smoke.pt \
  --batches 48 \
  --control-batches 24 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --seed 51000 \
  --report outputs/reports/benchmark-v12-slot-organ-clean-smoke.json
```

Run the v0.13 static-damage 3-pair survival benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v13 \
  --model outputs/models/organism-v13-damage010.pt \
  --batches 24 \
  --control-batches 12 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --damage-prob 0.10 \
  --seed 69100 \
  --report outputs/reports/benchmark-v13-damage010.json
```

Run the v0.14 dynamic-injury 3-pair recovery benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v14 \
  --model outputs/models/organism-v13-damage010.pt \
  --batches 24 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --pre-steps 48 \
  --damage-prob 0.05 \
  --injury-prob 0.10 \
  --seed 71400 \
  --report outputs/reports/benchmark-v14-dynamic-injury.json
```

Train and run the v0.15 compounded-damage recovery benchmark:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum rule_binding_final \
  --steps 450 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_rule_cued \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 48 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.1 \
  --lr 0.00025 \
  --seed 915 \
  --init-model outputs/models/organism-v13-damage010.pt \
  --save-model outputs/models/organism-v15-dynamic010.pt \
  --report outputs/reports/train-v15-dynamic010.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v15 \
  --model outputs/models/organism-v15-dynamic010.pt \
  --batches 24 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --pre-steps 48 \
  --seed 91400 \
  --report outputs/reports/benchmark-v15-dynamic010.json
```

Run the v0.16 generalization audit:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v16 \
  --model outputs/models/organism-v15-dynamic010.pt \
  --batches 8 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --scenarios all \
  --seed 101600 \
  --report outputs/reports/benchmark-v16-generalization.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v16 \
  --model outputs/models/organism-v15-dynamic010.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --scenarios larger_grid \
  --seed 101600 \
  --report outputs/reports/benchmark-v16-larger-grid-confirm.json
```

Train the v0.17 scale-generalized recovery checkpoint:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum rule_binding_final \
  --steps 500 \
  --batch-size 16 \
  --grid-size 12 \
  --grid-size-choices 12,14 \
  --rollout-steps 96 \
  --rollout-steps-choices 96,112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_rule_cued \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.1 \
  --lr 0.00018 \
  --seed 1017 \
  --init-model outputs/models/organism-v15-dynamic010.pt \
  --save-model outputs/models/organism-v17-scale.pt \
  --report outputs/reports/train-v17-scale.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v16 \
  --model outputs/models/organism-v17-scale.pt \
  --batches 8 \
  --batch-size 16 \
  --grid-size 12 \
  --rollout-steps 96 \
  --scenarios all \
  --seed 101600 \
  --report outputs/reports/benchmark-v17-scale-generalization.json
```

Audit whether two generated assignment rules are input-identical but target-conflicting:

```bash
PYTHONPATH=src python3 -m organism_v01.ambiguity \
  --assignment-a reverse \
  --assignment-b cycle \
  --seeds 64 \
  --batch-size 32 \
  --report outputs/reports/assignment-ambiguity-reverse-cycle.json
```

Train the first v0.6 self-tagging candidate:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum multi_pair \
  --steps 1000 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 40 \
  --hidden-channels 24 \
  --cell-hidden 64 \
  --update-rule self_tagging \
  --tag-slots 6 \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --lr 0.0012 \
  --save-model outputs/models/organism-v06-self-tagging.pt \
  --report outputs/reports/train-v06-self-tagging.json
```

Train the first v0.7 rank-binding candidate:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum binding \
  --steps 1200 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 48 \
  --hidden-channels 24 \
  --cell-hidden 64 \
  --update-rule rank_binding \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --lr 0.0011 \
  --save-model outputs/models/organism-v07-rank-binding.pt \
  --report outputs/reports/train-v07-rank-binding.json
```

Run hidden/rank binding diagnostics:

```bash
PYTHONPATH=src python3 -m organism_v01.diagnose_binding \
  --model outputs/models/organism-v07-rank-binding.pt \
  --report outputs/reports/diagnose-v07-rank-binding.json
```

Train the first v0.8 sink-stabilized rank candidate:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum binding \
  --steps 1200 \
  --batch-size 32 \
  --grid-size 16 \
  --rollout-steps 56 \
  --hidden-channels 32 \
  --cell-hidden 72 \
  --update-rule sink_stabilized_rank \
  --damage-prob 0.10 \
  --pair-count 3 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --lr 0.0010 \
  --save-model outputs/models/organism-v08-sink-stabilized-rank.pt \
  --report outputs/reports/train-v08-sink-stabilized-rank.json
```

## Architecture

Each cell shares the same tiny update network. The body is a grid of state vectors:

- source A/B channels
- sink marker
- damage/alive channels
- x/y chemical fields
- optional route-cue channels
- hidden tissue channels
- output A/B channels

Cells only see their local 3x3 neighborhood. Immutable environment channels are clamped after each step, and damaged cells cannot carry hidden or output state.

The default update rule is `standard`. The experimental `gated_message` update
adds transient message slots, local message mixing, and learned gates. It is not
yet sufficient for uncued reverse crossing. The experimental `self_tagging`
update adds persistent internal tag slots inside hidden tissue channels, plus
trainable tag diffusion and tag readout. It improved some two-pair behavior in
training but still failed the three-pair uncued binding benchmark.

The experimental `rank_binding` update adds four internal hidden channels for
directional source/sink order waves. It is the first mechanism here to push the
2-pair reverse stress above `0.60`, but it still does not keep three simultaneous
bindings stable under damage.

The experimental `sink_stabilized_rank` update adds lateral source/sink waves
plus endpoint anchor channels. It fixes the source/sink rank-magnitude imbalance
seen in v0.7 diagnostics, but the learned readout still does not generalize to
reliable 3-pair uncued binding.

The experimental `rank_slot_rule_cued` update adds explicit top/middle/bottom
label slots and a global rule cue. In v0.12, those slots are seeded by separate
vertical morphogen waves, then carried rightward through local tissue dynamics.
This is the first mechanism here to pass the clean 3-pair reverse/cycle gate
without pair route cues or stored answer tables.

In v0.13, `rank_slot_rule_cued` uses one-hot rule cues and gates output by rule
presence, so erasing the rule cue produces a true blank instead of an accidental
default rule. Its rank-slot organ update is protected from learned throttling,
and the trained checkpoint survives static damage up to the current 10% gate.

In v0.14, mid-rollout injury is measured as a recovery curve rather than a
single final score. The benchmark damages new tissue after the organism has
already spent half its rollout forming internal waves, then tracks immediate,
12-step, 24-step, and 48-step recovery. The current checkpoint improves after
injury on both reverse and cycle assignments while keeping routed rank-slot
accuracy above the dynamic gate.

In v0.15, the same injury mechanism is part of training. The trainer runs the
first half of the rollout, applies new damage, then backprops through the
recovery half against the injured batch. This teaches repair under compounded
damage without adding pair route cues or stored answers.

In v0.16, the benchmark stops asking only "did the trained condition pass?" and
starts asking "where does recovery fail when conditions move?" The first answer
is encouraging but specific: timing and severity generalize, while larger-grid
reverse recovery needs scale-aware training.

In v0.17, scale-aware training fixes that failure by varying both body size and
rollout length during dynamic-injury training. Grid choices advance in two-step
blocks so the alternating reverse/cycle curriculum sees both assignments at
each scale instead of accidentally pairing one assignment with one grid size.
