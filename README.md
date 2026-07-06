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
- Multi-pair v0.18 four-pair probe: added a clean 4-pair dynamic-injury audit without changing the trained model or adding pair route cues. The fixed top/middle/bottom rank-slot diagnostics are now explicitly marked over-capacity for `pair_count = 4`, so the probe gates only live sink outputs and injury evidence. The v0.17 checkpoint did not pass the full probe because reverse finished at dynamic `target_set_accuracy = 0.3359375`, but the result exposed a useful surprise: cycle generalized strongly to four pairs, with static/dynamic `target_set_accuracy = 0.75390625 / 0.7578125`, dynamic `target_peak_accuracy = 0.9921875`, and real newly blocked tissue at `0.06391501822508872`.
- Multi-pair v0.19 rank diagnostic: added a diagnostic-only assignment map for 4-pair aligned/cycle/reverse runs, including source-rank accuracy, sink-rank accuracy, margins, label bias, recovery curves, and exact counts of how many sinks were correct per item. The 16-batch result shows cycle is balanced across all ranks and still strong after injury: dynamic `target_set_accuracy = 0.76171875`, per-sink accuracy `0.93359375`, and all source ranks above `0.92`. Reverse is not failing uniformly: outer source ranks were `0.8984375 / 0.94921875`, but the two inner ranks were `0.59375 / 0.56640625`, leaving dynamic `target_set_accuracy = 0.390625`. This makes the next architecture target clear: separate middle ranks cleanly under the reverse transform.
- Multi-pair v0.20 relative-rank/mirror experiment: added a `relative_rank_rule_cued` organ with source/sink count waves and source-label rank moments, so the body can represent more than top/middle/bottom without adding a fourth answer slot. A cold 300-step run proved the organ can form distinct four-rank coordinates, but was not competitive. The stronger branch continued from the v0.17 organism with 4-pair mirror training and no slot supervision. On the same v0.19 seed, reverse inner source ranks improved from `0.57421875 / 0.546875` to `0.66015625 / 0.7421875`; reverse dynamic `target_set_accuracy` improved from `0.3359375` to `0.39453125`, while cycle stayed usable at `0.75390625`. The v0.18 gate still fails because reverse static accuracy is only `0.36328125` against the `0.40` static gate, so v0.20 is progress, not a solved 4-pair organism.
- Multi-pair v0.21 consistency objective: added a worst-sink consistency loss that penalizes each generated item's weakest correct-vs-wrong sink margin, targeting the common "3 of 4 sinks right" failure directly. The objective is tested and trainable through `--consistency-weight` / `--consistency-margin`, but it did not solve four-pair reverse by itself. A mixed continuation from v0.20 improved cycle dynamic `target_set_accuracy` to `0.828125`, but reverse dynamic stayed at `0.3828125` versus the v0.20 same-seed `0.39453125`, with `44.921875%` of items still landing at exactly 3 correct sinks. Reverse-only and static-reverse continuations also failed the v0.18 gate; the static-reverse branch ended at reverse static/dynamic `0.37109375 / 0.35546875`, while cycle stayed `0.71484375 / 0.71875`. The result is useful: local margin pressure can sharpen an easier rule, but the reverse wall is a coordinated rank-binding error, not just weak sink logits.
- Multi-pair v0.22 repair-bus experiment: added `rank_slot_repair_rule_cued`, which keeps the rank-slot organ but reserves four hidden channels for a recurrent sink/source repair loop. Sinks broadcast their current A/B vote leftward, sources answer rightward with label repair signals, and a new repair readout can be trained either with the whole organism or alone through `--train-repair-only`. The mechanism is alive and tested, but this version does not solve four-pair reverse. Mixed repair training reached reverse static/dynamic `0.3984375 / 0.37109375` and cycle dynamic `0.79296875`; reverse-only repair reached reverse dynamic `0.3671875`; repair-only training preserved the base more cleanly but reached only `0.359375`. The lesson is sharper: a feedback bus is not enough if its messages are just label votes. The next organ probably needs explicit permutation/assignment state, where sinks negotiate which source rank they claim, not only which label they currently believe.
- Multi-pair v0.23 rank-claim organ: added `rank_slot_claim_rule_cued`, with 8 source-rank label channels, 4 sink claim channels, claim-only warm starts, and a generated claim supervision loss. The internal claim state does learn above chance: static clean claim pretrain ended at train claim accuracy `0.4375`, and dynamic continuation peaked at `0.53125` on a logged minibatch. But the current output path does not generalize. Held-out v0.19 diagnostics for `organism-v23-claim-dynamic.pt` reached reverse static/dynamic `0.20703125 / 0.26171875`, cycle dynamic `0.05859375`, and aligned dynamic `0.07421875`, worse than v0.20. Conclusion: explicit claim state is learnable, but claim-to-output takeover is wrong; the next attempt should verify held-out claim accuracy first, then distill claims into output with a safer residual gate.
- Multi-pair v0.24 safe residual scaffold: added held-out claim diagnostics, `rank_slot_claim_residual_rule_cued`, claim-gate-only and claim-state-only training modes, and a hidden-expansion warm start that grows the 32-hidden v0.20 body into 44 hidden channels. This found the real v0.23 hazard: claim channels at hidden offsets 20-31 collided with tissue the old organism already used. With claims moved into new channels 32-43 and the residual gate closed, v0.24 preserves the v0.20 baseline on the same v0.19 matrix: reverse static/dynamic `0.390625 / 0.39453125`, cycle dynamic `0.75390625`. Claim-state-only training keeps that output behavior intact and lifts reverse dynamic claim accuracy from random `0.25` to `0.369140625`, but the inner claim ranks are still weak (`0.140625 / 0.0625`). Result: safe organ transplantation works; the next problem is a better claim coordinate/curriculum, not output gating.
- Multi-pair v0.25 claim-coordinate curriculum: added generated binary claim-coordinate supervision (`inner/outer` and `upper/lower`), a `claim_coordinate` curriculum, inner-rank-only claim loss, and logs for coordinate/inner accuracy. The answer body stayed intact, but the claim organ exposed a clearer tradeoff. Coordinate-heavy training improved reverse dynamic claim accuracy to `0.4365234375`, but it solved only outer ranks (`0.74609375 / 0.0 / 0.0 / 1.0`). Inner-heavy training flipped the failure: reverse dynamic inner claims rose to `0.55078125 / 0.40234375`, but outer ranks fell to `0.0 / 0.0`, with total claim accuracy `0.23828125`. This is progress as diagnosis, not as a solved organism.
- Multi-pair v0.26 factorized claim seed: added `rank_slot_claim_factor_rule_cued`, where two learned binary coordinate heads compose into four exact source-rank claim logits. This is more biology-ish than a flat four-way claim head, and it is tested/warm-startable, but it still did not solve the held-out four-pair wall. On the same v0.19 matrix, reverse dynamic output remained preserved at `0.39453125`, while reverse dynamic claim accuracy stayed near random at `0.248046875`; the two inner source ranks were learned (`0.4609375 / 0.53125`) and both outer ranks collapsed to `0.0`. Conclusion: loss/claim-head shaping can move which ranks the organism claims, but the current upstream sink-rank signal cannot bind all four ranks at once. The next frontier is diagnosing and improving the rank signal itself before opening the claim gate.

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

Run the v0.18 four-pair probe:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v18 \
  --model outputs/models/organism-v17-scale.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --seed 111800 \
  --report outputs/reports/benchmark-v18-four-pair.json
```

Run the v0.19 four-pair rank diagnostic:

```bash
PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v17-scale.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 111900 \
  --report outputs/reports/benchmark-v19-four-pair-diagnostics.json
```

Train and probe the v0.20 mirror continuation:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum rule_binding_final \
  --steps 300 \
  --batch-size 8 \
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
  --pair-count 4 \
  --min-pair-spacing 1 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --lr 0.00012 \
  --seed 1021 \
  --init-model outputs/models/organism-v17-scale.pt \
  --save-model outputs/models/organism-v20-mirror-finetune.pt \
  --report outputs/reports/train-v20-mirror-finetune.json

PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 200 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_rule_cued \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --lr 0.00008 \
  --seed 1022 \
  --init-model outputs/models/organism-v20-mirror-finetune.pt \
  --save-model outputs/models/organism-v20-reverse-polish.pt \
  --report outputs/reports/train-v20-reverse-polish.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v20-reverse-polish.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v20-reverse-polish-diagnostics.json
```

Train and probe the v0.21 worst-sink consistency objective:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum rule_binding_final \
  --steps 240 \
  --batch-size 8 \
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
  --pair-count 4 \
  --min-pair-spacing 1 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --consistency-weight 0.25 \
  --consistency-margin 1.0 \
  --lr 0.00006 \
  --seed 1023 \
  --init-model outputs/models/organism-v20-reverse-polish.pt \
  --save-model outputs/models/organism-v21-consistency.pt \
  --report outputs/reports/train-v21-consistency.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v21-consistency.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v21-consistency-diagnostics.json
```

Train and probe the v0.22 repair bus:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum rule_binding_final \
  --steps 260 \
  --batch-size 8 \
  --grid-size 12 \
  --grid-size-choices 12,14 \
  --rollout-steps 96 \
  --rollout-steps-choices 96,112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_repair_rule_cued \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --consistency-weight 0.20 \
  --consistency-margin 1.0 \
  --lr 0.00007 \
  --seed 1026 \
  --init-model outputs/models/organism-v20-reverse-polish.pt \
  --save-model outputs/models/organism-v22-repair.pt \
  --report outputs/reports/train-v22-repair.json

PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 260 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_repair_rule_cued \
  --train-repair-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --consistency-weight 0.25 \
  --consistency-margin 1.0 \
  --lr 0.0008 \
  --seed 1028 \
  --init-model outputs/models/organism-v20-reverse-polish.pt \
  --save-model outputs/models/organism-v22-repair-only.pt \
  --report outputs/reports/train-v22-repair-only.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v22-repair.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v22-repair-diagnostics.json
```

Train and probe the v0.23 rank-claim organ:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 220 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_rule_cued \
  --train-claim-only \
  --damage-prob 0.0 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --claim-weight 2.0 \
  --consistency-weight 0.10 \
  --consistency-margin 1.0 \
  --lr 0.0015 \
  --seed 1030 \
  --init-model outputs/models/organism-v20-reverse-polish.pt \
  --save-model outputs/models/organism-v23-claim-static.pt \
  --report outputs/reports/train-v23-claim-static.json

PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 260 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 32 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_rule_cued \
  --train-claim-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --claim-weight 1.2 \
  --consistency-weight 0.15 \
  --consistency-margin 1.0 \
  --lr 0.0008 \
  --seed 1031 \
  --init-model outputs/models/organism-v23-claim-static.pt \
  --save-model outputs/models/organism-v23-claim-dynamic.pt \
  --report outputs/reports/train-v23-claim-dynamic.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v23-claim-dynamic.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v23-claim-dynamic-diagnostics.json
```

Train and probe the v0.24 safe residual scaffold:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 1 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 44 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_residual_rule_cued \
  --train-claim-gate-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --lr 0.0 \
  --seed 1036 \
  --init-model outputs/models/organism-v20-reverse-polish.pt \
  --save-model outputs/models/organism-v24-claim-residual-expanded-closed.pt \
  --report outputs/reports/train-v24-claim-residual-expanded-closed.json

PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum none \
  --steps 420 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 44 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_residual_rule_cued \
  --train-claim-state-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --min-pair-spacing 1 \
  --sink-assignment reverse \
  --field-weight 0.5 \
  --localization-weight 1.0 \
  --slot-weight 0.0 \
  --claim-weight 5.0 \
  --lr 0.004 \
  --seed 1037 \
  --init-model outputs/models/organism-v24-claim-residual-expanded-closed.pt \
  --save-model outputs/models/organism-v24-claim-state.pt \
  --report outputs/reports/train-v24-claim-state.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v24-claim-state.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v24-claim-state-diagnostics.json
```

Train and probe the v0.25/v0.26 claim-coordinate variants:

```bash
PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum claim_coordinate \
  --steps 600 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 44 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_residual_rule_cued \
  --train-claim-state-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --sink-assignment reverse \
  --claim-weight 1.0 \
  --claim-coordinate-weight 5.0 \
  --lr 0.003 \
  --seed 1038 \
  --init-model outputs/models/organism-v24-claim-state.pt \
  --save-model outputs/models/organism-v25-claim-coordinate.pt \
  --report outputs/reports/train-v25-claim-coordinate.json

PYTHONPATH=src python3 -m organism_v01.train \
  --task multi \
  --curriculum claim_coordinate \
  --steps 600 \
  --batch-size 8 \
  --grid-size 14 \
  --rollout-steps 112 \
  --hidden-channels 44 \
  --rule-channels 3 \
  --cell-hidden 64 \
  --update-rule rank_slot_claim_factor_rule_cued \
  --train-claim-state-only \
  --damage-prob 0.10 \
  --dynamic-injury-prob 0.10 \
  --dynamic-injury-pre-steps 56 \
  --pair-count 4 \
  --sink-assignment reverse \
  --claim-weight 1.0 \
  --claim-coordinate-weight 2.0 \
  --claim-inner-weight 1.0 \
  --lr 0.003 \
  --seed 1040 \
  --init-model outputs/models/organism-v24-claim-residual-expanded-closed.pt \
  --save-model outputs/models/organism-v26-claim-factor.pt \
  --report outputs/reports/train-v26-claim-factor.json

PYTHONPATH=src python3 -m organism_v01.benchmark_v19 \
  --model outputs/models/organism-v26-claim-factor.pt \
  --batches 16 \
  --batch-size 16 \
  --grid-size 14 \
  --rollout-steps 112 \
  --pre-steps 56 \
  --damage-prob 0.10 \
  --injury-prob 0.10 \
  --pair-count 4 \
  --assignments aligned,cycle,reverse \
  --seed 112100 \
  --report outputs/reports/benchmark-v26-claim-factor-diagnostics.json
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

In v0.18, the benchmark probes four simultaneous pairs before adding a new
organ. The current top/middle/bottom rank-slot diagnostics are intentionally
reported as unsupported above three pairs, while output metrics still measure
whether the tissue answers every live sink. This exposed an asymmetry worth
studying: the existing organism transfers surprisingly well to the four-pair
cycle rule but not to the four-pair reverse rule.

In v0.19, the diagnostic view shows why. Four-pair reverse keeps the outer
ranks mostly intact but confuses the two inner ranks, while four-pair cycle is
balanced across every source and sink rank. The next scalable organ should
therefore represent relative rank with enough resolution to distinguish inner
positions under mirror-like transforms, not just add another fixed answer slot.

In v0.20, `relative_rank_rule_cued` adds that representation directly: count
waves produce continuous source/sink rank coordinates, and label-moment channels
carry rank-indexed source labels across the body. The cold-start organ is not
yet competitive, which suggests the successful 3-pair organism has learned
useful dynamics that should not be thrown away. A mirror-focused continuation
from v0.17 improves the measured inner-rank failure, but still leaves too many
items with exactly one wrong sink. The next target is converting that per-rank
improvement into full-item consistency.

In v0.21, the first attempt at full-item consistency is a loss-side pressure,
not a new organ: each generated item is scored by its weakest live sink margin.
That makes the failure visible to backprop, but the experiments show margin
pressure alone does not coordinate the four sink decisions under reverse. The
next useful change should probably add an item-level repair or verifier loop
that lets sinks resolve a shared rank assignment, rather than asking each sink
to become more confident independently.

In v0.22, that repair loop exists as state, not just loss: sink votes and source
replies move through reserved hidden channels and can be used by a repair
readout. The failed result says the content of the message matters. Label-vote
feedback tends to reinforce local beliefs, while the unsolved reverse case
needs shared assignment state: a sink should communicate "I claim source rank
2" or "rank 1 and rank 2 conflict", not merely "I think the answer is A".

In v0.23, that shared assignment state is explicit. Sources seed rank-indexed
A/B labels, sinks maintain four claim channels, and the claim readout can choose
the source-rank label a sink believes it owns. The encouraging sign is that the
claim channels can learn generated rank targets without hardcoded answers. The
bad sign is that using those claims as an output path currently destabilizes the
held-out four-pair behavior. The next version should separate three questions:
whether held-out claims are right, whether claims survive injury, and whether a
small residual output gate can use them without erasing the already useful v0.20
behavior.

In v0.24, the residual organ is separated from the old tissue instead of sharing
hidden channels. The loader can transplant a 32-hidden checkpoint into a
44-hidden body by copying old env/hidden/output channel weights into their new
positions and leaving the added claim tissue quiet. That makes a closed claim
gate genuinely safe: output behavior matches v0.20 while the new claim channels
can be measured and trained. The remaining failure is narrower and cleaner:
claim-state training helps the outer reverse ranks, but the two inner claim
ranks are still not represented well enough to deserve control over answers.

In v0.25/v0.26, the claim organ gained explicit generated coordinate losses and
then a factorized two-bit claim seed. These additions made the failure more
legible: the same body can learn outer claims or inner claims, and the factorized
seed can represent the inner split, but the system still does not maintain a
balanced four-rank claim assignment under reverse/cycle injury. That points away
from the output gate and toward the upstream rank representation: before claims
can safely drive answers, the organism needs a stronger local signal for all four
sink/source ranks, not just a better loss on the final claim channels.
