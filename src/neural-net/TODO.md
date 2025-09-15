## Phase 0 — set the goalposts (1 min)

In Conf, set:

```ts
TARGET_REJECTIONS = 4000

SAFETY_CUSHION = 1
```

Make sure your real samples file is ready, e.g. `data/samples/real-01.json`.

## Phase 1 — fresh burn-in on your dataset (≈60–70 min)

No resume; high oracle signal to shape the policy.

```bash
bun run src/neural-net/runner train 12 160 \
  --datafile=data/samples/sample-07.json \
  --assistGain=2.5 \
  --oracleRelabelFrac=0.70 \
  --elitePercentile=0.10 \
  --explorationStart=0.35 --explorationEnd=0.10 --explorationDecay=0.90
```

Tips while it runs:

- You’ll get per-epoch checkpoints at `bouncer-data/weights-scenario-2.epoch-<N>.json` and “latest” at `weights-scenario-2.json`.
- Watch for the **“Success rate”** climbing and **“Avg rejections (successful)”** trending toward **≤4k**. If it’s already excellent (success >90% and avg `rejections < TARGET_REJECTIONS`), you can jump to Phase 3.

## Phase 2 — stabilize & sharpen (≈40–50 min)

Resume from the latest; lower oracle mixing so the NN stands on its own a bit more, keep assist strong.

```bash
bun run src/neural-net/runner resume 12 180 \
  --datafile=data/samples/sample-01.json \
  --assistGain=2.0 \
  --oracleRelabelFrac=0.50 \
  --elitePercentile=0.08 \
  --explorationStart=0.15 --explorationEnd=0.05 --explorationDecay=0.90
```

What this does:

- Your trainer already decays LR every 3 epochs (×0.7), so you’ll get fine-tuning automatically.
- The elite slice is a bit tighter; training batches stay focused.
- `MAX_SAMPLES_PER_EPOCH` cap keeps memory/time sane.

## Phase 3 — quick reality checks (5–10 min total)

Run all three modes on the same file you’ll compete on:

```bash
bun run src/neural-net/runner test data/samples/sample-07.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-06.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-05.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-04.json --mode=bouncer
```

If Hybrid is clearly best (it usually is near the endgame), use that for the challenge. If Pure NN is already ≥99% success and ~4k or less rejections, you can go pure for simplicity.

## Phase 5 - Late micro adjustments

If you’re within ~500 rejections of your target and want one more nudge:

```bash
bun run src/neural-net/runner resume 3 220 \
  --datafile=data/samples/sample-01.json \
  --assistGain=2.0 \
  --oracleRelabelFrac=0.40 \
  --elitePercentile=0.12
```

## Safety valves (use if needed)

If you see a sudden “Success rate: 0.0%” collapse and loss plummets (mode-collapse style), just resume from the last good epoch file:

```bash
cp bouncer-data/weights-scenario-2.epoch-<GOOD>.json bouncer-data/weights-scenario-2.json
bun run src/neural-net/runner resume 6 150 \
  --datafile=data/samples/real-01.json \
  --assistGain=2.5 \
  --oracleRelabelFrac=0.50 \
  --elitePercentile=0.10
```

Go-time settings in the app

In initializeNeuralNetwork:

- Use the most recent good weights (or the best-\*.json you liked).
- Keep explorationRate: 0.
- Keep the updated admit(...) with the quota-helpful threshold nudges you added.

That’s it. Kick off Phase 1 now; when it finishes, run Phase 2, then the three eval commands, and ship the best mode.

## Next Steps

```bash
# save best weights when they are found:
cp bouncer-data/weights-scenario-2.json data/best/weights-scenario2-01.json

# run against all current files...
bun run src/neural-net/runner test data/samples/sample-07.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-06.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-05.json --mode=bouncer
bun run src/neural-net/runner test data/samples/sample-04.json --mode=bouncer
bun run src/neural-net/runner benchmark --mode=bouncer

# after adding new samples - lock in there distributions
bun run src/neural-net/runner curriculum \
"data/samples/sample-11.json:1,data/samples/sample-12.json:1,data/samples/sample-13.json:1,data/samples/sample-14.json:1,data/samples/sample-15.json:1,data/samples/sample-16.json:1" \
120 \
--explorationStart=0.04 --explorationEnd=0.03 \
--oracleRelabelFrac=0.40 --assistGain=3 --elitePercentile=0.10

```
