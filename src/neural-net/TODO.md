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
bun run src/neural-net/runner train 20 150 \
  --datafile=data/samples/real-01.json \
  --assistGain=3 \
  --oracleRelabelFrac=0.80 \
  --elitePercentile=0.12

```

Tips while it runs:

- You’ll get per-epoch checkpoints at `bouncer-data/weights-scenario-2.epoch-<N>.json` and “latest” at `weights-scenario-2.json`.
- Watch for the **“Success rate”** climbing and **“Avg rejections (successful)”** trending toward **≤4k**. If it’s already excellent (success >90% and avg `rejections < TARGET_REJECTIONS`), you can jump to Phase 3.

## Phase 2 — stabilize & sharpen (≈40–50 min)

Resume from the latest; lower oracle mixing so the NN stands on its own a bit more, keep assist strong.

```bash
bun run src/neural-net/runner resume 12 180 \
  --datafile=data/samples/real-01.json \
  --assistGain=2.5 \
  --oracleRelabelFrac=0.45 \
  --elitePercentile=0.08
```

What this does:

- Your trainer already decays LR every 3 epochs (×0.7), so you’ll get fine-tuning automatically.
- The elite slice is a bit tighter; training batches stay focused.
- `MAX_SAMPLES_PER_EPOCH` cap keeps memory/time sane.

## Phase 3 — quick reality checks (5–10 min total)

Run all three modes on the same file you’ll compete on:

```bash
# Pure NN
bun run src/neural-net/runner test data/samples/real-01.json --mode=bouncer

# Hybrid (usually best)
bun run src/neural-net/runner test data/samples/real-01.json --mode=hybrid

# Policy baseline
bun run src/neural-net/runner test data/samples/real-01.json --mode=score

# Alias
bun run neural sanity
```

If Hybrid is clearly best (it usually is near the endgame), use that for the challenge. If Pure NN is already ≥99% success and ~4k or less rejections, you can go pure for simplicity.

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
