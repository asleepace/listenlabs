# INFO.md — Neural Net Bouncer

A small self-play trainer + neural network policy for a “bouncer” that admits or rejects people to satisfy quota constraints with minimal rejections. It combines a **seat/urgency scoring policy** with a **learned neural net policy**, and supports training, testing, and checkpointing.

---

## Quick Start

```bash
# If you don’t have weights yet (fast warmup, then test)
bun run src/neural-net/runner train 10 50 --assistGain=3 --oracleRelabelFrac=1 --elitePercentile=0.05
bun run src/neural-net/runner test --mode=hybrid

# Quick sanity test (uses latest saved weights):
# - score: hand policy only
# - bouncer: pure neural net (requires weights file)
# - hybrid: hybrid (NN + policy overrule on denies) — usually best early on
bun run src/neural-net/runner test --mode=score
bun run src/neural-net/runner test --mode=bouncer
bun run src/neural-net/runner test --mode=hybrid

# Run a small benchmark (10 games)
bun run src/neural-net/runner benchmark --mode=hybrid

# Test on a fixed dataset (optional)
bun run src/neural-net/runner test ./fixtures/samples.json --mode=hybrid

# Train on a specific dataset:
bun run src/neural-net/runner train 10 60 --data=./data/samples/sample-01.json --elitePercentile=0.10 --assistGain=3 --oracleRelabelFrac=0.5

### --- Training w/ Warm Start ---

# Re-align + warm train
bun run neural train 6 100 \
  --assistGain=1.5 \
  --oracleRelabelFrac=0.60 \
  --elitePercentile=0.20 \
  --resume

# Stabilize a bit more
bun run neural train 6 120 \
  --assistGain=0.8 \
  --oracleRelabelFrac=0.40 \
  --elitePercentile=0.20 \
  --resume

# Wean (final 3 epochs are assist/fusion OFF automatically)
bun run neural train 4 150 \
  --assistGain=0.4 \
  --oracleRelabelFrac=0.20 \
  --elitePercentile=0.15 \
  --resume

# Verify output:
bun run neural diagnose
bun run neural sanity

# hybrid (what you actually care about)
bun run neural test "" --mode=hybrid
bun run neural benchmark --mode=hybrid
```

## Example Flow

```bash
# Train longer & bigger batches of experience
bun run src/neural-net/runner train 40 200 --assistGain=2 --oracleRelabelFrac=0.8 --elitePercentile=0.1

# After ~10–15 epochs, anneal oracle relabel:
bun run src/neural-net/runner resume 30 200 --assistGain=2 --oracleRelabelFrac=0.5 --elitePercentile=0.1

# Hybrid test (no sample file):
# Pure bouncer (to gauge net-only progress):
# Score-only (to see the heuristic baseline):
bun run src/neural-net/runner test "" --mode=hybrid
bun run src/neural-net/runner test "" --mode=score
bun run src/neural-net/runner test "" --mode=bouncer

# aggressive oracle distillation
bun run src/neural-net/runner train 30 200 --oracleRelabelFrac=1.0 --elitePercentile=0.1

### --- train mode ---

# (A) Train with strong oracle relabel + bigger elite slice
bun run src/neural-net/runner train 30 200 --oracleRelabelFrac=1.0 --elitePercentile=0.1

# (B) Resume with slightly gentler relabel (keeps exploring)
bun run src/neural-net/runner resume 20 200 --oracleRelabelFrac=0.6 --elitePercentile=0.1

# (C) Test – pure net vs hybrid vs heuristic
bun run src/neural-net/runner test "" --mode=bouncer
bun run src/neural-net/runner test "" --mode=hybrid
bun run src/neural-net/runner test "" --mode=score
```

Keep policy fusion ON during training (already the default in `SelfPlayTrainer.runEpisode`) but OFF in `test()` to see pure-net progress.
Hybrid success means the scoring policy is good; the net just needs more supervised signal. The high `oracleRelabelFrac` + larger episode counts do that.

Match scoring config between training & run
Make sure `initializeScoring` in training uses the same `targetRejections` and `weights` you use in the runner. Consistency improves transfer to test/inference.

## What’s in here

- **`neural-net.ts`** – Lightweight MLP with JSON (de)serialization (`toJSON` / `fromJSON`) and batch training.
- **`training.ts`** – Self-play trainer that:
  - Simulates people using attribute frequencies + correlations.
  - Uses an **oracle** and **policy-fusion** for shaped learning.
  - Relabels elite episodes (optional) for more stable supervision.
- **`scoring.ts`** – Hand-crafted seat/urgency policy:
  - Tracks quotas, shortfalls, overages, seat scarcity, pair bonuses for negatively correlated combos, etc.
  - Provides `shouldAdmit()` and a global `getLossScore()`-style penalty metric.
- **`runner.ts`** – CLI entrypoint:
  - Trains (with checkpointing & logs), resumes, tests, benchmarks, and runs a greedy feasibility “diagnose”.

---

## Inputs & Features (state encoding)

The neural net receives a compact feature vector (default **17** features). Exact composition lives in `StateEncoder`, but it includes:

- Seats/admissions progress (capacity used / remaining)
- Rejections so far / people left in line
- Per-quota progress & urgency signals
- Attributes of the next person mapped onto unmet quotas
- Global scarcity/pressure summaries

> Architecture (default): **[17] → 32(ReLU) → 16(ReLU) → 1(Sigmoid)**

---

## Policy blend (during training)

- **NeuralNetBouncer** proposes admit/deny (with ε-exploration).
- **Policy Fusion** (seat/urgency scoring) can overrule a **deny** if the hand policy strongly prefers **admit**.
- **Teacher Assist (Oracle)** can nudge decisions probabilistically when shortfall risk is high.

At **test time** you can choose:

- `--mode=score` (hand policy only)
- `--mode=bouncer` (pure neural net)
- `--mode=hybrid` (net with fusion overrule on denies)

---

## Files & Paths

- **Weights**: `./bouncer-data/weights-scenario-2.json`
- **Checkpoints**: `./bouncer-data/weights-scenario-2.epoch-<N>.json`
- **Log**: `./bouncer-data/training-log-2.json` (per-epoch summaries & final results)

---

## Quick Start

Install deps and run via Bun:

```bash
# train 20 epochs, 50 episodes/epoch
bun run src/neural-net/runner train 20 50
```

## Overnight Training

```bash
# Phase 1 — shape the policy hard with full relabeling
nohup bun run src/neural-net/runner train 30 150 \
  --assistGain=3 \
  --oracleRelabelFrac=1 \
  --elitePercentile=0.10 \
  > train.p1.out 2>&1 &

# (later) Phase 2 — let the net carry more of the load
nohup bun run src/neural-net/runner resume 30 150 \
  --assistGain=3 \
  --oracleRelabelFrac=0.5 \
  --elitePercentile=0.10 \
  > train.p2.out 2>&1 &

```

## Test the latest weights

```bash
# score (hand policy) / bouncer (pure NN) / hybrid (NN + fusion)
bun run src/neural-net/runner test --mode=hybrid

# Run a 10 game benchmark
bun run src/neural-net/runner benchmark --mode=hybrid

# Check scenario feasibility with a greedy oracle:
bun run src/neural-net/runner diagnose
```

## Training — initial, resume, longer runs

```bash
# stronger supervision early
bun run src/neural-net/runner train 40 80 \
  --assistGain=3 \
  --oracleRelabelFrac=1.0 \
  --elitePercentile=0.05

```

Saves checkpoints + latest weights to bouncer-data/ and appends to training-log-2.json. Ends with a 100-episode test.

```bash
# cotinue training: warm-start from latest
bun run src/neural-net/runner resume 20 80 \
  --assistGain=2.5 \
  --oracleRelabelFrac=0.5 \
  --elitePercentile=0.1
```

The trainer auto-warm-starts the net (lower exploration if resuming).

### Useful flags

- `--assistGain=<k>`: scales teacher-assist probability (higher → more oracle nudges when behind).
- `--oracleRelabelFrac=<0..1>`: % of elite steps relabeled by oracle (1.0 early, taper to 0.3–0.5).
- `--elitePercentile=<0..1>`: top slice used for training batches (e.g., 0.05 = top 5%).
- `--resume`: same effect as the resume command.

## Reward shaping (high level)

(See `training.ts` → `scoreEpisode()`.)

- Base objective: minimize rejections
- Penalize shortfalls (linear + quadratic) → strong pressure to meet all quotas
- Penalize surplus (mild) → avoid overshooting a single quota
- Large loss penalty if any quota unmet or rejection cap hit

Net effect: “meet all constraints with low rejections.”

## Tips & Troubleshooting

Archive old weights (don’t delete) to avoid mismatches:

```bash
mkdir -p bouncer-data/archive
mv bouncer-data/weights-scenario-2.json bouncer-data/archive/weights-scenario-2.$(date +%s).json
```

Then train fresh (omit `--resume`).

### Stuck at 0% success? Try:

- Increase supervision early: `--oracleRelabelFrac=1.0`, `--assistGain=3`.
- Narrow elite slice: `--elitePercentile=0.05`.
- Train longer (more epochs/episodes).
- Confirm feature size (`17`) and architecture match the weights you’re loading.

### Logs & checkpoints:

Inspect `training-log-2.json` for loss, success rate, exploration, best episode per epoch.

You can revert to a specific checkpoint by copying it over `weights-scenario-2.json` and re-running test/resume.

### Handy One-liners

```bash
# fresh train, heavier supervision early
bun run src/neural-net/runner train 40 80 --assistGain=3 --oracleRelabelFrac=1 --elitePercentile=0.05

# continue training from latest
bun run src/neural-net/runner resume 20 80 --assistGain=2.5 --oracleRelabelFrac=0.5 --elitePercentile=0.1

# quick sanity test (hybrid tends to be strongest early)
bun run src/neural-net/runner test --mode=hybrid

# feasibility check on the scenario
bun run src/neural-net/runner diagnose
```
