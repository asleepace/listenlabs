# Bandit Bouncer

A pragmatic contextual bandit that admits people into a limited-capacity event while satisfying minimum attribute counts (constraints) and pacing the admit rate so you neither run out of seats too early nor starve the queue at the end.

#### This README covers:

- What the algorithm optimizes and learns
- How decisions are made (gates → bandit → controller)
- Output fields (getProgress() / getOutput())
- All config knobs (CFG) and how to tune them
- Troubleshooting endgame and other common issues

## Quickstart

### 1. What it’s optimizing

Behaviorally, it tries to:

1. Meet all minimum counts for attributes (e.g., techno_lover ≥ 650, berlin_local ≥ 750) before capacity runs out.
2. Keep rejections low (your “score” is totalRejected), i.e., be selective but not too selective.
3. Track a target admit-rate curve so you don’t blow through seats early or finish late.

There’s no single closed-form objective. Instead we combine:

- Reward shaping (teaches the bandit which admits are useful),
- Hard policy gates (feasibility/pacing safety rails),
- A PI controller that nudges the threshold to hit a target admit rate.

### 2. What it learns

A linear value model over contextual features:

```
value = w · x
```

#### Features include:

- Indicators for each attribute if the person has it and that constraint is still unmet.
- Progress terms per constraint (we feed 1 - progress, so weights are clamped negative → “already-met” tracks are penalized).
- Capacity used (keeps behavior appropriate as seats dwindle).
- Scarcity/feasibility signals (e.g., a scarcity feature for creative).

#### Online updates happen every decision:

- Admits apply full reward updates.
- Rejections update after warmup with a clamped negative reward (prevents over-learning from noisy early rejections).
- Clamps keep signs sensible:
  - Progress weights ≤ progressFloor (negative)
  - Capacity weight ≤ capacityFloor (≤ 0)
  - Scarcity weight ≥ scarcityFloor (≥ 0)

### 3. How a decision happens

#### 3.1 Policy Gates (no learning)

Before the bandit score is considered, gates ensure feasibility and pacing:

- Pace gate: if the most-behind constraint lags your overall usage by a gap, reject people who don’t help it.
- Reserve (late only): if projected draws needed for unmet constraints exceed a buffer of remaining seats, hold back seats.
- Top-1 must-have: when one constraint is dominant and far behind, require that attribute.
- Late feasibility: late phases require each admit to meaningfully help at-risk tracks.
- Infeasible cutoff by phase: if feasibility ratio is too high for the current phase, reject non-helpers.

If none block, the bandit decides.

#### 3.2 Bandit Score vs Adaptive Threshold

- Compute score = `w·x` (+ small decaying noise early for exploration).
- Compare to an adaptive threshold:
  - Robust quantile of recent scores (median + z\*sigma)
  - Capacity bias (higher when you’ve used more seats)
  - PI controller correction (tracks target admit rate)
    − Urgency if constraints look risky (higher feasibility ratios)
- Admit if score + noise > threshold, else reject.

### 3.3 Endgame helpers

- `FILL`: when all constraints are satisfied and you’re late, just fill seats (optionally without learning).
- `FINISH`: when some constraints remain but the smallest shortfall is small or you have roomy ratio (remaining slots vs smallest shortfall), admit anyone who helps any unmet one. This prevents the “stuck at 1–10 short” failure mode.

## 4. Public API

```ts
await bouncer.initializeLearningData()
```

Loads recent games (if any), optionally warm-starts the bandit or restores from snapshot.

```ts
bouncer.admit({ status, nextPerson }) => boolean
```

Returns true if admitted, false if rejected.

Internally: gates → optional FINISH/FILL → bandit → learning → counters.

```ts
bouncer.getProgress()
```

Returns a compact live snapshot, e.g.:

```json
{
  "attributes": [
    {
      "attribute": "techno_lover",
      "admitted": 655,
      "required": 650,
      "satisfied": true,
      "progress": 1,
      "shortfall": 0,
      "frequency": 0.6265,
      "empiricalFrequency": 0.6382,
      "scarcity": 0
    }
  ],
  "lastLogs": ["ADMIT: r=... v=... useful=[...]"],
  "banditStats": {
    "weights": [2.12, 5, 5, 2.32, -0.15, -0.15],
    "decisionCount": 2795,
    "capacityUsed": 0.962,
    "threshold": 3.53,
    "avgWeight": 1.71
  },
  "remainingSlots": 38,
  "totalAdmitted": 962,
  "totalRejected": 6147,
  "threshold": 3.525,
  "lastRawValue": -0.097,
  "admitRate": 0.351,
  "feasibility": { "maxRatio": 0, "mostCritical": "..." },
  "top": [], // top 3 unmet ratios
  "controller": {
    "targetRate": 0.1,
    "error": 0.251
  }
}
```

along with

```ts
bouncer.getOutput()
```

Final snapshot and summary. Also persists:

```
summary.json
```

Full game state via `Disk.saveGameState(...)`

### 5. Configuration (CFG) & Tuning Guide

Below, each block shows what the knobs do and how to tune them.

### 5.1 Capacity / schedule

```ts
{
  TARGET_RATE_BASE: { early: 0.24, mid: 0.16, late: 0.1 }
  TARGET_RATE_MIN: 0.08
  TARGET_RISK_PULL: { slope: 0.025, max: 0.04 }
}
```

- What: Desired admit rate by phase. The controller nudges the threshold to keep the admit EMA near this target.
- Tune:
  - If you run out of seats early, lower early and/or increase RISK_PULL (more conservative when feasibility looks bad).
  - If you starve late, raise late slightly or increase TARGET_RATE_MIN.

### 5.2 Linear bandit

```ts
{
  BANDIT: {
    featureDim: 10,
    eta: 0.12,
    emaBeta: 0.035,
    iBeta: 0.002,
    priorWeights: [1.2, 1.0, 1.8, 1.0, -0.8, -0.8, -1.4, -0.8, -1.6, 1.2],
    weightClamp: [-5, 5] as const,
    progressFloor: -0.15,
    capacityFloor: -0.1,
    scarcityFloor: 0,
    initRecentValues: { n: 60, start: 1.6, step: 0.01 },
    recentCap: 500,
    earlyNoiseBoostDecisions: 200,
  }
}
```

- `eta`: learning rate for the diagonal A/b update. Lower if weights swing; raise if learning feels too slow.
- `emaBeta` / `iBeta`: smoothing for admit EMA & integral error—smaller = smoother, larger = snappier.
- `priorWeights`: prior belief for features. Adjust if you know some attrs should start favored.
- `weightClamp` / `floors`: prevent sign flips and runaway weights.
- `initRecentValues`: seeds the score buffer the threshold uses; keep near plausible early scores.
- `earlyNoiseBoostDecisions`: how long to keep extra score noise for exploration.

### 5.3 Thresholding & controller

```ts
{
  THRESH: {
    sigmaFloor: 0.4,
    warmupDecisions: 300,
    warmupErrCap: 0.25,
    floorBumpBase: 0.2,
    floorBumpSlope: 1.25,
    capacityBiasScale: { early: 0.4, late: 0.8 },
    urgencyMax: 2.5,
    ctrlGains: { kP: 1.6, kI: 0.6, boostEdge: 0.2, boostFactor: 1.35 },
  }
}
```

- `sigmaFloor`: prevents the threshold from getting too tight (especially early); raise if you see choppy admits.
- `warmupDecisions` / `warmupErrCap`: more forgiving floor during early learning.
- `capacityBiasScale`: adds a bias proportional to seats used; larger = stricter later.
- `urgencyMax`: caps how much feasibility urgency can lower the threshold.
- `ctrlGains` (kP/kI): PI controller.

If admit rate lags target persistently → bump `kI` slightly.

If admit rate oscillates → reduce kP (and/or sigmaFloor up a bit).

### 5.4 Gates (hard policy)

```ts
{
  GATES: {
    pace: { minUsed: 0.35, gap: 0.16, penalty: -1.5 },
    infeasibleCutoff: { earlyOffUntil: 0.35, mid: 2.2, late1: 1.6, late2: 1.2 },
    reserve: { minUsed: 0.75, buffer: 0.95, penalty: -2 },
    top1MustHave: { minUsed: 0.7, minRatio: 1.6, dominance: 1.3, penalty: -2 },
    lateFeas: { minUsed: 0.82, maxRatio: 1.2, helpAtRiskMin: 2, penalty: -2 },
  }
}
```

- `pace`: when behind on a constraint by gap, and `usage` > `minUsed`, reject non-helpers. Raise gap to be stricter.
- `infeasibleCutoff`: phase-based feasibility cut. Lower numbers = more aggressive rejections of non-helpers.
- `reserve`: late-only seat reservation; stronger buffer holds back more seats.
- `top1MustHave`: when one constraint dominates, require it; lower minRatio / dominance to trigger earlier.
- `lateFeas`: late admits must help enough at-risk tracks; relax if you’re starving.

### 5.5 Reward shaping (teaches “what’s good?”)

```ts
{
  REWARD: {
    baseCreative: 3.4,
    baseLocal: 2.0,
    baseDefault: 0.8,
    feasBoost: { slope: 0.65, base: 0.75, max: 2.2 },
    scarcityClamp: [0.5, 2.0] as const,
    mcBoost: { slope: 0.5, max: 2.2 },
    pace: { margin: 0.04, kUnder: 2.2, kOver: 1.6, clamp: [0.2, 2.0] as const },
    lateGuardrail: { minUsed: 0.5, penaltyIfUsefulLE1: 1.4 },
    rejectPenalty: { early: -0.15, late: -0.5, step: 0.12, extraLate: 0.08, min: -1.0 },
    clamp: [-2, 6] as const,
  }
}
```

- `base*:**` baseline utility per attribute when unmet. Increase for tracks you want prioritized.
- `feasBoost` / `mcBoost`: extra juice when feasibility looks risky; raise to be more constraint-focused.
- `pace`: rewards helping tracks you’re behind on; raise kUnder if pace catching is too slow.
- `lateGuardrail`: discourages low-leverage admits late. Lower if it blocks too much.
- `rejectPenalty`: small negatives prevent over-rejecting; make less negative to be pickier.

### 5.6 Fill / Finish (endgame helpers)

```ts
{
  FILL: { enableAtUsed: 0.9, learn: false }
  FINISH: { enableAtUsed: 0.9, maxShortfall: 10, ratioMin: 3 }
}
```

- `FILL`: once all constraints are met and usage ≥ enableAtUsed, admit to capacity.
- `learn=false` avoids overshoot penalties from continued training on satisfied tracks.
- `FINISH`: if constraints remain unmet but smallest shortfall ≤ maxShortfall or remainingSlots / smallestShortfall ≥ ratioMin, then admit any person who helps any unmet constraint.

If you still get “stuck 5–10 short”, raise maxShortfall or lower ratioMin.

### 5.7 Learning warmup / Exploration

```ts
{
  WARMUP: { usedMax: 0.15, minEma: 0.02, negClampAfter: -0.3 }
  EXPLORE: { epsAdmit: 0.08, epsUntilUsed: 0.05 }
}
```

- `WARMUP`: during early usage or low admit-rate EMA, skip/soften negative updates to prevent early pessimism.
- `EXPLORE`: very early, occasionally admit a helpful candidate at random to break stalemates.

### 5.8 Debug

```ts
{
  DEBUG: { keepLastLogs: 2, includeThresholdBlock: false }
}
```

- `keepLastLogs`: number of decision lines to keep in lastLogs.
- `includeThresholdBlock`: if true, `getProgress()` includes the full threshold debug block (median, sigma, capacity bias, controller terms, etc.).

### 6. Interpreting outputs & logs

- `admitRate`: EMA of admits (not instantaneous %).
- `threshold` / `lastRawValue`: if `score < threshold` repeatedly late, your controller/threshold/capacity bias may be too strict—see tuning above.

- `feasibility.maxRatio` near 0 with unmet constraints means your endgame helpers should kick in; adjust `FINISH` if not.
- `banditStats.weights` show the first 6 weights (indicators & part of progress). Progress weights are clamped ≤ progressFloor (negative).
- `lastLogs`: quick sense of whether admits were useful (which attrs) and the reward behind each decision.

### 7. Troubleshooting

Stuck near the end needing 1–10 of a single attribute:

- Increase `FINISH.maxShortfall` (e.g., 10→15) or reduce `FINISH.ratioMin` (e.g., 3→2.5).
- Relax `GATES.infeasibleCutoff.late2` slightly (e.g., 1.2→1.3) so non-critical helpers slip through when appropriate.

If threshold seems too high late, lower `THRESH.capacityBiasScale.late` or reduce `ctrlGains.kP`.

Run out of seats early:

- Lower `TARGET_RATE_BASE.early`, increase `TARGET_RISK_PULL.slope/max`.
- Increase `THRESH.capacityBiasScale.early`.
- Increase `REJECT` penalties a bit (be pickier) if you want fewer “meh” admits early.

Oscillating admit rate (choppy behavior):

- Increase `THRESH.sigmaFloor`.
- Reduce `ctrlGains.kP` slightly; if lagging then add a touch of `kI`.
- Decrease `BANDIT.eta` if weights bounce.

Model seems slow to learn:

- Increase `BANDIT.eta` a bit or start with stronger `priorWeights`.
- Consider slightly larger `EXPLORE.epsAdmit` early.

8. Quickstart

```ts
import { BanditBouncer } from './bandit-bouncer'
// state: GameState, config: { MAX_CAPACITY, ... } provided by your harness

const bouncer = new BanditBouncer(state, config)
await bouncer.initializeLearningData()

while (game.status === 'running') {
  const admitted = bouncer.admit(game)
  // ... advance simulation
  if (shouldLog) console.log(bouncer.getProgress())
}

const result = bouncer.getOutput()
console.log(result.summary)
```

### 9. Notes on persistence

- Each run saves a compact summary plus a full game state with decisions and a bandit snapshot (`A`, `b`, `weights`, recent score buffer size, etc.).
- New games optionally warm start from recent history or restore from snapshot if present.
