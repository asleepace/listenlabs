# Bandit Bouncer

A contextual bandit that learns a scoring function to admit the right people to satisfy the min-counts (constraints) with minimal waste, while a controller paces the admit rate so you don’t run out of seats too early or finish too late.

## What it’s optimizing (behaviorally)

1. **Primary goal**: meet all attribute minimums (e.g., `techno_lover ≥ 650`, `local ≥ 750`) before capacity runs out.
2. **Secondary goal**: keep rejections low (your "finalScore" is `totalRejected`), i.e., be selective but not too selective.
3. **Scheduling goal**: track a target admit-rate curve (higher early, lower late) so you don’t blow through seats or starve the queue.

None of this is written as one math objective; instead, it’s achieved by the reward shaping + gates + controller below.

## What it learns

A linear value model: `value = w · features` (contextual bandit).

### Features include:

- Unmet constraint indicators (1 if this person has an attribute that’s still short).
- Progress per constraint (now using progress ∈ [0,1] so negative weights penalize already-met tracks).
- Capacity used (to nudge behavior as seats run down).

A scarcity proxy (e.g., for creative).

Weights update online from admitted decisions using a ridge-like diagonal update (the A/b arrays). Rejections also train (after warmup) with a clamped negative update so the model doesn’t over-learn from “noisy” negatives early.

### Clamps keep certain weights sane:

- Progress weights are clamped ≤ progressFloor (negative) so “already met” gets penalized.
- Capacity weight ≤ capacityFloor (≤ 0) so being late doesn’t inflate value.
- Scarcity weight ≥ 0 so scarcity can only help.

### How it decides (per person)

1. Policy gates (no learning) protect feasibility:
2. Pace gate: don’t admit people who don’t help the most-behind constraint when you’re lagging.
3. Reserve/late-feasibility/top-1-must-have: keep seats for critical needs late.
4. Infeasible-cutoff bands: reject if someone doesn’t help when the feasibility ratio is too high for the phase.

If not blocked: Bandit score w·x vs an adaptive threshold:

> Threshold = robust quantile (median + z\*sigma) of recent scores

capacity bias (depends on seats used)

PI controller term (pushes toward the target admit rate)
− urgency (if constraints look risky).

The admit decision is `score + small_noise > threshold`.

## Finishers (late only):

If all constraints are met and you’re late: fill seats (optionally without learning).

If some constraints are just short, admit anyone who helps any eligible unmet one (absolute `shortfall ≤ maxShortfall` or roomy by ratio).

## What the reward is teaching

When you admit, the reward is larger if the person:

Helps an unmet constraint, especially the most-critical one (`mcBoost`).

Helps when a constraint is scarce or feasibility ratio is high.

Helps the pace (you’re behind vs ahead on that track).

When you reject, the reward is a small negative (scaled by feasibility pressure & phase). That discourages needless rejections but avoids pushing admit-rate too high.

## There’s also:

Overshoot penalty if you keep admitting an already-satisfied attribute.

Late guardrail if someone helps ≤1 unmet track late (to avoid wasting seats on low leverage admits).

The controller & target admit rate

Target admit rate = phase schedule (early/mid/late) minus a “risk pull” if feasibility looks bad.

A PI controller compares the admit EMA to target and nudges the threshold so actual admits track the plan.

## Big picture

Learning: how much each context feature correlates with “useful” admits under your shaped reward.

Optimizing for: “meet all min-counts reliably with as few rejections as practical, while pacing admits so the endgame is feasible,” not a single closed-form objective, but a practical blend for this simulation.

If you ever want to bias the system differently, change reward terms (that’s your objective), and the bandit will re-learn weights to match it. The gates are your “hard policy” safety rails, and the controller is your pacing knob.
