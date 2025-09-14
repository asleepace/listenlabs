/** @file scoring.ts */

import type { GameConstraints, GameState, ScenarioAttributes } from '../types'

export type Correlation = GameState['game']['attributeStatistics']['correlations'][string]
export type Quota = ReturnType<typeof createQuota>
export type Guest = ReturnType<ReturnType<typeof initializeScoring>['forGuest']>

export type ScoringConfig = {
  maxRejections: number
  maxAdmissions: number
  targetRejections?: number // default 5,000 (target number of rejections to complete by)
  weights?: {
    shortfallPerHead?: number // default 2.0
    overagePerHead?: number // default 0.5
    unmetQuotaConstant?: number // default 100
    targetPerPerson?: number // default 1.0  (penalize people seen past target)
    targetBonusCap?: number // default 1000 (cap reward for finishing early)
    targetPenaltyCap?: number // default 5000 (cap penalty for finishing late)
  }
}

export const average = (values: number[]): number => {
  if (!values.length) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

export const diff = (v1: number, v2: number) => {
  return (v1 - v2) / 2
}

export const clamp = (lo: number, val: number, hi: number): number => {
  return Math.max(lo, Math.min(val, hi))
}

export const sum = (vals: number[]): number => {
  return vals.reduce((a, b) => a + b, 0) || 0
}

/**
 *  Extract only attributes attributes as an array.
 */
export const getAttributes = (scenarioAttributes: ScenarioAttributes): string[] => {
  return Object.entries(scenarioAttributes)
    .filter(([key, value]) => value)
    .map(([key]) => key)
}

/**
 *  Creates a quota which is used to track overall progress.
 */
export function createQuota(constraint: GameConstraints, frequency: number, correlations: Correlation) {
  return {
    attribute: constraint.attribute,
    minCount: constraint.minCount,
    count: 0,
    frequency,
    correlations,
    relativeProgress(peopleLeftInLine: number): number {
      if (this.isComplete()) return 1.0
      const expectedRemaining = Math.max(1, peopleLeftInLine * Math.max(0, this.frequency))
      return this.needed() / expectedRemaining
    },
    progress() {
      return this.count / this.minCount
    },
    needed(): number {
      if (this.count >= this.minCount) return 0
      return this.minCount - this.count
    },
    isComplete() {
      return this.count >= this.minCount
    },
    inProgress() {
      return this.minCount > this.count
    },
  }
}

/**
 *  Create a scoring object which is used to track quotas, guests, and progress.
 */
export function initializeScoring(game: GameState['game'], config: ScoringConfig) {
  const correlations = game.attributeStatistics.correlations
  const frequencies = game.attributeStatistics.relativeFrequencies

  // target threshold we want to hit all quotas by
  const TARGET_REJECTIONS = config.targetRejections ?? 5_000

  const quotas = Object.fromEntries(
    game.constraints.map((constraint) => {
      const attr = constraint.attribute
      const attributeFreq = frequencies[attr]
      const attributeCorr = correlations[attr]
      return [attr, createQuota(constraint, attributeFreq, attributeCorr)] as const
    })
  )

  // --- local helpers for scoring ---
  const pw = {
    shortfallPerHead: 2.0,
    overagePerHead: 0.5,
    unmetQuotaConstant: 100,
    targetPerPerson: 1.0,
    targetBonusCap: 1000,
    targetPenaltyCap: 5000,
    ...(config.weights ?? {}),
  }

  /** Per-quota urgency = deficit / expected arrivals with that attr in the remaining line mass. */
  const quotaUrgency = (q: Quota, peopleLeftInLine: number): number => {
    if (!q.inProgress()) return 0
    const expected = Math.max(1, peopleLeftInLine * Math.max(0, q.frequency))
    return q.needed() / expected // 1.0 = exactly on pace, >1 behind, <1 ahead
  }

  /** Pair bonus when guest hits two unmet quotas that are negatively correlated (rarer combo). */
  const pairBonus = (attrs: string[], peopleLeftInLine: number): number => {
    if (attrs.length < 2) return 0
    let bonus = 0
    for (let i = 0; i < attrs.length; i++) {
      for (let j = i + 1; j < attrs.length; j++) {
        const a = attrs[i],
          b = attrs[j]
        const corr = quotas[a]?.correlations?.[b] ?? 0
        if (corr < -0.15) {
          // weight by combined urgency; scale modestly so it’s a nudge not a flood
          const ua = quotaUrgency(quotas[a], peopleLeftInLine)
          const ub = quotaUrgency(quotas[b], peopleLeftInLine)
          bonus += -corr * (ua + ub) * 0.25
        }
      }
    }
    return bonus
  }

  /** Seat scarcity in [0..1]: closer to 1 when seats are getting scarce. */
  const seatScarcity = (seatsLeft: number): number => {
    // start tightening once under ~200 seats; feel free to tune this
    return clamp(0, 200 / Math.max(1, seatsLeft), 1)
  }

  return {
    // treat this as *line size*; it’s fine to set equal to the reject cap if you like
    peopleInLine: config.maxRejections,
    admitted: 0,
    rejected: 0,
    get correlations() {
      return correlations
    },
    get frequencies() {
      return frequencies
    },
    unmetQuotasCount(): number {
      return this.quotas().length
    },

    /** Worst expected shortfall given people left, optionally adding this guest. */
    worstExpectedShortfall(peopleLeftInLine: number, guest?: ScenarioAttributes): number {
      let worst = 0
      for (const q of this.quotas()) {
        const cur = q.count + (guest && guest[q.attribute] ? 1 : 0)
        const expectedFuture = Math.max(0, peopleLeftInLine) * Math.max(0, q.frequency)
        const gap = Math.max(0, q.minCount - (cur + expectedFuture)) // shortfall after all remaining arrivals
        if (gap > worst) worst = gap
      }
      return worst
    },

    /** Seat scarcity in [0..1]; higher when seats are scarce. */
    seatScarcity(): number {
      const seats = Math.max(1, this.getTotalSpotsAvailable())
      // start tightening under ~200 seats (tunable)
      return Math.min(1, 200 / seats)
    },
    // keep your existing constants + pw as-is

    getLossScore() {
      let unmet = 0
      const terms: number[] = []

      for (const q of Object.values(quotas)) {
        const gap = Math.abs(q.count - q.minCount)
        if (q.isComplete()) {
          // small penalty for overfilling
          terms.push(gap * pw.overagePerHead)
        } else {
          unmet++
          // larger penalty for being short
          terms.push(gap * pw.shortfallPerHead)
        }
      }

      // If all met: use average overage; else: sum + flat per-unmet penalty
      const completionPenalty = unmet === 0 ? average(terms) : sum(terms) + pw.unmetQuotaConstant * unmet

      // Rejection-based term (positive if beyond target, negative if earlier)
      // (This matches "target rejections" semantics.)
      const overUnder = (this.rejected - TARGET_REJECTIONS) * pw.targetPerPerson

      // Optional guard: don’t reward early finishing unless all quotas are met
      // const gatedOverUnder = unmet === 0 ? overUnder : Math.max(0, overUnder)

      // Cap so this term can't explode
      let delta = overUnder
      if (delta < 0) delta = Math.max(delta, -pw.targetBonusCap)
      else delta = Math.min(delta, pw.targetPenaltyCap)

      // Total penalty-style score (lower is better)
      return completionPenalty + delta
    },

    /**
     * Single, conflict-free decision:
     * 1) If there’s any expected shortfall, admit only if this guest *reduces* the worst shortfall
     *    by at least a scarcity-weighted amount.
     * 2) If no shortfall remains, use your heuristic (score/fraction) to keep the mix tight.
     */
    shouldAdmit(guest: ScenarioAttributes, baseTheta = 1.0, baseFrac = 0.5): boolean {
      const peopleLeft = this.getPeopleLeftInLine()
      const seatsLeft = this.getTotalSpotsAvailable()

      // If no seats, trivially false
      if (seatsLeft <= 0) return false

      const before = this.worstExpectedShortfall(peopleLeft)
      const after = this.worstExpectedShortfall(Math.max(0, peopleLeft - 1), guest)
      const delta = before - after // improvement in worst gap (>=0 is non-worsening)

      if (before > 0) {
        // We still have expected shortfall: require *meaningful* improvement.
        const tighten = 0.15 + 0.35 * this.seatScarcity() // tunable threshold
        return delta >= tighten
      }

      // No shortfall expected → use your heuristics
      const byScore = this.admitByScore(guest, baseTheta)
      const byFraction = this.admitByFraction(guest, baseFrac)
      return byScore || byFraction
    },

    /** how many of the unmet quotas this guest hits */
    guestHitCount(guest: ScenarioAttributes): number {
      let k = 0
      for (const q of this.quotas()) if (guest[q.attribute]) k++
      return k
    },

    /** fraction of unmet quotas this guest hits (0..1) */
    guestFractionHit(guest: ScenarioAttributes): number {
      const total = this.unmetQuotasCount()
      if (total === 0) return 1
      return this.guestHitCount(guest) / total
    },

    /** Numeric guest score built from urgencies + rare-combo bonus. */
    guestScore(guest: ScenarioAttributes): number {
      const attrs = getAttributes(guest).filter((a) => quotas[a].inProgress())
      if (!attrs.length) return 0

      const peopleLeft = this.getPeopleLeftInLine()
      // sum of urgencies for each unmet quota the guest hits
      let score = 0
      for (const a of attrs) score += quotaUrgency(quotas[a], peopleLeft)

      // add pair bonus for neg-correlated combos
      score += pairBonus(attrs, peopleLeft)

      return score
    },

    /**
     * Admit if the guest covers at least `baseFrac` of the currently unmet quotas.
     * We keep your rule, but also provide a seat/urgency-aware numeric path (`admitByScore` below).
     */
    admitByFraction(guest: ScenarioAttributes, baseFrac: number): boolean {
      const totalQuotas = this.quotas()
      if (totalQuotas.length === 0) return true

      const seats = Math.max(1, this.getTotalSpotsAvailable())
      const scarcity = seatScarcity(seats)

      // “how we’re pacing seats” vs “how we’re pacing quotas”
      const seatProgress = this.getTotalProgress() // seats used ratio
      const quotaProgress = this.getQuotaProgress() // 1.0 = on pace; >1 behind
      // map these to [-1,1] intuition: positive => ahead on quotas vs seats, negative => behind
      const progressDelta = clamp(-1, 1 - quotaProgress - seatProgress, 1)

      // tighten threshold if seats are scarce *and* we’re behind
      const tighten = 0.15 * scarcity * clamp(0, -progressDelta, 1)
      const needFrac = Math.min(0.95, baseFrac + tighten)

      return this.guestFractionHit(guest) >= needFrac
    },
    /**
     * Admit using numeric urgency score.
     * Base idea: one “strong” unmet quota (urgency≈1) should usually be enough early,
     * but as seats get scarce, demand either higher urgency or multiple hits.
     */
    admitByScore(guest: ScenarioAttributes, baseTheta = 1.0): boolean {
      const seats = Math.max(1, this.getTotalSpotsAvailable())
      const scarcity = seatScarcity(seats)

      // compute score
      const score = this.guestScore(guest)

      // dynamic threshold: raise it when seats are scarce or quotas are still badly behind
      const quotaBehind = Math.max(0, this.getQuotaProgress() - 1.0) // 0 when on pace/ahead; grows if behind
      const theta = baseTheta + 0.6 * scarcity + 0.4 * Math.min(1, quotaBehind)

      return score >= theta
    },
    isComplete() {
      return this.quotas().length === 0
    },
    /** Returns true if we are below the max admissions and rejections. */
    inProgress() {
      return this.admitted < config.maxAdmissions && this.rejected < config.maxRejections
    },
    /** Returns the total number people we have seen and processed. */
    getTotalPeopleSeen() {
      return this.admitted + this.rejected
    },
    /** returns the total number of people left in line. */
    getPeopleLeftInLine(): number {
      return Math.max(0, config.maxRejections - this.getTotalPeopleSeen())
    },
    /** Returns the total number of spots availble in the venue. */
    getTotalSpotsAvailable() {
      return Math.max(0, config.maxAdmissions - this.admitted)
    },
    /** Seats progress (0..1) — fraction of capacity already used. */
    getTotalProgress() {
      return (config.maxAdmissions - this.getTotalSpotsAvailable()) / Math.max(1, config.maxAdmissions)
    },
    /**
     * Quota progress averaged across unmet quotas:
     * 1.0 = on pace (expected remaining arrivals cover all deficits),
     * >1 behind pace, <1 ahead of pace.
     */
    getQuotaProgress() {
      const peopleLeftInLine = this.getPeopleLeftInLine()
      const vals = this.quotas().map((q) => clamp(0, q.relativeProgress(peopleLeftInLine), 3))
      return average(vals) || 1.0
    },
    /** Returns true when finished meeting quotas. */
    isFinishedWithQuotas(): boolean {
      return this.quotas().length === 0
    },
    /** Returns an array of quotas which are currently in progress. */
    quotas() {
      return Object.values(quotas).filter((quota) => quota.inProgress())
    },
    /** Returns the specific quota for the attribute. */
    get(attr: string): Quota {
      return quotas[attr]
    },
    /** Returns a helper object for common interactions with guests. */
    forGuest(guest: ScenarioAttributes) {
      return {
        attributes: getAttributes(guest)
          .filter((attr) => quotas[attr].inProgress())
          .sort((a, b) => quotas[a].progress() - quotas[b].progress()),
        count() {
          return this.attributes.length
        },
        hasEveryAttribute: (): boolean => {
          return Object.values(quotas)
            .filter((q) => q.inProgress())
            .every((q) => guest[q.attribute])
        },
        hasSomeAttribute: (): boolean => {
          return Object.values(quotas)
            .filter((q) => q.inProgress())
            .some((q) => guest[q.attribute])
        },
      }
    },

    /** Returns the max number of people used for any of the quotas. */
    getMaxPeopleNeeded() {
      const arr = this.quotas().map((quota) => quota.needed())
      return arr.length ? Math.max(...arr) : 0
    },
    /** Callback which updates the internal state counters. */
    update({ guest, admit }: { guest: ScenarioAttributes; admit: boolean }): boolean {
      if (admit === false) {
        this.rejected++
        return admit
      }
      // update individual quota counts
      for (const attr in quotas) {
        if (!guest[attr]) continue
        quotas[attr].count++
      }
      // update global admitted count
      this.admitted++
      return admit
    },
  }
}
