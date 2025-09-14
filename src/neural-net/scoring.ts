/** @file scoring.ts */

import type { GameConstraints, GameState, ScenarioAttributes } from '../types'

export type Correlation = GameState['game']['attributeStatistics']['correlations'][string]
export type Quota = ReturnType<typeof createQuota>
export type Guest = ReturnType<ReturnType<typeof initializeScoring>['forGuest']>

export type ScoringConfig = {
  maxRejections: number
  maxAdmissions: number
  targetRejections?: number // default 5,000 (target number of rejections to complete by)
  safetyCushion?: number // NEW: heads of buffer to pace against (default 1)
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

// NOTE: in this file we use clamp(lo, val, hi) because many call sites read naturally as clamp(0, value, 1)
export const clamp = (lo: number, val: number, hi: number): number => {
  return Math.max(lo, Math.min(val, hi))
}

export const sum = (vals: number[]): number => vals.reduce((a, b) => a + b, 0) || 0

/** Extract true attributes as an array. */
export const getAttributes = (scenarioAttributes: ScenarioAttributes): string[] => {
  return Object.entries(scenarioAttributes)
    .filter(([_, value]) => value)
    .map(([key]) => key)
}

/** Creates a quota tracker. */
export function createQuota(
  constraint: GameConstraints,
  frequency: number,
  correlations: Correlation,
  cushion: number = 0
) {
  return {
    attribute: constraint.attribute,
    minCount: constraint.minCount,
    count: 0,
    frequency,
    correlations,
    relativeProgress(peopleLeftInLine: number) {
      if (this.isComplete()) return 1.0
      const expectedRemaining = Math.max(1, peopleLeftInLine * Math.max(0, this.frequency))
      const need = Math.max(0, this.minCount + cushion - this.count)
      return need / expectedRemaining
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

/** Create a scoring object used to track quotas, guests, and progress. */
export function initializeScoring(game: GameState['game'], config: ScoringConfig) {
  const correlations = game.attributeStatistics.correlations
  const frequencies = game.attributeStatistics.relativeFrequencies

  const TARGET_REJECTIONS = config.targetRejections ?? 5_000
  const CUSHION = config.safetyCushion ?? 1 // you already pass safetyCushion: 1
  const CUSHION_PER_QUOTA = 4 // “4 person cushion” per unmet quota
  const BREATH_MULTIPLIER = 8 // scales when to start getting strict

  const quotas = Object.fromEntries(
    game.constraints.map((constraint) => {
      const attr = constraint.attribute
      const attributeFreq = frequencies[attr]
      const attributeCorr = correlations[attr]
      return [attr, createQuota(constraint, attributeFreq, attributeCorr, CUSHION)] as const
    })
  )

  // penalty weights
  const pw: NonNullable<ScoringConfig['weights']> = {
    shortfallPerHead: 3.0, // ↑ was 2.0
    overagePerHead: 0.25, // ↓ was 0.5
    unmetQuotaConstant: 200, // ↑ was 100
    targetPerPerson: 2.0,
    targetBonusCap: 1000,
    targetPenaltyCap: 5000,
    ...(config.weights ?? {}),
  }

  /** Per-quota urgency = deficit / expected arrivals with that attr in the remaining line mass. */
  const quotaUrgency = (q: Quota, peopleLeftInLine: number): number => {
    if (!q.inProgress()) return 0
    const expected = Math.max(1, peopleLeftInLine * Math.max(0, q.frequency))
    const need = Math.max(0, q.minCount + CUSHION - q.count)
    return need / expected
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
          const ua = quotaUrgency(quotas[a], peopleLeftInLine)
          const ub = quotaUrgency(quotas[b], peopleLeftInLine)
          bonus += -corr * (ua + ub) * 0.25
        }
      }
    }
    return bonus
  }

  return {
    peopleInLine: config.maxRejections,
    admitted: 0,
    rejected: 0,
    get correlations() {
      return correlations
    },
    get frequencies() {
      return frequencies
    },
    /**
     * Start tightening admits when projected demand risks exceeding seats left.
     * - If only one quota remains → allow a 1-seat slack (off-by-one guard).
     * - Otherwise, start being stricter roughly ~200 seats before the end when all 4 quotas are unmet:
     *   breathingRoom = CUSHION_PER_QUOTA (5) * unmetQuotas * BREATH_MULTIPLIER (10)
     *   => 5 * 4 * 10 = 200
     */
    isRunningOutOfAvailableSpots(): boolean {
      if (this.isComplete()) return false

      const totalUnmet = this.unmetQuotasCount()
      const spotsReserved = this.getMaxPeopleNeeded()
      const totalSpots = this.getTotalSpotsAvailable()

      if (totalUnmet < 2) {
        // one quota left → allow off-by-one tolerance
        return spotsReserved + 1 >= totalSpots
      }

      const breathingRoom = CUSHION * CUSHION_PER_QUOTA * BREATH_MULTIPLIER * totalUnmet
      return spotsReserved + breathingRoom >= totalSpots
    },
    unmetQuotasCount(): number {
      return this.quotas().length
    },
    worstExpectedShortfall(peopleLeftInLine: number, guest?: ScenarioAttributes): number {
      let worst = 0
      for (const q of this.quotas()) {
        const cur = q.count + (guest && guest[q.attribute] ? 1 : 0)
        const expectedFuture = Math.max(0, peopleLeftInLine) * Math.max(0, q.frequency)
        const target = q.minCount // ✅ no cushion in the gate
        const gap = Math.max(0, target - (cur + expectedFuture))
        if (gap > worst) worst = gap
      }
      return worst
    },

    /** Seat scarcity (recomputed from current seats left). */
    seatScarcity(): number {
      const seats = Math.max(1, this.getTotalSpotsAvailable())
      return Math.min(1, 120 / seats)
    },

    /** Penalty-style score (lower is better) you can log or use for shaping. */
    getLossScore() {
      let unmet = 0
      const terms: number[] = []

      for (const q of Object.values(quotas)) {
        const gap = Math.abs(q.count - q.minCount)
        if (q.isComplete()) {
          terms.push(gap * pw.overagePerHead!) // mild overage cost
        } else {
          unmet++
          terms.push(gap * pw.shortfallPerHead!) // stronger shortfall cost
        }
      }

      const completionPenalty = unmet === 0 ? average(terms) : sum(terms) + pw.unmetQuotaConstant! * unmet

      const peopleSeen = this.getTotalPeopleSeen()
      let delta = (peopleSeen - TARGET_REJECTIONS) * pw.targetPerPerson!
      if (delta < 0) delta = Math.max(delta, -pw.targetBonusCap!)
      else delta = Math.min(delta, pw.targetPenaltyCap!)

      return completionPenalty + delta
    },

    /**
     * Conflict-free decision:
     * 1) If any expected shortfall remains, admit only if this guest reduces the worst shortfall
     *    by a scarcity-weighted amount.
     * 2) If no shortfall remains, use heuristic (score/fraction) to keep mix tight.
     */
    shouldAdmit(guest: ScenarioAttributes, baseTheta = 1.0, baseFrac = 0.5): boolean {
      const peopleLeft = this.getPeopleLeftInLine()
      const seatsLeft = this.getTotalSpotsAvailable()
      if (seatsLeft <= 0) return false

      const before = this.worstExpectedShortfall(peopleLeft)
      const after = this.worstExpectedShortfall(Math.max(0, peopleLeft - 1), guest)
      const delta = before - after // improvement (heads of shortfall reduced)

      // optional tiny debug remains...
      if (before > 0) {
        const scarcity = this.seatScarcity()
        // ↓ require smaller improvement; earlier it was 0.05 + 0.25*scarcity
        const tighten = 0.02 + 0.15 * scarcity
        return delta >= tighten * Math.min(1, before)
      }

      const byScore = this.admitByScore(guest, baseTheta)
      const byFraction = this.admitByFraction(guest, baseFrac)
      return byScore || byFraction
    },
    /** returns the current counts of each quota */
    getCounts(): Record<string, number> {
      return Object.fromEntries(Object.entries(quotas).map(([key, value]) => [key, value.count]))
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
      let score = 0
      for (const a of attrs) score += quotaUrgency(quotas[a], peopleLeft)
      score += pairBonus(attrs, peopleLeft)
      return score
    },

    /** Admit if guest covers ≥ baseFrac of current unmet quotas; tighten if scarce/behind. */
    admitByFraction(guest: ScenarioAttributes, baseFrac: number): boolean {
      const totalQuotas = this.quotas()
      if (totalQuotas.length === 0) return true

      const seats = Math.max(1, this.getTotalSpotsAvailable())
      const scarcity = this.seatScarcity()

      const seatProgress = this.getTotalProgress()
      const quotaProgress = this.getQuotaProgress() // 1.0 = on pace; >1 behind
      const progressDelta = clamp(-1, 1 - quotaProgress - seatProgress, 1)

      const tighten = 0.15 * scarcity * clamp(0, -progressDelta, 1)
      const needFrac = Math.min(0.95, baseFrac + tighten)

      return this.guestFractionHit(guest) >= needFrac
    },

    /** Admit using numeric urgency score with dynamic threshold. */
    admitByScore(guest: ScenarioAttributes, baseTheta = 1.0): boolean {
      const seats = Math.max(1, this.getTotalSpotsAvailable())
      const scarcity = this.seatScarcity()
      const score = this.guestScore(guest)
      const quotaBehind = Math.max(0, this.getQuotaProgress() - 1.0)
      const theta = baseTheta + 0.6 * scarcity + 0.4 * Math.min(1, quotaBehind)
      return score >= theta
    },
    /** returns if all quotas have been met and we've hit 1,000 admissions. */
    isComplete() {
      return this.quotas().length === 0
    },
    inProgress() {
      return this.admitted < config.maxAdmissions && this.rejected < config.maxRejections
    },
    getTotalPeopleSeen() {
      return this.admitted + this.rejected
    },
    getPeopleLeftInLine(): number {
      return Math.max(0, config.maxRejections - this.getTotalPeopleSeen())
    },
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
    isFinishedWithQuotas(): boolean {
      return this.quotas().length === 0
    },
    quotas() {
      return Object.values(quotas).filter((quota) => quota.inProgress())
    },
    get(attr: string): Quota {
      return quotas[attr]
    },
    forGuest(guest: ScenarioAttributes) {
      return {
        attributes: getAttributes(guest)
          .filter((attr) => quotas[attr].inProgress())
          .sort((a, b) => quotas[a].progress() - quotas[b].progress()),
        count() {
          return this.attributes.length
        },
        hasEveryAttribute: (): boolean => {
          return this.quotas().every((quota) => guest[quota.attribute])
        },
        hasSomeAttribute: (): boolean => {
          return this.quotas().some((quota) => guest[quota.attribute])
        },
      }
    },
    getMaxPeopleNeeded() {
      const arr = this.quotas().map((quota) => quota.needed())
      return arr.length ? Math.max(...arr) : 0
    },
    update({ guest, admit }: { guest: ScenarioAttributes; admit: boolean }): boolean {
      if (admit === false) {
        this.rejected++
        return admit
      }
      for (const attr in quotas) {
        if (!guest[attr]) continue
        quotas[attr].count++
      }
      this.admitted++
      return admit
    },
  }
}
