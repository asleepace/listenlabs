/** @file scoring.ts */

import type { GameConstraints, GameState, ScenarioAttributes } from '../types'
import { Conf } from './config'
import { average, sum, clamp, getAttributes } from './util'

export type Correlation = GameState['game']['attributeStatistics']['correlations'][string]
export type Quota = ReturnType<typeof createQuota>
export type Scoring = ReturnType<typeof initializeScoring>
export type Guest = ReturnType<Scoring['forGuest']>

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
    getSummary() {
      return {
        attribute: constraint.attribute,
        current: this.count,
        required: constraint.minCount,
        satisfied: this.isComplete(),
      } as const
    },
  }
}

/** Create a scoring object used to track quotas, guests, and progress. */
export function initializeScoring(game: GameState['game'], config: ScoringConfig) {
  const correlations = game.attributeStatistics.correlations
  const frequencies = game.attributeStatistics.relativeFrequencies

  const TARGET_REJECTIONS = config.targetRejections ?? Conf.TARGET_REJECTIONS
  const CUSHION = config.safetyCushion ?? Conf.SAFETY_CUSHION // you already pass safetyCushion: 1
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
    /** returns the total number of people seen (admitted + rejected). */
    get nextIndex() {
      return this.admitted + this.rejected
    },
    /**
     * Returns true when the venue has more spots available than the sum of all people
     * needed to meet every quota.
     */
    isUnderFilled(): boolean {
      return this.getTotalSpotsAvailable() > this.getMaxPeopleNeeded()
    },
    /**
     * Returns true when the venue only has enough available spots for the quota with
     * the max number of people needed (no cushion).
     */
    isOverFilled(): boolean {
      return this.getTotalSpotsAvailable() <= this.getMinPeopleNeeded()
    },
    /**
     * Start tightening admits when projected demand risks exceeding seats left (simplified.)
     */
    isRunningOutOfAvailableSpots(): boolean {
      if (this.isComplete()) return false
      if (this.isUnderFilled()) return false

      const totalUnmet = this.unmetQuotasCount()
      const spotsReserved = this.getMinPeopleNeeded()
      const totalSpots = this.getTotalSpotsAvailable()

      if (totalUnmet === 1) {
        // one quota left → allow off-by-one tolerance
        return spotsReserved >= totalSpots
      }

      // Replace the current return with this:
      const breathingRoom = Math.min(100, CUSHION * CUSHION_PER_QUOTA * BREATH_MULTIPLIER * totalUnmet)
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
        const target = q.minCount + CUSHION // <— was q.minCount
        const gap = Math.max(0, target - (cur + expectedFuture))
        if (gap > worst) worst = gap
      }
      return worst
    },

    /** Seat scarcity (recomputed from current seats left). */
    seatScarcity(): number {
      const seats = Math.max(1, this.getTotalSpotsAvailable())
      const k = Math.max(60, Math.min(180, Math.round(0.15 * config.maxAdmissions))) // ~15% of capacity
      return Math.min(1, k / seats)
    },

    /** Penalty-style score (lower is better) you can log or use for shaping. */
    getLossScore() {
      let unmet = 0
      const terms: number[] = []
      for (const q of Object.values(quotas)) {
        const gap = Math.abs(q.count - q.minCount)
        if (q.isComplete()) terms.push(gap * pw.overagePerHead!)
        else {
          unmet++
          terms.push(gap * pw.shortfallPerHead!)
        }
      }
      const completionPenalty = unmet === 0 ? average(terms) : sum(terms) + pw.unmetQuotaConstant! * unmet

      // use rejections vs targetRejections
      let delta = (this.rejected - TARGET_REJECTIONS) * pw.targetPerPerson!
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
      const attrs = getAttributes(guest).filter((a) => quotas[a]?.inProgress?.())
      if (!attrs.length) return 0
      const peopleLeft = this.getPeopleLeftInLine()
      let score = 0
      for (const a of attrs) score += quotaUrgency(quotas[a], peopleLeft)
      score += pairBonus(attrs, peopleLeft)
      return score
    },

    /** Admit if guest covers ≥ baseFrac of current unmet quotas; tighten if scarce/behind. */
    admitByFraction(guest: ScenarioAttributes, baseFrac: number): boolean {
      if (this.isFinishedWithQuotas()) return true
      const scarcity = this.seatScarcity()

      const seatProgress = this.getTotalProgress()
      const quotaProgress = this.getQuotaProgress() // 1.0 = on pace; >1 behind
      const progressDelta = clamp(1 - quotaProgress - seatProgress, [-1, 1])

      const tighten = 0.15 * scarcity * clamp(-progressDelta, [0, 1])
      const needFrac = Math.min(0.95, baseFrac + tighten)

      return this.guestFractionHit(guest) >= needFrac
    },

    /** Admit using numeric urgency score with dynamic threshold. */
    admitByScore(guest: ScenarioAttributes, baseTheta = 1.0): boolean {
      const scarcity = this.seatScarcity()
      const score = this.guestScore(guest)
      const quotaBehind = Math.max(0, this.getQuotaProgress() - 1.0)
      const theta = baseTheta + 0.6 * scarcity + 0.4 * Math.min(1, quotaBehind)
      return score >= theta
    },
    /** returns true when above max admissions or max rejections. */
    isComplete() {
      return !this.inProgress()
    },
    /** returns true when under max admissions and max rejections. */
    inProgress() {
      return this.admitted < config.maxAdmissions && this.rejected < config.maxRejections
    },
    /** returns the total number of people admitted and rejected. */
    getTotalPeopleSeen() {
      return this.admitted + this.rejected
    },
    /** returns the total number of people left in line. */
    getPeopleLeftInLine(): number {
      return Math.max(0, config.maxRejections - this.getTotalPeopleSeen())
    },
    /** returns the total number of available spaces for admissions. */
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
      const vals = this.quotas().map((q) => clamp(q.relativeProgress(peopleLeftInLine), [0, 3]))
      return average(vals) || 1.0
    },
    /** Returns true when all quotas have been met. */
    isFinishedWithQuotas(): boolean {
      return this.quotas().length === 0
    },
    /** Returns an array of quotas which are still in progress. */
    quotas() {
      return Object.values(quotas).filter((quota) => quota.inProgress())
    },
    /** Returns quota data for a specific attribute. */
    get(attr: string): Quota {
      return quotas[attr]
    },
    /** Returns an object with helpers for a guest. */
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
    /**
     * (optimistic) Returns the total number of people needed to reach the quota with the
     * highest amout to go, since this needs to be filled no matter what.
     */
    getMinPeopleNeeded(): number {
      const arr = this.quotas().map((quota) => quota.needed())
      return arr.length ? Math.max(...arr) : 0
    },
    /**
     * (pessimistic) Returns the sum of all the people needed to meet every quota, generally
     * the actual amount will be lower since a single person may have multiple attributes,
     * but at later stages in the game this is helpful.
     */
    getMaxPeopleNeeded(): number {
      return sum(this.quotas().map((quota) => quota.needed()))
    },
    /**
     * @important make sure to call this after making a decision to update the counts on all the
     * quotas and overall progress.
     */
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
    /**
     * Returns a summary of the current state for logging.
     */
    getSummary() {
      const isSuccess = this.isFinishedWithQuotas()
      return {
        status: isSuccess ? 'completed' : 'failed',
        success: isSuccess,
        reason: `Admissions=${this.admitted} Rejected=${this.rejected}`,
        finalRejections: this.rejected,
        finalAdmissions: this.admitted,
        constraints: Object.values(quotas).map((quota) => quota.getSummary()),
      } as const
    },
  }
}
