import type { Keys, GameState, Person } from '.'

interface Constraint {
  attribute: string
  minCount: number
}

interface GameData {
  gameId: string
  constraints: Constraint[]
  attributeStatistics: {
    relativeFrequencies: Record<string, number>
    correlations: Record<string, Record<string, number>>
  }
}

interface GameCounter {
  state: GameState
  admit(status: GameState['status']): boolean
}

interface GameQuota {
  attribute: Keys
  needed: number
}

interface GameProgress {
  config: typeof CONFIG
  critical: string[]
  quotas: GameQuota[]
  quotaProgress: Record<string, number>
  admissionRate: number
  admitted: number
  rejected: number
  info: any
}

namespace Stats {
  /**
   * Calculate the average value of an array.
   */
  export function average(items: number[]): number {
    if (!items || items.length === 0) return 0
    return items.reduce((total, current) => total + current, 0) / items.length
  }

  /**
   * Round the number to the specified places.
   */
  export function round(num: number, places = 100): number {
    return Math.round(num * places) / places
  }

  /**
   * Quickly convert a decimal to a percentage with 2 decimals.
   */
  export function percentage(num: number): number {
    return Stats.round(num, 10_000)
  }

  /**
   * Calculate the median value.
   */
  export function median(nums: number[]): number {
    let max = 0
    let min = Infinity
    nums.forEach((n) => {
      max = Math.max(n, max)
      min = Math.min(n, min)
    })
    return (max - min) / 2.0
  }
}

/** ## Bouncer Configuration
 
HIGH IMPACT (tune these first):
 
1. MIN_THRESHOLD (Range: 0.2 - 0.7)

  Impact: Controls base admission strictness
  Sweet spot: 0.3 - 0.5
  Lower = more lenient early (admit more people)
  Higher = stricter (risk not filling venue)
 
2. TARGET_RANGE (Range: 2000 - 6000)

  Impact: When you aim to complete quotas
  Sweet spot: 3500 - 4500
  Lower = rush to fill quotas early (risk running out of spots)
  Higher = spread out (risk missing rare attributes)
  Depends on rarest attribute frequency

3. URGENCY_MODIFIER (Range: 1.0 - 6.0)

  Impact: How much being behind schedule matters
  Sweet spot: 2.5 - 4.0
  Lower = relaxed about timing (risk missing quotas)
  Higher = panic when behind (risk over-admitting)

==== MEDIUM IMPACT (fine-tune after high impact) ====

4. THRESHOLD_RAMP (Range: 0.2 - 0.8)

  Impact: How threshold changes as venue fills
  Sweet spot: 0.4 - 0.6
  Lower = consistent threshold throughout
  Higher = gets much stricter as you fill up

5. MULTI_ATTRIBUTE_BONUS (Range: 0.3 - 1.5)

  Impact: Reward for people with multiple needed attributes
  Sweet spot: 0.5 - 0.8
  Too high = over-value "jack of all trades"
  Too low = miss efficient multi-quota fills

6. CRITICAL_THRESHOLD (Range: 10 - 100)

  Impact: When to panic about unfilled quotas
  Sweet spot: 30 - 60
  Lower = panic mode engages late
  Higher = conservative/safe approach

==== LOW IMPACT (minor tweaks) ====

7. CORRELATION_BONUS (Range: 0.1 - 0.5)

  Impact: Reward for positively correlated attributes
  Sweet spot: 0.2 - 0.3
  Minor effect on overall strategy

8. NEGATIVE_CORRELATION_BONUS (Range: 0.3 - 1.0)

  Impact: Reward for rare combinations
  Sweet spot: 0.5 - 0.7
  Helps with edge cases

9. RARE_PERSON_BONUS (Range: 0.3 - 1.0)

  Impact: Extra boost for rare combos
  Similar to NEGATIVE_CORRELATION_BONUS

10. NEGATIVE_CORRELATION_THRESHOLD (Range: -0.7 to -0.3)

  Impact: What counts as "negatively correlated"
  Sweet spot: -0.4 to -0.5
  Very minor impact
 
  **/
const CONFIG = {
  // Admission threshold settings
  /**
   * Base admission score threshold less is more lenient early on.
   * @range 0.2 to 0.7
   * @default 0.7
   */
  MIN_THRESHOLD: 0.7, // (less = moderately lenient, 0.7=default)
  /**
   * How quickly threshold decreases as we fill up, lesser for gradual tightening.
   * Lower = consistent threshold throughout
   * Higher = gets much stricter as you fill up
   * @range 0.2 to 0.8
   * @default 0.5
   */
  THRESHOLD_RAMP: 0.5,

  /**
   * Aim to complete all quotas by persons by this value (out of 10,000)
   * Lower = rush to fill quotas early (risk running out of spots)
   * Higher = spread out (risk missing rare attributes)
   * @range 2000 to 6000
   * @default 4000 (people)
   * @note current best score on leaderboard.
   */
  TARGET_RANGE: 4000,

  /**
   * Multiplier for how much being behind schedule matters.
   * Lower = relaxed about timing (risk missing quotas)
   * Higher = panic when behind (risk over-admitting)
   * @range 1.0 to 6.0
   * @default 2.2
   */
  URGENCY_MODIFIER: 3.0,
  /**
   * Reward for positively correlated attributes.
   * @range 0.1 to 0.5
   * @default 0.3
   */
  CORRELATION_BONUS: 0.3,
  /**
   * Bonus for rare combinations (negatively correlated but both needed).
   * @default 0.5
   */
  NEGATIVE_CORRELATION_BONUS: 0.5,
  /**
   * Correlation below this triggers special handling.
   * @range -0.7 to -0.3
   * @default -0.5
   */
  NEGATIVE_CORRELATION_THRESHOLD: -0.5,
  /**
   * Reward for people with multiple needed attributes (compounds)
   * Too high = over-value "jack of all trades"
   * Too low = miss efficient multi-quota fills
   * @range 0.5 to 0.8
   * @default 0.5
   */
  MULTI_ATTRIBUTE_BONUS: 0.5,
  /**
   * Bonus for rare attribute combinations (negatively correlated)
   * @range 0.3 to 1.0
   * @default 0.5
   */
  RARE_PERSON_BONUS: 0.5,

  /**
   * Total maximum people we can admit (constant)
   */
  MAX_CAPACITY: 1_000,
  /**
   * Total people in line to select from (constant)
   */
  TOTAL_PEOPLE: 10_000,
  /**
   * When to panic about unfilled quotas.
   * Lower = panic mode engages late
   * Higher = conservative/safe approach
   * @range 10 to 100
   * @default 60 (people)
   */
  CRITICAL_THRESHOLD: 50,
  /**
   * Threshold where we can be more lenient at start of night.
   * @default 100 (people)
   */
  EARLY_THRESHOLD: 100,

  /**
   * Arbitrary max value to guarentee admission.
   */
  GUARENTEED: 10,
}

/**
 *  ## Nightclub Bouncer
 *
 *  You're the bouncer at a night club. Your goal is to fill the venue with N=1000 people
 *  while satisfying constraints like "at least 40% Berlin locals",  or "at least 80% wearing all black".
 *
 *  People arrive one by one, and you must immediately decide whether to let them in or turn them away.
 *  Your challenge is to fill the venue with as few rejections as possible while meeting all minimum requirements.
 *
 *  - People arrive sequentially with binary attributes (e.g., female/male, young/old, regular/new)
 *  - You must make immediate accept/reject decisions
 *  - The game ends when either:
 *    (a) venue is full (1000 people)
 *    (b) you rejected 10,000 people
 *
 *  @scoring You score is the number of people you rejected before filling the venue (the less the better).
 */
export class NightclubGameCounter implements GameCounter {
  private gameData: GameData
  private attributeCounts: Record<string, number> = {}
  private maxCapacity = CONFIG.MAX_CAPACITY
  private totalPeople = CONFIG.TOTAL_PEOPLE

  public progress: GameProgress
  public state: GameState
  public info: Record<string, any> = {}

  public totalScores: number[] = []
  public stats = {
    average: 0,
    medium: 0,
    mode: 0,
  }

  constructor(initialData: GameState) {
    this.gameData = initialData.game
    this.state = initialData
    this.gameData.constraints.forEach((constraint) => {
      this.attributeCounts[constraint.attribute] = 0
    })
    this.progress = this.getProgress()
  }

  // getters

  private get constraints() {
    return this.gameData.constraints
  }

  private get correlations() {
    return this.gameData.attributeStatistics.correlations
  }

  private get frequencies() {
    return this.gameData.attributeStatistics.relativeFrequencies
  }

  /**
   * Total number of available spots left for entries.
   */
  private get totalSpotsLeft(): number {
    return this.maxCapacity - this.admittedCount
  }

  /**
   * Estimated number of people left in the line to check.
   */
  private get estimatedPeopleInLineLeft(): number {
    return this.totalPeople - this.admittedCount - this.rejectedCount
  }

  /**
   * Ratio of people already admitted over max capicity.
   * @note closer to 1.0 means more full
   */
  private get progressRatio(): number {
    return this.admittedCount / this.maxCapacity
  }

  /**
   * Check how many quotas have been filled.
   */
  private get totalQuotasFulfilled(): number {
    return this.progress.quotas.reduce((total, quota) => {
      const isFulfilled = quota.needed <= 0
      return total + (isFulfilled ? 1 : 0)
    }, 0)
  }

  /**
   * True when all quotas have been fulfilled.
   */
  private get allQuotasMet(): boolean {
    return this.gameData.constraints.every(
      (constraint) =>
        this.attributeCounts[constraint.attribute]! >= constraint.minCount
    )
  }

  /**
   * Check if any of the quotas are near the spots remaining.
   */
  private get criticalAttributes(): GameQuota[] {
    return this.progress.quotas.filter((quota) => {
      if (quota.needed < 0) return false
      return this.totalSpotsLeft - quota.needed < CONFIG.CRITICAL_THRESHOLD
    })
  }

  /**
   * Get the current average score (with normalized values 0.0 to 1.0)
   */
  private get averageScore(): number {
    return Stats.average(
      this.totalScores.map((item) => (item > 1.0 ? 1 : item))
    )
  }

  /**
   * Update our current counts when we admint a new person.
   * @param person
   * @param shouldAdmit
   */
  updateCounts(attributes: Person['attributes'], shouldAdmit: boolean) {
    if (!shouldAdmit) return
    Object.entries(attributes).forEach(([attr, hasAttr]) => {
      if (hasAttr && this.attributeCounts[attr] !== undefined) {
        this.attributeCounts[attr]++
      }
    })
  }

  /**
   * Check if we should admit or reject the next person in line.
   */
  admit(status: GameState['status']): boolean {
    // update state status
    this.state.status = status
    const { nextPerson } = status

    // prevent issues when network request fails
    if (!nextPerson) return false

    // check total progress and spots left
    const progress = this.getProgress()
    const spotsLeft = this.totalSpotsLeft
    const peopleInLineLeft = this.estimatedPeopleInLineLeft

    // if at capacity reject everyone
    if (spotsLeft < 0) return false

    const personAttributes = nextPerson.attributes

    /**
     * Calculate admission score
     */
    let score = this.calculateAdmissionScore(
      personAttributes,
      spotsLeft,
      peopleInLineLeft
    )

    // Identify attributes we MUST prioritize based on frequency vs needs
    const mustHaveAttributes: string[] = []
    this.gameData.constraints.forEach((constraint) => {
      const needed =
        constraint.minCount - this.attributeCounts[constraint.attribute]!
      const frequency = this.frequencies[constraint.attribute] || 0.5
      const expectedRemaining = peopleInLineLeft * frequency

      // If we need more than 80% of expected remaining, it's critical
      if (needed > expectedRemaining * 0.8) {
        mustHaveAttributes.push(constraint.attribute)
      }
    })

    // Auto-admit if person has 2+ must-have attributes
    const mustHaveCount = mustHaveAttributes.filter(
      (attr) => personAttributes[attr as Keys]
    ).length

    if (mustHaveCount >= 2) {
      this.updateCounts(personAttributes, true)
      return true
    }

    // Boost score significantly for must-have attributes
    if (mustHaveCount === 1) {
      score *= 2 // Double the score
    }

    /**
     * Dynamic threshold based on how many spots are left
     */
    const progressRatio = this.admittedCount / this.maxCapacity
    // Dynamic threshold that accounts for critical needs
    const baseThreshold = CONFIG.MIN_THRESHOLD
    let threshold = baseThreshold * (1 - progressRatio * CONFIG.THRESHOLD_RAMP)

    // Lower threshold if we have critical unmet quotas
    if (mustHaveAttributes.length > 0) {
      threshold *= 0.7 // 30% more lenient when critical needs exist
    }

    // Even more lenient in first 10% to grab rare combos
    if (progressRatio < 0.1) {
      threshold *= 0.8
    }

    /**
     * update generic game information for debugging.
     */
    this.totalScores.push(score)
    this.info['last_score'] = score
    this.info['best_score'] = Math.max(this.info['best_score'] ?? 0, score)
    this.info['avrg_score'] = this.averageScore
    this.info['critical_attributes'] = this.criticalAttributes

    /**
     * count unmet quotas this person helps with
     */
    let unmetQuotasHelped = 0
    let criticalQuotasHelped = 0

    this.gameData.constraints.forEach((constraint) => {
      const needed =
        constraint.minCount - this.attributeCounts[constraint.attribute]!

      if (personAttributes[constraint.attribute as Keys] && needed > 0) {
        unmetQuotasHelped++
        if (needed > spotsLeft + 50) {
          criticalQuotasHelped++
        }
      }
    })

    // Stricter requirements as we fill up
    if (progressRatio > 0.8 && unmetQuotasHelped === 0) {
      return false // Must help with something when almost full
    }

    if (
      progressRatio > 0.9 &&
      criticalQuotasHelped === 0 &&
      this.criticalAttributes.length > 0
    ) {
      return false // Must help with critical quotas in final stretch
    }

    const shouldAdmit = score >= threshold || criticalQuotasHelped > 0

    /**
     * Update the counts.
     */
    this.updateCounts(personAttributes, shouldAdmit)

    return shouldAdmit
  }

  /**
   * Total number of people admitted.
   */
  get admittedCount() {
    if (this.state.status.status !== 'running') throw this.state.status
    return this.state.status.admittedCount
  }

  /**
   * Total number of people rejected.
   */
  get rejectedCount() {
    if (this.state.status.status !== 'running') throw this.state.status
    return this.state.status.rejectedCount
  }

  /**
   * Get the current number of people with the attribute already admited.
   */
  public getCount(attribute: Keys): number {
    return this.attributeCounts[attribute] ?? 0
  }

  /**
   * Calculate the persons admission score.
   */
  private calculateAdmissionScore(
    attributes: Record<string, boolean>,
    spotsLeft: number,
    peopleInLineLeft: number
  ): number {
    let score = 0
    const frequencies = this.frequencies

    // Check if all quotas are already met
    const allQuotasMet = this.gameData.constraints.every(
      (constraint) =>
        this.attributeCounts[constraint.attribute]! >= constraint.minCount
    )

    // If all quotas are met, admit everyone
    if (allQuotasMet || this.state.status.status !== 'running') {
      return CONFIG.GUARENTEED // High score to guarantee admission (arbitrary)
    }

    const { admittedCount, rejectedCount } = this.state.status
    const totalProcessed = admittedCount + rejectedCount

    // Calculate score for each attribute the person has
    this.gameData.constraints.forEach((constraint) => {
      const attr = constraint.attribute

      if (!attributes[attr]) return
      if ((attr as Keys) === 'underground_veteran') return // fuck 'em

      const currentCount = this.attributeCounts[attr]!
      const needed = constraint.minCount - currentCount

      if (needed <= 0) return // Quota already met

      const frequency = frequencies[attr] || 0.5
      const expectedRemaining = peopleInLineLeft * frequency

      // Expected progress: where we should be at this point in the line
      // We want to fill quotas by person 5000 (halfway through the line)
      const targetProgress = Math.min(totalProcessed / CONFIG.TARGET_RANGE, 1.0)
      const actualProgress = currentCount / constraint.minCount
      const progressGap = targetProgress - actualProgress

      // Base urgency: how behind schedule are we?
      const urgency =
        progressGap > 0 ? progressGap * CONFIG.URGENCY_MODIFIER : 0

      // Scarcity factor: how rare is this attribute?
      const scarcityFactor = 1 / Math.max(frequency, 0.01)

      // Risk factor: can we afford to wait?
      const riskFactor = needed / Math.max(expectedRemaining, 1)

      // Component score combines all factors
      let componentScore = (urgency + riskFactor) * Math.log(scarcityFactor + 1)

      // Add correlation bonus for multiple needed attributes
      let correlationBonus = 0
      this.gameData.constraints.forEach((otherConstraint) => {
        if (otherConstraint.attribute === attr) return

        const otherAttr = otherConstraint.attribute
        const otherNeeded =
          otherConstraint.minCount - this.attributeCounts[otherAttr]!

        if (attributes[otherAttr] && otherNeeded > 0) {
          const correlation = this.correlations[attr]?.[otherAttr] || 0

          // Special handling for negatively correlated attributes
          if (correlation < CONFIG.NEGATIVE_CORRELATION_THRESHOLD) {
            // If negatively correlated but both needed, this person is extra valuable
            correlationBonus += Math.abs(correlation) * CONFIG.RARE_PERSON_BONUS // Reward rare combination
          } else {
            // Positive correlation is good when both are needed
            correlationBonus += correlation * 0.2
          }
        }
      })

      score += componentScore * (1 + correlationBonus)
    })

    // Bonus for having multiple useful attributes
    const usefulAttributes = Object.entries(attributes).filter(
      ([attr, has]) => {
        if (!has) return false
        const constraint = this.gameData.constraints.find(
          (c) => c.attribute === attr
        )
        if (!constraint) return false
        return this.attributeCounts[attr]! < constraint.minCount
      }
    ).length

    // Give multiplicative bonus for multiple attributes (compounds nicely)
    if (usefulAttributes > 1) {
      score *= 1 + CONFIG.MULTI_ATTRIBUTE_BONUS * (usefulAttributes - 1)
    }

    return score
  }

  public getGameData() {
    return {
      config: CONFIG,
      scores: this.totalScores,
      admittedCount: this.admittedCount,
      rejectedCount: this.rejectedCount,
    }
  }

  /**
   * Helper method to calculate current progress.
   */
  public getProgress(): GameProgress {
    const quotas: { attribute: Keys; needed: number }[] = []
    const quotaProgress: Record<string, number> = {}

    this.gameData.constraints.forEach((constraint) => {
      const attr = constraint.attribute
      const quota = constraint.minCount - this.attributeCounts[attr]!
      if (quota > 0) {
        quotas.push({ attribute: attr as Keys, needed: quota })
      }
      quotaProgress[attr] = this.attributeCounts[attr]! / constraint.minCount
    })

    const { critical_attributes = [], ...info } = this.info

    const progress: GameProgress = {
      info,
      critical: (critical_attributes as GameQuota[]).map(
        (item) => item.attribute
      ),
      config: CONFIG,
      quotas: quotas.toSorted((a, b) => a.needed - b.needed),
      quotaProgress,
      admissionRate:
        this.admittedCount / (this.admittedCount + this.rejectedCount),
      admitted: this.admittedCount,
      rejected: this.rejectedCount,
    }

    return progress
  }
}
