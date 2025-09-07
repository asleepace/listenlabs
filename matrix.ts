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

type CriticalAttributes = Partial<
  Record<Keys, { needed: number; required: boolean }>
>

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
  export function percent(num: number): number {
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
 *  
 * current best:
    MIN_THRESHOLD: 0.75,
    THRESHOLD_RAMP: 0.3,
    TARGET_RANGE: 5000,
    URGENCY_MODIFIER: 3,
    CORRELATION_BONUS: 0.3,
    NEGATIVE_CORRELATION_BONUS: 0.5,
    NEGATIVE_CORRELATION_THRESHOLD: -0.5,
    MULTI_ATTRIBUTE_BONUS: 0.5,
    RARE_PERSON_BONUS: 0.5,
    MAX_CAPACITY: 1000,
    TOTAL_PEOPLE: 10000,
    CRITICAL_THRESHOLD: 50,
    CRITICAL_IN_LINE_RATIO: 0.75,
    CRITICAL_CAPACITY_RATIO: 0.8,
    GUARENTEED: 10,

 
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
   * @note normalized game averages ~0.51 to admit, current best with 0.75
   * @range 0.2 to 0.7
   * @default 0.7
   */
  MIN_THRESHOLD: 0.75, // (less = moderately lenient, 0.7=default)
  /**
   * How quickly threshold decreases as we fill up, lesser for gradual tightening.
   * Lower = consistent threshold throughout
   * Higher = gets much stricter as you fill up
   * @range 0.2 to 0.8
   * @default 0.5
   */
  THRESHOLD_RAMP: 0.3,

  /**
   * Aim to complete all quotas by persons by this value (out of 10,000)
   * Lower = rush to fill quotas early (risk running out of spots)
   * Higher = spread out (risk missing rare attributes)
   * @range 2000 to 6000
   * @default 4000 (people)
   * @note current best score on leaderboard.
   */
  TARGET_RANGE: 5000,

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
  NEGATIVE_CORRELATION_BONUS: 0.7,
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
   * Number of available spots left where attribute becomes required.
   * @default 1 (person)
   */
  CRITICAL_REQUIRED_THRESHOLD: 5,

  /**
   * Percentage of remaining people we need to fill quota.
   * @default 0.75 (75% percent)
   */
  CRITICAL_IN_LINE_RATIO: 0.75,

  /**
   * Percentage of remaining spots needed.
   * @default 0.8 (80% full)
   */
  CRITICAL_CAPACITY_RATIO: 0.8,
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
  public totalAdmittedScores: number[] = []
  public lowestAcceptedScore = Infinity
  public totalUnicorns = 0
  public stats = {
    average: 0,
    medium: 0,
    mode: 0,
  }

  public criticalAttributes: CriticalAttributes = {}

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
   * Total number of quotas which have been reached.
   */
  private get totalQuotasMet(): number {
    return this.gameData.constraints.reduce((total, constraint) => {
      const constraintCount = this.getCount(constraint.attribute)
      return total + (constraintCount <= constraint.minCount ? 1 : 0)
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
   * Get the current average score (with normalized values 0.0 to 1.0)
   */
  private get averageScore(): number {
    return Stats.average(
      this.totalScores.map((item) => (item > 1.0 ? 1 : item))
    )
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
  public getCount(attribute: Keys | string): number {
    if (!(attribute in this.attributeCounts))
      throw new Error(`Missing attribute count for: ${attribute}`)
    return this.attributeCounts[attribute] ?? 0
  }

  public getCorrelations(attribute: Keys | string) {
    if (!(attribute in this.correlations))
      throw new Error(`Missing correlation for: ${attribute}`)
    return this.correlations[attribute]!
  }

  public getFrequency(attribute: Keys | string): number {
    if (!(attribute in this.frequencies))
      throw new Error(`Missing frequency for: ${attribute}`)
    return this.frequencies[attribute]!
  }

  /**
   * Get the total number of people left to fill the attributes quota.
   */
  private getPeopleNeeded(attribute: Keys | string): number {
    const constraint = this.constraints.find(
      (constraint) => constraint.attribute === attribute
    )
    if (!constraint) throw new Error('Missing constraint for:' + attribute)
    const totalPeople = constraint.minCount - this.getCount(attribute)
    return totalPeople > 0 ? totalPeople : 0
  }

  /**
   * If we need more than 80% of expected remaining people for quota, or
   * we need 90% of total spot available.
   */
  private getCriticalAttributes(): CriticalAttributes {
    const peopleInLineLeft = this.estimatedPeopleInLineLeft
    const totalSpotsLeft = this.totalSpotsLeft

    const criticalLineThreshold =
      peopleInLineLeft * CONFIG.CRITICAL_IN_LINE_RATIO

    const criticalCapacityThreshold =
      totalSpotsLeft * CONFIG.CRITICAL_CAPACITY_RATIO

    this.criticalAttributes = this.gameData.constraints.reduce(
      (output, current) => {
        const peopleNeeded = this.getPeopleNeeded(current.attribute)
        if (!peopleNeeded) return output

        const estimatedPeopleLeftInLine =
          peopleInLineLeft * this.getFrequency(current.attribute)

        // check if at thresholds...
        const isCriticalLineThreshold =
          estimatedPeopleLeftInLine >= criticalLineThreshold

        const isCriticalCapacityThreshold =
          peopleNeeded >= criticalCapacityThreshold

        if (isCriticalLineThreshold || isCriticalCapacityThreshold) {
          const isRequired =
            this.totalSpotsLeft >=
            peopleNeeded + CONFIG.CRITICAL_REQUIRED_THRESHOLD

          return {
            ...output,
            [current.attribute as Keys]: {
              needed: peopleNeeded,
              required: isRequired,
            },
          }
        } else {
          return output
        }
      },
      {} as Record<Keys, { needed: number; required: boolean }>
    )

    // do a second pass to make sure two attrs don't both become required

    return this.criticalAttributes
  }

  /**
   * Update our current counts when we admint a new person.
   * @param person
   * @param shouldAdmit
   */
  updateCounts(
    attributes: Person['attributes'],
    score: number,
    shouldAdmit: boolean
  ) {
    if (!shouldAdmit) return
    this.totalAdmittedScores.push(score)
    this.lowestAcceptedScore = Math.min(score, this.lowestAcceptedScore)
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
    this.state.status = status // update state status
    const { nextPerson } = status

    /**
     * prevent issues when network request fails
     */
    if (!nextPerson) return false

    /**
     * auto-admit if all quotas already met.
     */
    if (this.allQuotasMet) {
      console.log('[game] auto-admitting all quotas met...')
      return true
    }

    /**
     * check total progress and spots left
     */
    const spotsLeft = this.totalSpotsLeft
    const totalSpotsLeft = this.estimatedPeopleInLineLeft
    this.criticalAttributes = this.getCriticalAttributes()

    const criticalKeys = Object.keys(this.criticalAttributes) as Keys[]

    /**
     * if at capacity reject everyone
     * @testing
     */
    // if (spotsLeft < 0) return false

    /**
     * Attributes for person to check.
     */
    const personAttributes = nextPerson.attributes

    /**
     * Calculate admission score
     */
    let score = this.calculateAdmissionScore(personAttributes)

    /**
     * dynamic threshold based on how many spots are left
     */
    const progressRatio = this.admittedCount / this.maxCapacity

    /**
     * Should be easier to get in if we are ahead on quotas,
     * Should be harder to get in if we are behind in qoutas.
     * @testing dynamic schedule.
     */
    const totalProgress =
      this.constraints.reduce((total, constraint) => {
        const currentCount = this.getCount(constraint.attribute)
        const totalNeeded = constraint.minCount - currentCount
        if (totalNeeded < 1) return total + 1
        const attrProgress = currentCount / totalNeeded
        return total + attrProgress
      }, 0) / this.constraints.length

    // Option 1: Exponential decay (recommended)
    // When on track (ratio = 1.0): threshold ≈ 0.37
    // When behind (ratio < 1.0): threshold approaches 1.0
    // When ahead (ratio > 1.0): threshold approaches 0.0
    // Handles any value > 0 gracefully
    // const threshold = Math.exp(-(totalProgress / progressRatio))

    // OPTION #2:  Sigmoid/tanh for smoother transitions
    // When on track: threshold = 0.5
    // Smooth S-curve transition
    // Bounded between 0.0 and 1.0
    const threshold =
      (1.0 + Math.tanh(1.0 - totalProgress / progressRatio)) / 2.0

    // Option 3: Simple reciprocal with floor
    // Linear-ish but handles high values well
    // When ratio = 0: threshold = 1.0
    // When ratio = 1: threshold = 0.5
    // When ratio = ∞: threshold approaches 0.0
    // const ratio = totalProgress / progressRatio;
    // const threshold = Math.max(0.0, Math.min(1.0, 1.0 / (ratio + 1.0)));

    // const threshold =
    //   CONFIG.MIN_THRESHOLD * (1 - totalProgress * CONFIG.THRESHOLD_RAMP)

    /**
     * check if person has all attributes.
     */
    const hasEveryAttribute = this.constraints.every(
      (constraint) => personAttributes[constraint.attribute as Keys] === true
    )

    /**
     * person is a unicorn and has all critical attributes.
     */
    const hasEveryCriticalAttribute =
      criticalKeys.length > 0 &&
      criticalKeys.every((attrKey) => personAttributes[attrKey])

    /**
     * calculate if we should admit this person.
     */
    const shouldAdmit =
      score >= threshold || hasEveryCriticalAttribute || hasEveryAttribute

    /**
     * update generic game information for debugging.
     */
    this.totalScores.push(score)
    this.info['unicorns'] = hasEveryAttribute
      ? ++this.totalUnicorns
      : this.totalUnicorns
    this.info['last_score'] = score
    this.info['best_score'] = Math.max(this.info['best_score'] ?? 0, score)
    this.info['lows_score'] = this.lowestAcceptedScore
    this.info['avrg_score'] = Stats.average(this.totalAdmittedScores)
    this.info['total_progress'] = Stats.round(totalProgress, 10_000)
    this.info['progress_ratio'] = Stats.round(progressRatio, 10_000)
    this.info['threshold'] = threshold
    /**
     * Update the counts.
     */
    this.updateCounts(personAttributes, score, shouldAdmit)

    /**
     * Return our decision.
     */
    return shouldAdmit
  }

  /**
   * This returns an estimate of the number of people left in line with the given
   * attribute, this is calculated by:
   *
   *  1. Get total estimate people left in line
   *  2. Subtract non-correlated people x their frequency
   *  3. Multiply the remaining number by the attributes frequency
   *
   * NOTE: This can be overly pessimistic, but prevents impossible game states.
   */
  public getEstimatedPeopleWithAttributeLeftInLine(attribute: Keys) {
    const attributesCorrelations = this.getCorrelations(attribute)
    const attributeFrequency = this.getFrequency(attribute)
    const totalPeopleLeftInLine = this.gameData.constraints.reduce(
      (total, other) => {
        // skip current attribute
        if (attribute === other.attribute) return total
        // skip if non-critical other attribute
        if (!(other.attribute in this.criticalAttributes)) return total
        /**
         * before we only check if it was in critical attributes and negatively
         * correlated, but we should also check if the other one is required.
         * @testing
         */
        const criticalAttr = this.criticalAttributes[other.attribute as Keys]!
        if (criticalAttr.needed) {
          // NOTE: the goal is to prevent two required attributes at the same time,
          // since this creates gridlock.
          return (
            total - (criticalAttr.needed + CONFIG.CRITICAL_REQUIRED_THRESHOLD)
          )
        }

        // check if any other constraints are negatively correlated or required
        if (attributesCorrelations[other.attribute]! < 0) return total
        // calculate total number of other people neded
        const otherNeeded =
          other.minCount - this.getCount(other.attribute as Keys)
        // if we don't need other person just return total
        if (otherNeeded < 0) return total
        // get the frequncy then substract from total
        const frequency = this.getFrequency(other.attribute as Keys)
        // substract negative correlated people
        return total - otherNeeded * (1 - frequency)
      },
      this.estimatedPeopleInLineLeft
    )
    // assume that the person will show up by their frequency
    return totalPeopleLeftInLine * attributeFrequency
  }

  /**
   * Calculate the persons admission score.
   */
  private calculateAdmissionScore(attributes: Record<string, boolean>): number {
    let score = 0
    const frequencies = this.frequencies

    // If all quotas are met, admit everyone
    if (this.allQuotasMet || this.state.status.status !== 'running') {
      return 10.0 // High score to guarantee admission (arbitrary)
    }

    const { admittedCount, rejectedCount } = this.state.status
    const totalProcessed = admittedCount + rejectedCount

    // Calculate score for each attribute the person has
    this.gameData.constraints.forEach((constraint) => {
      const attr = constraint.attribute

      if (!attributes[attr]) return

      const currentCount = this.getCount(attr)
      const needed = constraint.minCount - currentCount

      /**
       * @testing prevent overfilling early attributes
       * same as if ((attr as Keys) === 'underground_veteran') return // fuck 'em
       */
      const minNeeded = this.totalQuotasMet === 0 ? 50 : 0
      if (needed <= minNeeded) return // Quota already met

      const frequency = frequencies[attr] || 0.5
      // const expectedRemaining =
      //   (peopleInLineLeft - otherNonCorrelatedPeopleLeft) * frequency

      const expectedRemaining = this.getEstimatedPeopleWithAttributeLeftInLine(
        constraint.attribute as Keys
      )

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

      /**
       * @testing improve score of critical attributes.
       */
      const critical = this.criticalAttributes[constraint.attribute as Keys]

      // Component score combines all factors
      let componentScore = (urgency + riskFactor) * Math.log(scarcityFactor + 1)

      /**
       * @testing boost critical attributes.
       */
      if (critical && critical.required) {
        componentScore *= 1.2
      }

      // Special boost for attributes that need above their natural rate
      const quotaRate = constraint.minCount / CONFIG.MAX_CAPACITY
      if (quotaRate > frequency * 1.5) {
        // Changed from 1.2 - only boost REALLY overdemanded
        componentScore *= 1.5 // Fixed multiplier instead of ratio
      }

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
    const targetScore = 5_000 // rejections
    return {
      ...this.getProgress(),
      accuracy: Stats.percent(
        Math.abs(targetScore - this.rejectedCount) / targetScore
      ),
      scores: this.totalScores,
    }
  }

  /**
   * Helper method to calculate current progress.
   * @note this is just for tracking.
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
      quotaProgress[attr] = Stats.round(
        this.attributeCounts[attr]! / constraint.minCount
      )
    })

    const { critical_attributes = [], ...info } = this.info

    const criticalAttributes = Object.entries(this.criticalAttributes).map(
      ([key, value]) => {
        if (value.required) return `${key}!`
        return key
      }
    )

    const progress: GameProgress = {
      info,
      critical: criticalAttributes,
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
