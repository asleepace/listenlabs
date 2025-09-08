import type { GameState, Person, PersonAttributes } from './types'
import type { BergainBouncer } from './berghain'
import { Stats } from './stats'
import { Metrics } from './metrics'

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

interface GameQuota<Attributes extends PersonAttributes> {
  attribute: keyof PersonAttributes
  needed: number
}

interface GameProgress<Attributes extends PersonAttributes> {
  config: typeof CONFIG
  critical: string[]
  quotas: GameQuota<PersonAttributes>[]
  admissionRate: number
  admitted: number
  rejected: number
  info: any
}

type CriticalAttributes<Attributes extends PersonAttributes> = Partial<
  Record<keyof Attributes, { needed: number; required: boolean }>
>

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
export const CONFIG = {
  // Admission threshold settings
  /**
   * Base admission score threshold less is more lenient early on.
   * @note normalized game averages ~0.51 to admit, current best with 0.75
   * @range 0.2 to 0.7
   * @default 0.7
   */
  BASE_THRESHOLD: 0.49,
  MIN_THRESHOLD: 0.45,
  MAX_THRESHOLD: 0.95,
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
  CRITICAL_IN_LINE_RATIO: 0.8,

  /**
   * Percentage of remaining spots needed.
   * @default 0.8 (80% full)
   */
  CRITICAL_CAPACITY_RATIO: 0.8,

  /**
   * Number of scores needed for calculations.
   */
  MIN_RAW_SCORES: 5,
}

export type BouncerConfig = typeof CONFIG

// ... existing interfaces and CONFIG ...

export class Bouncer<
  Attributes extends PersonAttributes,
  Keys extends keyof Attributes = keyof Attributes
> implements BergainBouncer
{
  static intialize(overrides: Partial<BouncerConfig>) {
    Object.entries(overrides).forEach(([key, value]) => {
      // @ts-ignore
      if (key in CONFIG) CONFIG[key] = value
      else throw new Error(`MISSING_CONFIG_KEY: ${key}!`)
    })
    return (gameState: GameState) => new Bouncer(gameState)
  }

  private metrics: Metrics<Attributes>
  private maxCapacity = CONFIG.MAX_CAPACITY
  private totalPeople = CONFIG.TOTAL_PEOPLE

  public progress: GameProgress<Attributes>
  public state: GameState
  public info: Record<string, any> = {}

  public totalScores: number[] = []
  public rawScores: number[] = []
  public totalAdmittedScores: number[] = []
  public lowestAcceptedScore = Infinity
  public totalUnicorns = 0

  public criticalAttributes: CriticalAttributes<Attributes> = {}

  constructor(initialData: GameState) {
    this.metrics = new Metrics(initialData.game)
    this.state = initialData
    this.progress = this.getProgress()
  }

  // Simplified getters using metrics
  private get constraints() {
    return this.metrics.constraints
  }

  private get correlations() {
    return this.metrics.correlations
  }

  private get frequencies() {
    return this.metrics.frequencies
  }

  private get totalSpotsLeft(): number {
    return this.maxCapacity - this.admittedCount
  }

  private get estimatedPeopleInLineLeft(): number {
    return this.totalPeople - this.admittedCount - this.rejectedCount
  }

  private get allQuotasMet(): boolean {
    return this.metrics.allConstraintsMet
  }

  get admittedCount() {
    if (this.state.status.status !== 'running') throw this.state.status
    return this.state.status.admittedCount
  }

  get rejectedCount() {
    if (this.state.status.status !== 'running') throw this.state.status
    return this.state.status.rejectedCount
  }

  // Delegate counting methods to metrics
  public getCount(attribute: Keys): number {
    return this.metrics.getCount(attribute)
  }

  public getCorrelations(attribute: Keys | string) {
    return this.metrics.correlations[attribute as Keys]
  }

  public getFrequency(attribute: Keys | string): number {
    return this.metrics.frequencies[attribute as Keys]
  }

  private getPeopleNeeded(attribute: Keys | string): number {
    return this.metrics.getNeeded(attribute as Keys)
  }

  // Enhanced critical attributes using metrics insights
  private getCriticalAttributes(): CriticalAttributes<Attributes> {
    const peopleInLineLeft = this.estimatedPeopleInLineLeft
    const totalSpotsLeft = this.totalSpotsLeft

    // Use metrics risk assessment
    const riskAssessment = this.metrics.getRiskAssessment()
    const incompleteConstraints = this.metrics.getIncompleteConstraints()

    this.criticalAttributes = {}

    incompleteConstraints.forEach((constraint) => {
      const attr = constraint.attribute
      const needed = constraint.needed
      const frequency = this.metrics.frequencies[attr]

      const estimatedRemaining = peopleInLineLeft * frequency
      const isCritical = riskAssessment.criticalAttributes.includes(attr)
      const isRequired =
        needed >= totalSpotsLeft - CONFIG.CRITICAL_REQUIRED_THRESHOLD

      if (isCritical || isRequired) {
        this.criticalAttributes[attr] = {
          needed,
          required: isRequired,
        }
      }
    })

    return this.criticalAttributes
  }

  private updateCounts(
    attributes: Attributes,
    score: number,
    shouldAdmit: boolean
  ) {
    if (!shouldAdmit) return

    this.totalAdmittedScores.push(score)
    this.lowestAcceptedScore = Math.min(score, this.lowestAcceptedScore)

    // Use metrics to update counts
    this.metrics.updateCounts(attributes)
  }

  private normalizeScore(rawScore: number): number {
    if (this.rawScores.length < CONFIG.MIN_RAW_SCORES) {
      return Math.min(rawScore / 5.0, 1.0)
    }
    const avgScore = Stats.average(this.rawScores) || 0.5
    const stdDev = Stats.stdDev(this.rawScores) || 0.2
    return 1.0 / (1.0 + Math.exp(-(rawScore - avgScore) / stdDev))
  }

  // Improved threshold calculation using metrics
  private getProgressThreshold(): number {
    const totalProcessed = this.admittedCount + this.rejectedCount
    const expectedProgress = Math.min(totalProcessed / CONFIG.TARGET_RANGE, 1.0)
    const totalProgress = this.metrics.totalProgress

    // Use metrics efficiency analysis
    const efficiency = this.metrics.getEfficiencyMetrics()
    const riskAssessment = this.metrics.getRiskAssessment()

    // Adjust sensitivity based on risk
    const baseSensitivity = 2.0
    const riskMultiplier = Stats.clamp(riskAssessment.riskScore / 5.0, 0.5, 2.0)
    const sensitivity = baseSensitivity * riskMultiplier

    const delta = (expectedProgress - totalProgress) * 0.5
    const threshold = Math.max(
      CONFIG.MIN_THRESHOLD,
      Math.min(
        CONFIG.MAX_THRESHOLD,
        CONFIG.BASE_THRESHOLD - delta * sensitivity
      )
    )

    // Enhanced logging with metrics insights
    this.info['expected_progress'] = Stats.round(expectedProgress, 10_000)
    this.info['total_progress'] = Stats.round(totalProgress, 10_000)
    this.info['efficiency'] = efficiency.actualEfficiency
    this.info['risk_score'] = riskAssessment.riskScore
    this.info['threshold'] = Stats.round(threshold, 10_000)

    return threshold
  }

  admit(status: GameState['status']): boolean {
    this.state.status = status
    const { nextPerson } = status

    if (!nextPerson) return false

    if (this.allQuotasMet) {
      console.log('[game] auto-admitting all quotas met...')
      return true
    }

    const spotsLeft = this.totalSpotsLeft
    this.criticalAttributes = this.getCriticalAttributes()
    const criticalKeys = Object.keys(this.criticalAttributes) as Keys[]
    const personAttributes = nextPerson.attributes

    // Enhanced critical attribute check
    if (spotsLeft < 100 && criticalKeys.length > 0) {
      for (const [key, value] of Object.entries(this.criticalAttributes)) {
        if (!value || !personAttributes) continue
        if (value.required && !personAttributes[key as any]) return false
      }
    }

    // Early game strategy using metrics
    if (this.admittedCount < CONFIG.MIN_RAW_SCORES) {
      const rarestAttrs = this.metrics.getRarestAttributes().slice(0, 2)
      const hasRareAttr = rarestAttrs.some((attr) => personAttributes[attr])
      const hasMultipleUseful =
        this.metrics.countUsefulAttributes(personAttributes as any) >= 2

      if (!hasRareAttr && !hasMultipleUseful) return false
    }

    const score = this.calculateAdmissionScore(personAttributes)
    const threshold = this.getProgressThreshold()

    // Use metrics for better person evaluation
    const hasEveryAttribute = this.metrics.hasAllAttributes(
      personAttributes as any
    )
    const hasEveryCriticalAttribute =
      criticalKeys.length > 0 &&
      criticalKeys.every((attr) => personAttributes[attr]!)

    if (hasEveryAttribute) this.totalUnicorns++

    const shouldAdmit =
      score > threshold || hasEveryCriticalAttribute || hasEveryAttribute

    // Enhanced logging
    this.totalScores.push(score)
    this.info['last_score'] = Stats.round(score)
    this.info['useful_attrs'] = this.metrics.countUsefulAttributes(
      personAttributes as any
    )
    this.info['avg_score'] = Stats.round(
      Stats.average(this.totalAdmittedScores)
    )

    // Add metrics summary to info
    const summary = this.metrics.getSummary()
    this.info['metrics_summary'] = summary

    this.updateCounts(personAttributes as any, score, shouldAdmit)
    return shouldAdmit
  }

  // Improved score calculation using metrics insights
  private calculateAdmissionScore(attributes: Record<string, boolean>): number {
    if (this.allQuotasMet || this.state.status.status !== 'running') {
      return 10.0
    }

    let score = 0
    const totalProcessed = this.admittedCount + this.rejectedCount

    // Get strategic insights from metrics
    const difficulty = this.metrics.getQuotaDifficulty()
    const correlationInsights = this.metrics.getCorrelationInsights()
    const usefulAttributes = this.metrics.getUsefulAttributes(
      attributes as Attributes
    )

    // Score each useful attribute
    usefulAttributes.forEach((attr) => {
      const constraint = this.constraints.find((c) => c.attribute === attr)
      if (!constraint) return

      const needed = this.metrics.getNeeded(attr)
      const frequency = this.metrics.frequencies[attr]
      const attrDifficulty = difficulty.get(attr)!

      // Base scoring factors
      const targetProgress = Math.min(totalProcessed / CONFIG.TARGET_RANGE, 1.0)
      const actualProgress = this.metrics.getProgress(attr)
      const progressGap = targetProgress - actualProgress
      const urgency =
        progressGap > 0 ? progressGap * CONFIG.URGENCY_MODIFIER : 0

      // Enhanced risk calculation using estimated remaining
      const expectedRemaining = this.getEstimatedPeopleWithAttributeLeftInLine(
        attr as Keys
      )
      const riskFactor = needed / Math.max(expectedRemaining, 1)

      // Use difficulty ranking from metrics
      const difficultyMultiplier = attrDifficulty.difficulty / 10.0

      let componentScore = (urgency + riskFactor) * difficultyMultiplier

      // Critical attribute adjustments
      const critical = this.criticalAttributes[attr]
      if (critical?.required) {
        componentScore *= 1.5
      } else if (critical) {
        componentScore *= 0.5
      }

      // Correlation bonuses using insights
      let correlationBonus = 0

      // Check for strong positive correlations
      const strongPairs = correlationInsights.strongPairs.filter(
        (pair) =>
          (pair.attr1 === attr || pair.attr2 === attr) && pair.bothNeeded
      )
      strongPairs.forEach((pair) => {
        const otherAttr = pair.attr1 === attr ? pair.attr2 : pair.attr1
        if (attributes[otherAttr as any]) {
          correlationBonus += pair.correlation * CONFIG.CORRELATION_BONUS
        }
      })

      // Check for conflict pairs (negatively correlated but both needed)
      const conflictPairs = correlationInsights.conflictPairs.filter(
        (pair) =>
          (pair.attr1 === attr || pair.attr2 === attr) && pair.bothNeeded
      )
      conflictPairs.forEach((pair) => {
        const otherAttr = pair.attr1 === attr ? pair.attr2 : pair.attr1
        if (attributes[otherAttr as any]) {
          correlationBonus +=
            Math.abs(pair.correlation) * CONFIG.RARE_PERSON_BONUS
        }
      })

      score += componentScore * (1 + correlationBonus)
    })

    // Multi-attribute bonus
    const usefulCount = usefulAttributes.length
    if (usefulCount > 1) {
      score *= 1 + CONFIG.MULTI_ATTRIBUTE_BONUS * (usefulCount - 1)
    }

    this.rawScores.push(score)

    // Trim arrays for memory management
    if (this.rawScores.length > 1000) {
      this.rawScores = this.rawScores.slice(-500)
    }

    return this.normalizeScore(score)
  }

  // Simplified estimation using metrics
  public getEstimatedPeopleWithAttributeLeftInLine(attribute: Keys): number {
    const frequency = this.metrics.frequencies[attribute]
    const peopleLeft = this.estimatedPeopleInLineLeft

    // Use correlation insights for better estimation
    const correlationInsights = this.metrics.getCorrelationInsights()
    const conflicts = correlationInsights.conflictPairs.filter(
      (pair) => pair.attr1 === attribute || pair.attr2 === attribute
    )

    // Reduce estimate if there are critical conflicts
    let adjustedEstimate = peopleLeft * frequency
    conflicts.forEach((conflict) => {
      if (conflict.bothNeeded) {
        const reduction = Math.abs(conflict.correlation) * 0.3
        adjustedEstimate *= 1 - reduction
      }
    })

    return Math.max(adjustedEstimate, 1)
  }

  public getProgress(): GameProgress<Attributes> {
    const incompleteQuotas = this.metrics
      .getIncompleteConstraints()
      .map((cp) => ({
        attribute: cp.attribute,
        needed: cp.needed,
      }))

    const criticalAttributes = Object.entries(this.criticalAttributes).map(
      ([key, value]) => (value?.required ? `${key}!` : key)
    )

    // Enhanced info with metrics insights
    const metricsAnalysis = this.metrics.getDetailedAnalysis()
    const enhancedInfo = {
      ...this.info,
      metrics_efficiency: metricsAnalysis.efficiency.actualEfficiency,
      metrics_risk: metricsAnalysis.risk.riskScore,
      metrics_balance: metricsAnalysis.variability.isBalanced,
      recommendations: metricsAnalysis.efficiency.recommendations,
    }

    return {
      info: enhancedInfo,
      critical: criticalAttributes,
      config: CONFIG,
      quotas: incompleteQuotas.sort((a, b) => a.needed - b.needed) as any,
      admissionRate:
        this.admittedCount / (this.admittedCount + this.rejectedCount),
      admitted: this.admittedCount,
      rejected: this.rejectedCount,
    }
  }

  public getOutput() {
    const analysis = this.metrics.getDetailedAnalysis()
    return {
      ...this.getProgress(),
      accuracy: Stats.percent(
        Math.abs(CONFIG.TARGET_RANGE - this.rejectedCount) / CONFIG.TARGET_RANGE
      ),
      scores: this.totalScores,
      metrics_analysis: analysis,
      final_summary: this.metrics.getSummary(),
    }
  }
}
