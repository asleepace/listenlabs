import type { GameState, PersonAttributes } from '../types'
import type { BergainBouncer } from './berghain'
import { BASE_CONFIG, type GameConfig } from '../conf/game-config'
import { Stats } from './stats'
import { Metrics, type AttributeRisk } from './metrics'

interface GameQuota<Attributes extends PersonAttributes> {
  attribute: keyof Attributes
  needed: number
}

interface GameProgress<Attributes extends PersonAttributes> {
  config: GameConfig
  critical: string[]
  quotas: GameQuota<Attributes>[]
  admissionRate: number
  admitted: number
  rejected: number
  info: any
}

type Critical = {
  needed: number
  required: boolean
  modifier: number
}

type CriticalAttributes<Attributes extends PersonAttributes> = Partial<
  Record<keyof Attributes, Critical>
>

// fine tune config here...

const TUNED_CONFIG: Partial<GameConfig> = {
  // Most impactful
  BASE_THRESHOLD: 0.45, // Back closer to successful run (0.41)
  MIN_THRESHOLD: 0.3,
  MAX_THRESHOLD: 0.9,
  TARGET_RANGE: 4000,
  URGENCY_MODIFIER: 3.0,
  MULTI_ATTRIBUTE_BONUS: 1.2,

  // Critical attributes
  CRITICAL_REQUIRED_THRESHOLD: 20,
  CRITICAL_IN_LINE_RATIO: 0.8,
  CRITICAL_CAPACITY_RATIO: 0.9,

  // Less important with simplified approach
  CORRELATION_BONUS: 0.2,
  NEGATIVE_CORRELATION_BONUS: 0.5,
  NEGATIVE_CORRELATION_THRESHOLD: -0.5,
  RARE_PERSON_BONUS: 0.4,

  // Constants
  MAX_CAPACITY: 1000,
  TOTAL_PEOPLE: 10000,
  MIN_RAW_SCORES: 1,
}

export class Bouncer<
  Attributes extends PersonAttributes,
  Keys extends keyof Attributes = keyof Attributes
> implements BergainBouncer
{
  static CONFIG = { ...BASE_CONFIG, ...TUNED_CONFIG }
  static intialize(overrides: Partial<GameConfig>) {
    Object.entries(overrides).forEach(([key, value]) => {
      // @ts-ignore
      Bouncer.CONFIG[key as keyof GameConfig] = value
    })

    return (gameState: GameState) => new Bouncer(gameState)
  }

  private metrics: Metrics<Attributes>
  private maxCapacity = Bouncer.CONFIG.MAX_CAPACITY
  private totalPeople = Bouncer.CONFIG.TOTAL_PEOPLE

  public progress: GameProgress<Attributes>
  public state: GameState
  public info: Record<string, any> = {}

  public totalScores: number[] = []
  public rawScores: number[] = []
  public totalAdmittedScores: number[] = []
  public lowestAcceptedScore = Infinity
  public totalUnicorns = 0
  public rateGap = 0

  public criticalAttributes: CriticalAttributes<Attributes> = {}
  public riskAssessment: AttributeRisk<Attributes>

  constructor(initialData: GameState) {
    this.metrics = new Metrics(initialData.game)
    this.state = initialData
    this.progress = this.getProgress()
    this.riskAssessment = this.metrics.getRiskAssessment(
      this.estimatedPeopleInLineLeft
    )
  }

  // Simplified getters using metrics
  private get constraints() {
    return this.metrics.constraints
  }

  private get totalSpotsLeft(): number {
    return Bouncer.CONFIG.MAX_CAPACITY! - this.admittedCount
  }

  private get estimatedPeopleInLineLeft(): number {
    return (
      Bouncer.CONFIG.TOTAL_PEOPLE! - this.admittedCount - this.rejectedCount
    )
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

  private getCriticalAttributes(): CriticalAttributes<Attributes> {
    if (this.admittedCount < 50) return {}

    const peopleInLineLeft = this.estimatedPeopleInLineLeft
    const totalSpotsLeft = this.totalSpotsLeft
    const incompleteConstraints = this.metrics.getIncompleteConstraints()

    this.criticalAttributes = {}

    incompleteConstraints.forEach((constraint) => {
      const attr = constraint.attribute
      const needed = constraint.needed
      const frequency = this.metrics.frequencies[attr]
      const estimatedRemaining = peopleInLineLeft * frequency

      // MORE AGGRESSIVE CRITICAL DETECTION
      const urgencyRatio = needed / Math.max(1, totalSpotsLeft)
      const scarcityRatio = needed / Math.max(1, estimatedRemaining)

      // Critical if we need more than 15% of remaining spots for this attribute
      const isCapacityCritical = urgencyRatio > 0.15

      // Critical if we need more than 90% of expected remaining people with this attribute
      const isScarcityCritical = scarcityRatio > 0.9

      // Risk assessment critical
      const isRiskCritical =
        this.riskAssessment.criticalAttributes.includes(attr)

      if (isCapacityCritical || isScarcityCritical || isRiskCritical) {
        // MUCH MORE AGGRESSIVE MODIFIER
        const modifier = Math.max(
          2,
          Math.min(10, urgencyRatio * 10 + scarcityRatio * 5)
        )

        this.criticalAttributes[attr] = {
          needed,
          required: isCapacityCritical, // Only require if capacity critical
          modifier,
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
    this.riskAssessment = this.metrics.getRiskAssessment(
      this.estimatedPeopleInLineLeft
    )
  }

  private getDynamicThresholdAdjustment(): number {
    const rateAnalysis = this.getAdmissionRateAnalysis()

    // Small additional adjustment based on rate performance
    if (rateAnalysis.status === 'too_high') {
      return 0.02 // Raise threshold slightly
    } else if (rateAnalysis.status === 'too_low') {
      return -0.02 // Lower threshold slightly
    }

    return 0
  }

  /**
   *
   *  @progress calculate dynamic progress threshold.
   *
   */
  private getProgressThreshold(): number {
    const totalProcessed = this.admittedCount + this.rejectedCount
    const naturalProgress = Math.min(
      totalProcessed / Bouncer.CONFIG.TARGET_RANGE!,
      1.0
    )

    const targetProgress = Math.min(naturalProgress * 1.1, 1.0)
    const actualProgress = this.metrics.totalProgress
    const progressGap = targetProgress - actualProgress

    const sigmoid = Math.tanh(progressGap * 3.0)
    const maxAdjustment = 0.3
    const progressAdjustment = sigmoid * maxAdjustment

    // Add dynamic rate-based adjustment
    const rateAdjustment = this.getDynamicThresholdAdjustment()

    const threshold = Stats.clamp(
      Bouncer.CONFIG.BASE_THRESHOLD! - progressAdjustment + rateAdjustment,
      Bouncer.CONFIG.MIN_THRESHOLD!,
      Bouncer.CONFIG.MAX_THRESHOLD!
    )

    // Enhanced debug logging
    this.info['natural_progress'] = Stats.round(naturalProgress, 10000)
    this.info['target_progress'] = Stats.round(targetProgress, 10000)
    this.info['actual_progress'] = Stats.round(actualProgress, 10000)
    this.info['progress_gap'] = Stats.round(progressGap, 10000)
    this.info['progress_adjustment'] = Stats.round(progressAdjustment, 10000)
    this.info['rate_adjustment'] = Stats.round(rateAdjustment, 10000)
    this.info['threshold'] = Stats.round(threshold, 10000)

    return threshold
  }

  private shouldEmergencyAdmit(attributes: Record<string, boolean>): boolean {
    const spotsLeft = this.totalSpotsLeft
    const peopleLeft = this.estimatedPeopleInLineLeft
    const incompleteQuotas = this.metrics.getIncompleteConstraints()
    const totalNeeded = incompleteQuotas.reduce((sum, q) => sum + q.needed, 0)

    // Emergency if we're running out of people relative to what we need
    const peoplePerNeed = peopleLeft / Math.max(totalNeeded, 1)

    if (peoplePerNeed < 5 && spotsLeft < 100) {
      // Less than 5 people per needed quota
      const usefulAttributes = this.metrics.getUsefulAttributes(
        attributes as Attributes
      )
      return usefulAttributes.length > 0
    }

    return false
  }

  /**
   *  # Admit
   *
   *  Determine is we should admit or reject the next person.
   *
   *  @note admission goes logic here....
   */
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

    // Calculate both regular and endgame scores
    const regularScore = this.calculateAdmissionScore(personAttributes)
    const endgameScore = this.getEndgameScore(personAttributes)
    const deflationFactor = this.getScoreDeflationFactor()

    // Use higher of regular or endgame score
    const rawScore = Math.max(regularScore, endgameScore)
    const score = rawScore * deflationFactor

    const threshold = this.getProgressThreshold()

    const hasEveryAttribute = this.metrics.hasAllAttributes(
      personAttributes as any
    )
    const hasEveryCriticalAttribute =
      criticalKeys.length > 0 &&
      criticalKeys.every((attr) => personAttributes[attr]!)
    const hasSomeCriticalAttribute =
      criticalKeys.length > 0 &&
      criticalKeys.some((attr) => personAttributes[attr]!)

    if (hasEveryAttribute) this.totalUnicorns++

    // Enhanced admission criteria with endgame priority
    const shouldAdmit =
      hasEveryAttribute || // Unicorns always get in
      hasEveryCriticalAttribute || // Perfect critical matches
      (this.isEndgame() && endgameScore > 0.5) || // Endgame priority
      score > threshold || // Standard threshold
      (spotsLeft < 20 && hasSomeCriticalAttribute) || // Final desperation
      this.shouldEmergencyAdmit(personAttributes) // Last resort

    this.totalScores.push(score)
    this.info['regular_score'] = Stats.round(regularScore)
    this.info['endgame_score'] = Stats.round(endgameScore)
    this.info['raw_score'] = Stats.round(rawScore)
    this.info['deflation_factor'] = Stats.round(deflationFactor)
    this.info['last_score'] = Stats.round(score)
    this.info['threshold'] = Stats.round(threshold)
    this.info['is_endgame'] = this.isEndgame()
    this.info['critical_attrs'] = criticalKeys.length
    this.info['has_critical'] = hasEveryCriticalAttribute
    this.info['has_some_critical'] = hasSomeCriticalAttribute
    this.info['emergency_eligible'] =
      this.shouldEmergencyAdmit(personAttributes)

    this.updateCounts(personAttributes as any, score, shouldAdmit)
    return shouldAdmit
  }

  /**
   *  Calculate the persons admission score.
   *  @param attributes
   *  @returns {number}
   */
  private calculateAdmissionScore(attributes: Record<string, boolean>): number {
    if (this.allQuotasMet) return 1.0

    const usefulAttributes = this.metrics.getUsefulAttributes(
      attributes as Attributes
    )
    if (usefulAttributes.length === 0) return 0.0

    let totalScore = 0
    let hasCriticalAttribute = false
    let maxCriticalMultiplier = 1.0

    usefulAttributes.forEach((attr) => {
      const needed = this.metrics.getNeeded(attr)
      const progress = this.metrics.getProgress(attr)

      // MUCH more conservative base scoring
      const urgencyScore = Math.min(needed / 200, 1.0) // Increased divisor from 50 to 200

      // Critical modifier - track max instead of applying to each
      const criticalInfo = this.criticalAttributes[attr]
      let criticalMultiplier = 1.0

      if (criticalInfo) {
        criticalMultiplier = Math.min(criticalInfo.modifier, 8.0) // Reduced cap from 15 to 8
        maxCriticalMultiplier = Math.max(
          maxCriticalMultiplier,
          criticalMultiplier
        )
        hasCriticalAttribute = true
      }

      // MUCH more conservative bonuses
      const frequency = this.metrics.frequencies[attr]
      const rarityBonus = frequency < 0.1 ? 1.8 : frequency < 0.4 ? 1.3 : 1.0 // Reduced from 3.0/1.5
      const progressUrgency = progress < 0.2 ? 1.8 : progress < 0.5 ? 1.4 : 1.0 // Reduced from 3.0/2.0

      // Don't apply critical multiplier per attribute - apply once at the end
      const attributeScore = urgencyScore * rarityBonus * progressUrgency
      totalScore += attributeScore
    })

    // Apply critical multiplier once to total, not per attribute
    if (hasCriticalAttribute) {
      totalScore *= Math.min(maxCriticalMultiplier, 3.0) // Much lower cap
    }

    // More conservative multi-attribute bonus
    if (usefulAttributes.length > 1) {
      totalScore *= 1 + (usefulAttributes.length - 1) * 0.2 // Reduced from 0.6
    }

    // MUCH more aggressive normalization
    const normalizedScore = Math.log(totalScore + 1) / Math.log(50) // Increased denominator from 20 to 50
    return Math.min(normalizedScore, 1.5) // Reduced cap from 2.0 to 1.5
  }

  private getScoreDeflationFactor(): number {
    const currentRate =
      this.admittedCount / (this.admittedCount + this.rejectedCount)
    const targetRate =
      Bouncer.CONFIG.MAX_CAPACITY! / Bouncer.CONFIG.TARGET_RANGE! // 25%

    // More aggressive deflation - aim for 20% not 25%
    const adjustedTargetRate = targetRate * 0.8 // Target 20% instead of 25%

    if (currentRate > adjustedTargetRate * 1.4) {
      // If over 28%
      return 0.5 // Heavy deflation
    } else if (currentRate > adjustedTargetRate * 1.2) {
      // If over 24%
      return 0.7 // Medium deflation
    } else if (currentRate > adjustedTargetRate * 1.1) {
      // If over 22%
      return 0.85 // Light deflation
    } else if (currentRate < adjustedTargetRate * 0.7) {
      // If under 14%
      return 1.3 // Boost scores
    } else if (currentRate < adjustedTargetRate * 0.85) {
      // If under 17%
      return 1.15 // Light boost
    }

    return 1.0 // Good range (17% - 22%)
  }

  // Simplified estimation using metrics
  public getEstimatedPeopleWithAttributeLeftInLine(attribute: Keys): number {
    const frequency = this.metrics.frequencies[attribute]
    const peopleLeft = this.estimatedPeopleInLineLeft
    return peopleLeft * frequency
  }

  private isEndgame(): boolean {
    const spotsLeft = this.totalSpotsLeft
    const incompleteQuotas = this.metrics.getIncompleteConstraints()
    const totalNeeded = incompleteQuotas.reduce((sum, q) => sum + q.needed, 0)

    // Endgame when we have very few spots left OR need very few people
    return spotsLeft < 50 || totalNeeded < 200
  }

  private getEndgameScore(attributes: Record<string, boolean>): number {
    if (!this.isEndgame()) return 0

    const usefulAttributes = this.metrics.getUsefulAttributes(
      attributes as Attributes
    )
    const incompleteQuotas = this.metrics.getIncompleteConstraints()

    // In endgame, score ONLY based on exact needs
    let endgameScore = 0

    usefulAttributes.forEach((attr) => {
      const quota = incompleteQuotas.find((q) => q.attribute === attr)
      if (quota) {
        // Score based on how desperately we need this attribute
        const urgency = Math.min(
          quota.needed / Math.max(this.totalSpotsLeft, 1),
          5.0
        )
        const frequency = this.metrics.frequencies[attr]
        const scarcity = 1 / Math.max(frequency, 0.01) // Rarer = higher score

        endgameScore += urgency * scarcity
      }
    })

    return Math.min(endgameScore, 3.0) // Cap endgame scores
  }

  getDebugInfo(): any {
    const criticalProgress = Object.entries(this.criticalAttributes).map(
      ([attr, info]) => ({
        attribute: attr,
        needed: info?.needed || 0,
        modifier: info?.modifier || 1,
        frequency: this.metrics.frequencies[attr as keyof Attributes],
        estimated_remaining:
          this.estimatedPeopleInLineLeft *
          (this.metrics.frequencies[attr as keyof Attributes] || 0),
      })
    )

    return {
      ...this.info,
      critical_progress: criticalProgress,
      spots_left: this.totalSpotsLeft,
      people_left: this.estimatedPeopleInLineLeft,
      incomplete_quotas: this.metrics.getIncompleteConstraints().length,
    }
  }

  // Get current total progress
  public getProgress(): GameProgress<Attributes> {
    const incompleteQuotas = this.metrics
      .getIncompleteConstraints()
      .map((cp) => ({
        attribute: cp.attribute,
        needed: cp.needed,
      }))

    const criticalAttributes = Object.entries(this.criticalAttributes).map(
      ([key, value]) => (value?.required ? `⚠️:${key}` : key)
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
      config: Bouncer.CONFIG,
      quotas: incompleteQuotas.sort((a, b) => a.needed - b.needed) as any,
      admissionRate:
        this.admittedCount / (this.admittedCount + this.rejectedCount),
      admitted: this.admittedCount,
      rejected: this.rejectedCount,
      ...this.getDebugInfo(),
    }
  }

  public getOutput() {
    const analysis = this.metrics.getDetailedAnalysis()
    return {
      ...this.getProgress(),
      accuracy: Stats.percent(
        Math.abs(Bouncer.CONFIG.TARGET_RANGE - this.rejectedCount) /
          Bouncer.CONFIG.TARGET_RANGE
      ),
      scores: this.totalScores,
      metrics_analysis: analysis,
      final_summary: this.metrics.getSummary(),
    }
  }

  private getAdmissionRateAnalysis(): {
    currentRate: number
    targetRate: number
    deviation: number
    status: 'too_high' | 'too_low' | 'optimal'
    recommendation: string
  } {
    const currentRate =
      this.admittedCount / (this.admittedCount + this.rejectedCount)
    const targetRate =
      Bouncer.CONFIG.MAX_CAPACITY! / Bouncer.CONFIG.TARGET_RANGE!
    const deviation = currentRate - targetRate

    let status: 'too_high' | 'too_low' | 'optimal'
    let recommendation: string

    if (Math.abs(deviation) < 0.02) {
      // Within 2%
      status = 'optimal'
      recommendation = 'Admission rate is well-calibrated'
    } else if (deviation > 0.05) {
      // More than 5% over
      status = 'too_high'
      recommendation = 'Consider raising thresholds or reducing score bonuses'
    } else if (deviation < -0.05) {
      // More than 5% under
      status = 'too_low'
      recommendation =
        'Consider lowering thresholds or increasing score bonuses'
    } else {
      status = deviation > 0 ? 'too_high' : 'too_low'
      recommendation = 'Minor adjustment needed'
    }

    return {
      currentRate: Stats.round(currentRate, 10000),
      targetRate: Stats.round(targetRate, 10000),
      deviation: Stats.round(deviation, 10000),
      status,
      recommendation,
    }
  }
}
