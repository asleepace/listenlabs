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
  // Threshold settings - allow wider range for rate control
  BASE_THRESHOLD: 0.45, // Lower base for more lenient starting point
  MIN_THRESHOLD: 0.25, // Much lower floor for aggressive recovery
  MAX_THRESHOLD: 0.95, // Higher ceiling for selectivity

  // Game timing - not used in constant rate but keep reasonable
  TARGET_RANGE: 4000, // Keep existing

  // Scoring modifiers - crucial for constant rate approach
  URGENCY_MODIFIER: 2.0, // Lower since urgency handled by rate control
  MULTI_ATTRIBUTE_BONUS: 0.5, // High - people with multiple attributes are very valuable

  // Critical detection - more aggressive
  CRITICAL_REQUIRED_THRESHOLD: 20, // Higher buffer before panic mode
  CRITICAL_IN_LINE_RATIO: 0.8, // More aggressive critical detection
  CRITICAL_CAPACITY_RATIO: 0.9, // More aggressive

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

  // Enhanced critical attributes using metrics insights
  private getCriticalAttributes(): CriticalAttributes<Attributes> {
    if (this.admittedCount < 50) {
      return {}
    }

    const peopleInLineLeft = this.estimatedPeopleInLineLeft
    const totalSpotsLeft = this.totalSpotsLeft

    // Use metrics risk assessment
    const incompleteConstraints = this.metrics.getIncompleteConstraints()

    this.criticalAttributes = {}

    incompleteConstraints.forEach((constraint) => {
      const attr = constraint.attribute
      const needed = constraint.needed
      const frequency = this.metrics.frequencies[attr]

      const estimatedRemaining = peopleInLineLeft * frequency
      const isCritical = this.riskAssessment.criticalAttributes.includes(attr)

      const spotsBuffer = Math.max(0, totalSpotsLeft - needed)
      const maxBuffer = Math.max(1, totalSpotsLeft * 0.2) // 20% of spots left as max buffer
      const urgencyRatio = 1 - spotsBuffer / maxBuffer
      const modifier = 1 + Math.max(0, Math.min(1, urgencyRatio)) * 2 // 1.0 to 3.0 range

      // Enhanced logic using estimated remaining
      const isRequired =
        needed >=
        Math.max(
          1,
          totalSpotsLeft - Bouncer.CONFIG.CRITICAL_REQUIRED_THRESHOLD!
        )

      const isEstimateShort = needed > estimatedRemaining * 0.95 // Need more than 80% of estimated remaining

      // An attribute becomes critical if:
      // 1. Risk assessment flags it, OR
      // 2. Venue capacity constraint, OR
      // 3. Estimated remaining people constraint
      if (isCritical || isRequired || isEstimateShort) {
        this.criticalAttributes[attr] = {
          needed,
          required: true,
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
  }

  // normalize score values
  private normalizeScore(rawScore: number): number {
    // Simple normalization for the log-based scoring
    return Math.min(rawScore / 4.0, 1.0) // Divide by ~max expected score
  }

  // calculate progress threshold
  private getProgressThreshold(): number {
    const totalNeeded = this.constraints.reduce(
      (total, current) => total + current.minCount,
      0
    )
    const targetAdmissionRate = totalNeeded / (Bouncer.CONFIG.TOTAL_PEOPLE || 1)
    const currentRate =
      this.admittedCount / (this.admittedCount + this.rejectedCount)

    this.rateGap = currentRate - targetAdmissionRate
    const adjustment = this.rateGap * 0.5 // MOVE THE 0.2 HERE, reduce from 0.5

    const threshold = Stats.clamp(
      Bouncer.CONFIG.BASE_THRESHOLD! + adjustment,
      Bouncer.CONFIG.MIN_THRESHOLD!,
      0.95 // RAISE MAX_THRESHOLD from 0.8
    )

    // Debug logging
    this.info['target_rate'] = Stats.round(targetAdmissionRate, 10000)
    this.info['current_rate'] = Stats.round(currentRate, 10000)
    this.info['rate_gap'] = Stats.round(this.rateGap, 10000)
    this.info['threshold'] = Stats.round(threshold, 10000)

    return threshold
  }

  private getProgressThresholdWave(): number {
    const totalProcessed = this.admittedCount + this.rejectedCount
    const naturalProgress = Math.min(
      totalProcessed / Bouncer.CONFIG.TARGET_RANGE!,
      1.0
    )

    // Target being 10% ahead of the natural schedule
    const targetProgress = Math.min(naturalProgress * 1.05, 1.0)
    const actualProgress = this.metrics.totalProgress

    // Now the gap measures against the "10% ahead" target
    const progressGap = targetProgress - actualProgress

    const sigmoid = Math.tanh(progressGap * 4.0)
    const maxAdjustment = 0.25
    const adjustment = sigmoid * maxAdjustment

    const threshold = Stats.clamp(
      Bouncer.CONFIG.BASE_THRESHOLD - adjustment,
      Bouncer.CONFIG.MIN_THRESHOLD,
      Bouncer.CONFIG.MAX_THRESHOLD
    )

    // Debug logging
    this.info['natural_progress'] = Stats.round(naturalProgress, 10000)
    this.info['target_progress'] = Stats.round(targetProgress, 10000)
    this.info['actual_progress'] = Stats.round(actualProgress, 10000)
    this.info['progress_gap'] = Stats.round(progressGap, 10000)
    this.info['threshold'] = Stats.round(threshold, 10000)

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

    // In admit() method, replace the complex critical check with:
    if (criticalKeys.length > 0) {
      for (const [key, value] of Object.entries(this.criticalAttributes)) {
        if (!value || !personAttributes) continue
        if (value.required && !personAttributes[key as any]) return false
      }
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

  // calculate admission scores
  private calculateAdmissionScore(attributes: Record<string, boolean>): number {
    if (this.allQuotasMet) return 1.0

    const usefulAttributes = this.metrics.getUsefulAttributes(
      attributes as Attributes
    )
    if (usefulAttributes.length === 0) return 0.0

    let totalScore = 0
    const allProgresses = this.constraints.map((c) =>
      this.metrics.getProgress(c.attribute)
    )
    const avgProgress = Stats.average(allProgresses)

    usefulAttributes.forEach((attr) => {
      const needed = this.metrics.getNeeded(attr)
      const progress = this.metrics.getProgress(attr)

      // Base urgency: 0-1 based on how much we need
      const urgencyScore = Math.min(needed / 200, 1.0)

      // Proactive priority: boost attributes that are falling behind BEFORE they become critical
      let priorityMultiplier = 1.0
      if (progress < avgProgress * 0.8) {
        priorityMultiplier = 2.0 // 2x boost for lagging quotas
      } else if (progress > avgProgress * 1.2) {
        priorityMultiplier = 0.3 // Penalize overfilled quotas
      }

      // Early intervention: boost rare attributes early
      const frequency = this.metrics.frequencies[attr]
      const rarityBoost = frequency < 0.1 ? 1.5 : 1.0 // 1.5x for very rare attributes
      const criticalModifier = this.criticalAttributes[attr]?.modifier || 1

      const attributeScore =
        urgencyScore * priorityMultiplier * rarityBoost * criticalModifier
      totalScore += attributeScore
    })

    // Multi-attribute bonus: strong reward for efficiency
    if (usefulAttributes.length > 1) {
      totalScore *= 1 + (usefulAttributes.length - 1) * 0.3
    }

    return Math.min(totalScore / usefulAttributes.length, 1.0)
  }

  // Simplified estimation using metrics
  public getEstimatedPeopleWithAttributeLeftInLine(attribute: Keys): number {
    const frequency = this.metrics.frequencies[attribute]
    const peopleLeft = this.estimatedPeopleInLineLeft
    return peopleLeft * frequency
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
}
