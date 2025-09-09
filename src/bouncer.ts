import type { GameState, PersonAttributes, ScenarioAttributes } from './types'
import type { BergainBouncer } from './core/berghain'
import { BASE_CONFIG, type GameConfig } from './conf/game-config'
import { Stats } from './math/statistics'
import { Metrics, type AttributeRisk } from './math/metrics'
import { Score } from './math/score'
import {
  DeflationPIDController,
  PID_PRESETS,
} from './math/deflation-controller'

interface GameQuota {
  attribute: keyof ScenarioAttributes
  needed: number
}

interface GameProgress {
  config: GameConfig
  critical: string[]
  quotas: GameQuota[]
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

type CriticalAttributes = Partial<Record<keyof ScenarioAttributes, Critical>>

// fine tune config here...

const TUNED_CONFIG: Partial<GameConfig> = {
  // Most impactful
  BASE_THRESHOLD: 0.42, // Lower = More lenient
  TARGET_RATE: 0.25, // Raise from 0.19 to match 1000/4000 = 25%
  TARGET_RANGE: 4000, // Back to 4000 from 3000/3050
  MIN_THRESHOLD: 0.2, // Lower floor
  MAX_THRESHOLD: 0.8, // Keep reasonable ceiling
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

  private metrics: Metrics
  public progress: GameProgress
  public state: GameState
  public info: Record<string, any> = {}

  public totalScores: number[] = []
  public rawScores: number[] = []
  public totalAdmittedScores: number[] = []
  public lowestAcceptedScore = Infinity
  public totalUnicorns = 0
  public rateGap = 0

  public criticalAttributes: CriticalAttributes = {}
  public riskAssessment: AttributeRisk
  public deflationController: DeflationPIDController

  constructor(initialData: GameState) {
    this.deflationController = new DeflationPIDController(
      PID_PRESETS.RESPONSIVE
    )
    this.metrics = new Metrics(initialData.game)
    this.state = initialData
    this.progress = this.getProgress()
    this.riskAssessment = this.metrics.getRiskAssessment(
      this.estimatedPeopleInLineLeft
    )
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

  private getScoreDeflationFactor(): number {
    const currentRate =
      this.admittedCount / (this.admittedCount + this.rejectedCount)
    const targetRate = Bouncer.CONFIG.TARGET_RATE || 0.25

    return this.deflationController.getDeflationFactor(currentRate, targetRate)
  }

  // Delegate counting methods to metrics
  public getCount(attribute: keyof ScenarioAttributes): number {
    return this.metrics.getCount(attribute)
  }

  public getCorrelations(attribute: keyof ScenarioAttributes) {
    return this.metrics.correlations[attribute]
  }

  public getFrequency(attribute: keyof ScenarioAttributes): number {
    return this.metrics.frequencies[attribute]!
  }

  private getPeopleNeeded(attribute: keyof ScenarioAttributes): number {
    return this.metrics.getNeeded(attribute)
  }

  private getCriticalAttributes(): CriticalAttributes {
    if (this.admittedCount < 50) return {}

    const peopleInLineLeft = this.estimatedPeopleInLineLeft
    const totalSpotsLeft = this.totalSpotsLeft
    const incompleteConstraints = this.metrics.getIncompleteConstraints()

    this.criticalAttributes = {}

    incompleteConstraints.forEach((constraint) => {
      const attr = constraint.attribute
      const needed = constraint.needed
      const frequency = this.metrics.frequencies[attr]!
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

    // Use actual quota progress for target, not admission rate
    const targetQuotaProgress = Math.min(naturalProgress * 1.1, 1.0)
    const actualQuotaProgress = this.metrics.totalProgress
    const progressGap = targetQuotaProgress - actualQuotaProgress

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
    this.info['target_quota_progress'] = Stats.round(targetQuotaProgress, 10000)
    this.info['actual_quota_progress'] = Stats.round(actualQuotaProgress, 10000)
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

    // const currentRate =
    //   this.admittedCount / (this.admittedCount + this.rejectedCount)
    const targetRate = Bouncer.CONFIG.TARGET_RATE ?? 0.25
    // const earlyBoost =
    //   this.admittedCount < 500 && currentRate < targetRate ? 0.9 : 1.0

    // Calculate both regular and endgame scores
    let regularScore = Score.calculateAdmissionScore(
      {
        allQuotasMet: this.allQuotasMet,
        attributes: personAttributes,
        criticalAttributes: this.criticalAttributes,
        admittedCount: this.admittedCount,
        metrics: this.metrics,
      },
      Score.PRESETS.CONSERVATIVE
    )

    const endgameScore =
      regularScore < 0.3
        ? Score.calculateEndgameScore(
            {
              allQuotasMet: this.allQuotasMet,
              totalSpotsLeft: this.totalSpotsLeft,
              attributes: personAttributes,
              criticalAttributes: this.criticalAttributes,
              admittedCount: this.admittedCount,
              metrics: this.metrics,
              isEndgame: this.isEndgame(),
            },
            {
              maxEndgameScore: 1.5,
            }
          )
        : 0

    // Calculate score default factor (see file for more implementations)
    // const deflationFactor = Score.getScoreDeflationFactorCombined({
    //   admittedCount: this.admittedCount,
    //   rejectedCount: this.rejectedCount,
    //   targetRate,
    // })
    const deflationFactor = this.getScoreDeflationFactor()

    // Use higher of regular or endgame score
    const rawScore = Math.max(regularScore, endgameScore)
    const score = rawScore * deflationFactor

    const threshold = this.getProgressThreshold()
    const effectiveThreshold = threshold

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
      hasEveryAttribute ||
      hasEveryCriticalAttribute ||
      (this.isEndgame() && endgameScore > 0.5) ||
      score > effectiveThreshold || // Use boosted threshold
      (spotsLeft < 20 && hasSomeCriticalAttribute) ||
      this.shouldEmergencyAdmit(personAttributes)

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

  // Simplified estimation using metrics
  public getEstimatedPeopleWithAttributeLeftInLine(
    attribute: keyof ScenarioAttributes
  ): number {
    const frequency = this.metrics.frequencies[attribute]!
    const peopleLeft = this.estimatedPeopleInLineLeft
    return peopleLeft * frequency
  }

  private isEndgame(): boolean {
    const spotsLeft = this.totalSpotsLeft
    return spotsLeft > 0 && spotsLeft <= 50
  }

  // private isEndgame(): boolean {
  //   const spotsLeft = this.totalSpotsLeft
  //   const incompleteQuotas = this.metrics.getIncompleteConstraints()
  //   const totalNeeded = incompleteQuotas.reduce((sum, q) => sum + q.needed, 0)

  //   // Trigger endgame when mathematically necessary
  //   return (
  //     spotsLeft > 0 &&
  //     (spotsLeft <= 50 || // Original condition
  //       totalNeeded >= spotsLeft) // New: when impossible math
  //   )
  // }

  getDebugInfo(): any {
    const criticalProgress = Object.entries(this.criticalAttributes).map(
      ([attr, info]) => ({
        attribute: attr,
        needed: info?.needed || 0,
        modifier: info?.modifier || 1,
        frequency: this.metrics.frequencies[attr],
        estimated_remaining:
          this.estimatedPeopleInLineLeft *
          (this.metrics.frequencies[attr] || 0),
      })
    )

    const rateAnalysis = this.getAdmissionRateAnalysis()
    const currentRejections = this.rejectedCount
    const targetRejections = 3000
    const rejectionGap = currentRejections - targetRejections

    const currentRate =
      this.admittedCount / (this.admittedCount + this.rejectedCount)
    const targetRate = Bouncer.CONFIG.TARGET_RATE || 0.25

    return {
      ...this.info,
      rate_analysis: rateAnalysis,
      critical_progress: criticalProgress,
      spots_left: this.totalSpotsLeft,
      people_left: this.estimatedPeopleInLineLeft,
      incomplete_quotas: this.metrics.getIncompleteConstraints().length,
      rejection_analysis: {
        current_rejections: currentRejections,
        target_rejections: targetRejections,
        rejection_gap: rejectionGap,
        on_track: Math.abs(rejectionGap) < 200,
      },
      pid_debug: this.deflationController.getDebugInfo(currentRate, targetRate),
    }
  }

  // Get current total progress
  public getProgress(): GameProgress {
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
    const targetRate = Bouncer.CONFIG.TARGET_RATE || 0.19 // Use config value consistently
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
