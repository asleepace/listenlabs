import type { GameState, Person, PersonAttributes } from '../types'
import type { BergainBouncer } from './berghain'
import { BASE_CONFIG, type GameConfig } from '../conf/game-config'
import { Stats } from './stats'
import { Metrics } from './metrics'

interface Constraint {
  attribute: string
  minCount: number
}

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

type CriticalAttributes<Attributes extends PersonAttributes> = Partial<
  Record<keyof Attributes, { needed: number; required: boolean }>
>

// ... existing interfaces and CONFIG ...

export class Bouncer<
  Attributes extends PersonAttributes,
  Keys extends keyof Attributes = keyof Attributes
> implements BergainBouncer
{
  static CONFIG = BASE_CONFIG
  static intialize(overrides: Partial<GameConfig>) {
    Bouncer.CONFIG = Object.assign(Bouncer.CONFIG, overrides)
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

  public criticalAttributes: CriticalAttributes<Attributes> = {}
  public riskAssessment: ReturnType<(typeof this.metrics)['getRiskAssessment']>

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
    const incompleteConstraints = this.metrics.getIncompleteConstraints()

    this.criticalAttributes = {}

    incompleteConstraints.forEach((constraint) => {
      const attr = constraint.attribute
      const needed = constraint.needed
      const frequency = this.metrics.frequencies[attr]

      const estimatedRemaining = peopleInLineLeft * frequency
      const isCritical = this.riskAssessment.criticalAttributes.includes(attr)

      // Enhanced logic using estimated remaining
      const isRequired =
        needed >= totalSpotsLeft - Bouncer.CONFIG.CRITICAL_REQUIRED_THRESHOLD
      const isEstimateShort = needed > estimatedRemaining * 0.8 // Need more than 80% of estimated remaining

      // An attribute becomes critical if:
      // 1. Risk assessment flags it, OR
      // 2. Venue capacity constraint, OR
      // 3. Estimated remaining people constraint
      if (isCritical || isRequired || isEstimateShort) {
        this.criticalAttributes[attr] = {
          needed,
          required: isRequired || isEstimateShort, // Either capacity or estimate constraint makes it required
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
    if (this.rawScores.length < Bouncer.CONFIG.MIN_RAW_SCORES) {
      return Math.min(rawScore / 5.0, 1.0)
    }
    const avgScore = Stats.average(this.rawScores) || 0.5
    const stdDev = Stats.stdDev(this.rawScores) || 0.2
    return 1.0 / (1.0 + Math.exp(-(rawScore - avgScore) / stdDev))
  }

  // Improved threshold calculation using metrics
  // private getProgressThresholdOld(): number {
  //   const totalProcessed = this.admittedCount + this.rejectedCount
  //   const expectedProgress = Math.min(totalProcessed / CONFIG.TARGET_RANGE, 1.0)
  //   const totalProgress = this.metrics.totalProgress

  //   // Use metrics efficiency analysis
  //   const efficiency = this.metrics.getEfficiencyMetrics()
  //   this.riskAssessment = this.metrics.getRiskAssessment(
  //     this.estimatedPeopleInLineLeft
  //   )

  //   // Adjust sensitivity based on risk
  //   const baseSensitivity = 1.0 // lower=less sensitive
  //   const riskMultiplier = Stats.clamp(
  //     this.riskAssessment.riskScore / 5.0,
  //     0.5,
  //     2.0
  //   )
  //   const sensitivity = baseSensitivity * riskMultiplier

  //   const delta = (expectedProgress - totalProgress) * 0.5
  //   const threshold = Math.max(
  //     CONFIG.MIN_THRESHOLD,
  //     Math.min(
  //       CONFIG.MAX_THRESHOLD,
  //       CONFIG.BASE_THRESHOLD - delta * sensitivity
  //     )
  //   )

  //   // Enhanced logging with metrics insights
  //   this.info['expected_progress'] = Stats.round(expectedProgress, 10_000)
  //   this.info['total_progress'] = Stats.round(totalProgress, 10_000)
  //   this.info['efficiency'] = efficiency.actualEfficiency
  //   this.info['risk_score'] = this.riskAssessment.riskScore
  //   this.info['threshold'] = Stats.round(threshold, 10_000)

  //   return threshold
  // }

  private getProgressThreshold(): number {
    const totalProcessed = this.admittedCount + this.rejectedCount
    const expectedProgress = Math.min(
      totalProcessed / Bouncer.CONFIG.TARGET_RANGE,
      1.0
    )
    const totalProgress = this.metrics.totalProgress

    // Calculate how far off we are from expected
    const progressGap = expectedProgress - totalProgress

    // Use a sigmoid function to create smooth, aggressive adjustments
    const sigmoid = Math.tanh(progressGap * 3.0) // 3.0 controls steepness

    // Base adjustment range - when behind, go much lower; when ahead, go higher
    const maxAdjustment = 0.15 // Can swing threshold by ±15%
    const adjustment = sigmoid * maxAdjustment

    const threshold = Stats.clamp(
      Bouncer.CONFIG.BASE_THRESHOLD - adjustment, // Subtract because behind = lower threshold
      Bouncer.CONFIG.MIN_THRESHOLD,
      Bouncer.CONFIG.MAX_THRESHOLD
    )

    // Enhanced logging
    this.info['progress_gap'] = Stats.round(progressGap, 10000)
    this.info['sigmoid_adj'] = Stats.round(sigmoid, 100)
    this.info['threshold_adj'] = Stats.round(adjustment, 100)
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

    if (criticalKeys.length > 0) {
      const requiredAttributes = Object.entries(this.criticalAttributes)
        .filter(([_, value]) => value?.required)
        .map(([key, _]) => key)

      const isEndgame = this.metrics.totalProgress > 0.95
      const hasMultipleRequired = requiredAttributes.length > 1

      if (isEndgame && hasMultipleRequired) {
        const totalNeeded = Object.values(this.criticalAttributes)
          .filter((value) => value?.required)
          .reduce((sum, value) => sum + (value?.needed || 0), 0)

        if (totalNeeded > this.totalSpotsLeft * 1.5) {
          const hasAnyCritical = requiredAttributes.some(
            (attr) => personAttributes[attr as any]
          )
          if (!hasAnyCritical) return false
        } else {
          // Still have reasonable space, use AND logic
          for (const [key, value] of Object.entries(this.criticalAttributes)) {
            if (!value || !personAttributes) continue
            if (value.required && !personAttributes[key as any]) return false
          }
        }
      } else {
        // Normal case: not endgame OR single required attribute - use AND logic
        for (const [key, value] of Object.entries(this.criticalAttributes)) {
          if (!value || !personAttributes) continue
          if (value.required && !personAttributes[key as any]) return false
        }
      }
    }

    // Early game strategy using metrics
    if (this.admittedCount < Bouncer.CONFIG.MIN_RAW_SCORES) {
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

      const quotaRate = constraint.minCount / Bouncer.CONFIG.MAX_CAPACITY

      // If this attribute is much more common than its quota rate AND we're early game
      const isOverabundant = frequency > quotaRate * 2.0
      const isEarlyGame = totalProcessed < Bouncer.CONFIG.TARGET_RANGE * 0.5

      // Base scoring factors
      const targetProgress = Math.min(
        totalProcessed / Bouncer.CONFIG.TARGET_RANGE,
        1.0
      )
      const actualProgress = this.metrics.getProgress(attr)
      const progressGap = targetProgress - actualProgress
      const urgency =
        progressGap > 0 ? progressGap * Bouncer.CONFIG.URGENCY_MODIFIER : 0

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
        componentScore *= 1.5 // Strong boost for required
      } else if (critical) {
        componentScore *= 1.1 // Small boost for critical
      }

      if (isOverabundant && isEarlyGame) {
        // Reduce importance of overabundant attributes early
        componentScore *= 0.7
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
          correlationBonus +=
            pair.correlation * Bouncer.CONFIG.CORRELATION_BONUS
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
            Math.abs(pair.correlation) * Bouncer.CONFIG.RARE_PERSON_BONUS
        }
      })

      score += componentScore * (1 + correlationBonus)
    })

    // Multi-attribute bonus
    const usefulCount = usefulAttributes.length
    if (usefulCount > 1) {
      score *= 1 + Bouncer.CONFIG.MULTI_ATTRIBUTE_BONUS * (usefulCount - 1)
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
      ([key, value]) => (value?.required ? `⚠️:${key}` : key)
    )

    // Enhanced info with metrics insights
    const metricsAnalysis = this.metrics.getDetailedAnalysis(
      this.estimatedPeopleInLineLeft
    )
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
    const analysis = this.metrics.getDetailedAnalysis(
      this.estimatedPeopleInLineLeft
    )
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
