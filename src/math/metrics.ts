import type {
  GameState,
  PersonAttributes,
  ScenarioAttributes,
  Keys,
} from '../types'
import { Stats } from './statistics'

export type Frequencies = {
  readonly [K in keyof ScenarioAttributes]: number
}

export type Correlations = {
  readonly [K in keyof ScenarioAttributes]: {
    readonly [J in keyof ScenarioAttributes]: number
  }
}

export type Constraint = {
  readonly attribute: Keys
  readonly minCount: number
  count: number
}

export type ConstraintProgress = {
  readonly attribute: keyof ScenarioAttributes
  readonly current: number
  readonly needed: number
  readonly progress: number
  readonly completed: boolean
}

export type AttributeStats = {
  readonly attribute: keyof ScenarioAttributes
  readonly frequency: number
  readonly rarity: number
  readonly quotaRate: number
  readonly overdemanded: boolean
}

export type AttributeRisk = {
  criticalAttributes: (keyof ScenarioAttributes)[]
  riskScore: number
  timeRemaining: number
  feasibilityScore: number
}

export class Metrics {
  private _constraints: Constraint[]
  private _frequencies: Frequencies
  private _correlations: Correlations
  private _attributeStats: Map<Keys, AttributeStats>
  private _riskAssessment: AttributeRisk
  private _correlationCache = new Map()

  constructor(public readonly gameData: GameState['game']) {
    this._constraints = gameData.constraints.map(
      (constraint): Constraint => ({
        attribute: constraint.attribute,
        minCount: constraint.minCount,
        count: 0,
      })
    )

    this._frequencies = gameData.attributeStatistics
      .relativeFrequencies as Frequencies
    this._correlations = gameData.attributeStatistics
      .correlations as Correlations
    this._attributeStats = this.preCalculateAttributeStats()
    this._riskAssessment = this.getRiskAssessment(10_000)
  }

  // Getters
  get constraints(): readonly Constraint[] {
    return this._constraints
  }

  get frequencies(): Frequencies {
    return this._frequencies
  }

  get correlations(): Correlations {
    return this._correlations
  }

  get attributeStats(): ReadonlyMap<Keys, AttributeStats> {
    return this._attributeStats
  }

  get totalConstraints(): number {
    return this._constraints.length
  }

  get completedConstraints(): number {
    return this._constraints.filter((c) => c.count >= c.minCount).length
  }

  get allConstraintsMet(): boolean {
    return this._constraints.every((c) => c.count >= c.minCount)
  }

  get totalProgress(): number {
    if (this._constraints.length === 0) return 1
    return (
      this._constraints.reduce((total, constraint) => {
        const progress = Math.min(constraint.count / constraint.minCount, 1)
        return total + progress
      }, 0) / this._constraints.length
    )
  }

  getFrequency(attr: keyof ScenarioAttributes) {
    if (!(attr in this.frequencies))
      throw new Error('Missing frequency:' + attr)
    return this.frequencies[attr]!
  }

  // Pre-calculate useful statistics
  private preCalculateAttributeStats(): Map<Keys, AttributeStats> {
    const stats = new Map<Keys, AttributeStats>()

    // Make capacity configurable instead of hardcoded 1000
    const totalCapacity =
      this.gameData.constraints.reduce((sum, c) => sum + c.minCount, 0) || 1000

    for (const constraint of this._constraints) {
      const attr = constraint.attribute
      const frequency = this._frequencies[attr] || 0
      const quotaRate = constraint.minCount / totalCapacity

      stats.set(attr as Keys, {
        attribute: attr,
        frequency,
        rarity: 1 / Math.max(frequency, 0.01),
        quotaRate,
        overdemanded: quotaRate > frequency * 1.5,
      })
    }

    return stats
  }

  // Counting methods
  updateCount(attribute: Keys, increment: number = 1): void {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (constraint) {
      constraint.count += increment
    }
  }

  updateCounts(attributes: Partial<ScenarioAttributes>): void {
    Object.entries(attributes).forEach(([attr, hasAttribute]) => {
      if (hasAttribute) {
        this.updateCount(attr as Keys)
      }
    })
  }

  getCount(attribute: Keys): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    return constraint?.count ?? 0
  }

  getNeeded(attribute: Keys): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (!constraint) return 0
    return Math.max(0, constraint.minCount - constraint.count)
  }

  getProgress(attribute: Keys): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (!constraint) return 0
    return Math.min(constraint.count / constraint.minCount, 1)
  }

  isCompleted(attribute: Keys): boolean {
    return this.getNeeded(attribute) === 0
  }

  // Analysis methods
  getConstraintProgress(): ConstraintProgress[] {
    return this._constraints.map((constraint) => ({
      attribute: constraint.attribute,
      current: constraint.count,
      needed: Math.max(0, constraint.minCount - constraint.count),
      progress: Math.min(constraint.count / constraint.minCount, 1),
      completed: constraint.count >= constraint.minCount,
    }))
  }

  getIncompleteConstraints(): ConstraintProgress[] {
    return this.getConstraintProgress().filter((cp) => !cp.completed)
  }

  getMostNeededAttributes(): Keys[] {
    return this.getIncompleteConstraints()
      .sort((a, b) => b.needed - a.needed)
      .map((cp) => cp.attribute as Keys)
  }

  getLeastProgressAttributes(): Keys[] {
    return this.getIncompleteConstraints()
      .sort((a, b) => a.progress - b.progress)
      .map((cp) => cp.attribute as Keys)
  }

  getOverdemandedAttributes(): Keys[] {
    return Array.from(this._attributeStats.entries())
      .filter(
        ([_, stats]) =>
          stats.overdemanded && !this.isCompleted(stats.attribute as Keys)
      )
      .map(([attr, _]) => attr)
  }

  getRarestAttributes(): Keys[] {
    return Array.from(this._attributeStats.entries())
      .filter(([attr, _]) => !this.isCompleted(attr))
      .sort(([_, a], [__, b]) => b.rarity - a.rarity)
      .map(([attr, _]) => attr)
  }

  // 5. Add validation method
  validateData(): { isValid: boolean; errors: string[] } {
    const errors: string[] = []

    // Check constraints
    if (this._constraints.length === 0) {
      errors.push('No constraints defined')
    }

    // Check frequencies sum (should be reasonable)
    const totalFreq = Object.values(this._frequencies).reduce(
      (sum, freq) => sum + freq,
      0
    )
    if (totalFreq > 5) {
      // Arbitrary reasonable limit
      errors.push(`Total frequency sum seems high: ${totalFreq}`)
    }

    // Check for missing frequency data
    this._constraints.forEach((constraint) => {
      if (!this._frequencies[constraint.attribute]) {
        errors.push(
          `Missing frequency data for ${String(constraint.attribute)}`
        )
      }
    })

    return {
      isValid: errors.length === 0,
      errors,
    }
  }

  getPositivelyCorrelated(
    attribute: keyof ScenarioAttributes,
    threshold: number = 0.3
  ): (keyof ScenarioAttributes)[] {
    const cacheKey = `${String(attribute)}_pos_${threshold}`

    if (this._correlationCache.has(cacheKey)) {
      return this._correlationCache.get(cacheKey)!
    }

    const correlations = this._correlations[attribute]
    if (!correlations) return []

    const result = Object.entries(correlations)
      .filter(([_, correlation]) => (correlation as number) > threshold)
      .map(([attr, _]) => attr)
      .filter((attr) => attr !== attribute)

    this._correlationCache.set(cacheKey, result)
    return result
  }

  getNegativelyCorrelated(
    attribute: keyof ScenarioAttributes,
    threshold: number = -0.3
  ): (keyof ScenarioAttributes)[] {
    const correlations = this._correlations[attribute]
    if (!correlations) return []

    return Object.entries(correlations)
      .filter(([_, correlation]) => (correlation as number) < threshold)
      .map(([attr, _]) => attr as keyof ScenarioAttributes)
      .filter((attr): attr is keyof ScenarioAttributes => attr !== attribute)
  }

  getCorrelation(attr1: Keys, attr2: Keys): number {
    return this._correlations[attr1]?.[attr2] ?? 0
  }

  getOverfillThreshold(frequency: number): number {
    if (frequency < 0.05) return 0.95  // Very rare - allow near completion
    if (frequency < 0.1) return 0.92   // Rare - allow high completion
    
    // Your existing logic for common attributes
    const baseThreshold = 0.78
    const frequencyMultiplier = 0.4
    const minThreshold = 0.85
    const maxThreshold = 0.98
    
    return Math.max(
      minThreshold,
      Math.min(maxThreshold, baseThreshold + frequency * frequencyMultiplier)
    )
  }

  // Utility methods for person evaluation
  getUsefulAttributes(
    personAttributes: Partial<ScenarioAttributes>,
    isEndGame: boolean
  ): (keyof ScenarioAttributes)[] {
    return Object.entries(personAttributes)
      .filter(([attr, hasAttr]) => {
        if (!hasAttr) return false
        // ignore overfills in the endgame
        if (isEndGame) return !this.isCompleted(attr)

        if (this.isCompleted(attr)) return false

        const progress = this.getProgress(attr)
        const frequency = this.frequencies[attr] || 0
        // Dynamic overfill threshold based on rarity
        // const overfillThreshold = Math.min(0.95, 0.85 + frequency * 0.2)
        // More aggressive separation
        const overfillThreshold = this.getOverfillThreshold(frequency)
        return progress < overfillThreshold
      })
      .map(([attr, _]) => attr as keyof ScenarioAttributes)
  }

  countUsefulAttributes(
    personAttributes: Partial<ScenarioAttributes>,
    isEndGame: boolean
  ): number {
    // ignore overfill in endgame
    return this.getUsefulAttributes(personAttributes, isEndGame).length
  }

  hasAttribute(
    personAttributes: Partial<ScenarioAttributes>,
    attribute: Keys
  ): boolean {
    return Boolean(personAttributes[attribute])
  }

  hasAllAttributes(personAttributes: Partial<ScenarioAttributes>): boolean {
    return this._constraints.every((constraint) =>
      Boolean(personAttributes[constraint.attribute])
    )
  }

  hasAnyNeededAttribute(
    personAttributes: Partial<ScenarioAttributes>
  ): boolean {
    return this._constraints.some(
      (constraint) =>
        !this.isCompleted(constraint.attribute as Keys) &&
        Boolean(personAttributes[constraint.attribute])
    )
  }

  // Reset counts (for testing or new games)
  resetCounts(): void {
    this._constraints.forEach((constraint) => {
      constraint.count = 0
    })
  }

  // Statistical analysis methods using Stats helpers
  getProgressDistribution(): {
    mean: number
    median: number
    stdDev: number
    min: number
    max: number
    quartiles: [number, number, number]
  } {
    const progressValues = this._constraints.map((c) =>
      this.getProgress(c.attribute as Keys)
    )
    return {
      mean: Stats.average(progressValues),
      median: Stats.median(progressValues),
      stdDev: Stats.stdDev(progressValues),
      min: Stats.min(progressValues),
      max: Stats.max(progressValues),
      quartiles: Stats.quartiles(progressValues),
    }
  }

  getProgressVariability(): {
    isBalanced: boolean
    coefficient: number
    outliers: Keys[]
  } {
    const progressValues = this._constraints.map((c) =>
      this.getProgress(c.attribute as Keys)
    )
    const mean = Stats.average(progressValues)
    const stdDev = Stats.stdDev(progressValues)
    const coefficient = mean > 0 ? stdDev / mean : 0

    // Find outliers (progress > 2 std devs from mean)
    const outliers = this._constraints
      .filter((c) => {
        const progress = this.getProgress(c.attribute as Keys)
        const zScore = Stats.zScore(progress, progressValues)
        return Math.abs(zScore) > 2
      })
      .map((c) => c.attribute as Keys)

    return {
      isBalanced: coefficient < 0.3, // Low coefficient of variation
      coefficient,
      outliers,
    }
  }

  getQuotaDifficulty(): Map<
    Keys,
    {
      difficulty: number
      rank: number
      factors: {
        rarity: number
        overdemand: number
        correlation: number
      }
    }
  > {
    const difficulties = new Map()

    this._constraints.forEach((constraint) => {
      const attr = constraint.attribute
      const stats = this._attributeStats.get(attr as Keys)!
      const correlatedAttrs = this.getPositivelyCorrelated(attr, 0.2)

      // Calculate difficulty factors
      const rarity = stats.rarity
      const overdemand = stats.overdemanded ? 2.0 : 1.0
      const correlation = correlatedAttrs.length > 0 ? 0.8 : 1.2 // Easier if correlated

      const difficulty = rarity * overdemand * correlation

      difficulties.set(attr, {
        difficulty,
        rank: 0, // Will be set after sorting
        factors: { rarity, overdemand, correlation },
      })
    })

    // Add ranks
    const sorted = Array.from(difficulties.entries()).sort(
      ([_, a], [__, b]) => b.difficulty - a.difficulty
    )

    sorted.forEach(([attr, data], index) => {
      data.rank = index + 1
    })

    return difficulties
  }

  getEfficiencyMetrics(): {
    targetEfficiency: number
    actualEfficiency: number
    wastedCapacity: number
    recommendations: string[]
  } {
    const progressValues = this._constraints.map((c) =>
      this.getProgress(c.attribute as Keys)
    )
    const targetEfficiency = Stats.average(progressValues)

    // Calculate how much "wasted" progress we have (over-filled quotas)
    const excessProgress = progressValues
      .map((p) => Math.max(0, p - 1))
      .reduce((sum, excess) => sum + excess, 0)

    const actualEfficiency =
      targetEfficiency - excessProgress / this._constraints.length
    const wastedCapacity = excessProgress * 100 // As percentage

    const recommendations: string[] = []

    if (wastedCapacity > 10) {
      recommendations.push('Reduce admission of over-filled attributes')
    }

    const variability = this.getProgressVariability()
    if (!variability.isBalanced) {
      recommendations.push('Focus on balancing quota progress')
    }

    const incomplete = this.getIncompleteConstraints()
    if (incomplete.length > 0) {
      const slowest = incomplete.sort((a, b) => a.progress - b.progress)[0]!
      recommendations.push(
        `Prioritize ${String(slowest.attribute)} (${Math.round(
          slowest.progress * 100
        )}% complete)`
      )
    }

    return {
      targetEfficiency: Stats.round(targetEfficiency, 100),
      actualEfficiency: Stats.round(actualEfficiency, 100),
      wastedCapacity: Stats.round(wastedCapacity, 100),
      recommendations,
    }
  }

  getRiskAssessment(peopleRemaining: number): AttributeRisk {
    const incomplete = this.getIncompleteConstraints()

    // Add early return for edge case
    if (incomplete.length === 0) {
      return {
        criticalAttributes: [],
        riskScore: 0,
        timeRemaining: 1 - this.totalProgress,
        feasibilityScore: 1,
      }
    }

    const sortedIncomplete = incomplete.sort((a, b) => {
      const statsA = this._attributeStats.get(a.attribute as Keys)
      const statsB = this._attributeStats.get(b.attribute as Keys)

      // Add null checks
      if (!statsA || !statsB) return 0

      return statsA.frequency - statsB.frequency
    })

    let availablePeople = peopleRemaining

    const riskFactors = sortedIncomplete.map((constraint) => {
      const stats = this._attributeStats.get(constraint.attribute as Keys)

      // Add null check
      if (!stats) return 0

      const needed = constraint.needed
      const frequency = stats.frequency

      const expectedWithAttribute = availablePeople * frequency
      const riskRatio = needed / Math.max(expectedWithAttribute, 1)

      const peopleUsed = Math.min(needed, expectedWithAttribute * 0.8)
      availablePeople = Math.max(0, availablePeople - peopleUsed)

      return Stats.clamp(riskRatio * 3, 0, 10)
    })

    const riskScore = riskFactors.length > 0 ? Stats.average(riskFactors) : 0
    const criticalAttributes = incomplete
      .filter((_, index) => {
        const factor = riskFactors[index]
        return (
          factor !== undefined && factor > Stats.percentile(riskFactors, 0.75)
        )
      })
      .map((c) => c.attribute)

    return {
      criticalAttributes,
      riskScore: Stats.round(riskScore, 100),
      timeRemaining: 1 - this.totalProgress,
      feasibilityScore: Stats.round(Math.max(0, 1 - riskScore / 10), 100),
    }
  }

  // 3. Add method to update risk assessment when needed
  updateRiskAssessment(peopleRemaining: number): void {
    this._riskAssessment = this.getRiskAssessment(peopleRemaining)
  }

  getCorrelationInsights(): {
    strongPairs: Array<{
      attr1: Keys
      attr2: Keys
      correlation: number
      bothNeeded: boolean
    }>
    conflictPairs: Array<{
      attr1: Keys
      attr2: Keys
      correlation: number
      bothNeeded: boolean
    }>
  } {
    const strongPairs: any[] = []
    const conflictPairs: any[] = []

    this._constraints.forEach((constraint1) => {
      this._constraints.forEach((constraint2) => {
        if (typeof constraint1.attribute !== 'string') return
        if (typeof constraint2.attribute !== 'string') return
        if (constraint1.attribute >= constraint2.attribute) return

        const correlation = this.getCorrelation(
          constraint1.attribute as Keys,
          constraint2.attribute as Keys
        )
        const bothNeeded =
          !this.isCompleted(constraint1.attribute as Keys) &&
          !this.isCompleted(constraint2.attribute as Keys)

        if (correlation > 0.4) {
          strongPairs.push({
            attr1: constraint1.attribute,
            attr2: constraint2.attribute,
            correlation: Stats.round(correlation, 100),
            bothNeeded,
          })
        } else if (correlation < -0.4) {
          conflictPairs.push({
            attr1: constraint1.attribute,
            attr2: constraint2.attribute,
            correlation: Stats.round(correlation, 100),
            bothNeeded,
          })
        }
      })
    })

    return {
      strongPairs: strongPairs.sort((a, b) => b.correlation - a.correlation),
      conflictPairs: conflictPairs.sort(
        (a, b) => a.correlation - b.correlation
      ),
    }
  }

  // Enhanced debug/info methods
  getSummary(): {
    totalConstraints: number
    completedConstraints: number
    totalProgress: number
    allMet: boolean
    mostNeeded: Keys[]
    leastProgress: Keys[]
    efficiency: number
    riskScore: number
    isBalanced: boolean
  } {
    const efficiency = this.getEfficiencyMetrics()
    const risk = this._riskAssessment
    const variability = this.getProgressVariability()

    return {
      totalConstraints: this.totalConstraints,
      completedConstraints: this.completedConstraints,
      totalProgress: Stats.round(this.totalProgress, 100),
      allMet: this.allConstraintsMet,
      mostNeeded: this.getMostNeededAttributes().slice(0, 3),
      leastProgress: this.getLeastProgressAttributes().slice(0, 3),
      efficiency: efficiency.actualEfficiency,
      riskScore: risk.riskScore,
      isBalanced: variability.isBalanced,
    }
  }

  getDetailedAnalysis() {
    return {
      progress: this.getProgressDistribution(),
      variability: this.getProgressVariability(),
      difficulty: this.getQuotaDifficulty(),
      efficiency: this.getEfficiencyMetrics(),
      risk: this._riskAssessment,
      correlations: this.getCorrelationInsights(),
    } as const
  }

  getPerformanceMetrics(): {
    constraintCount: number
    attributeCount: number
    correlationMatrixSize: number
    memoryEstimate: string
  } {
    const constraintCount = this._constraints.length
    const attributeCount = Object.keys(this._frequencies).length
    const correlationMatrixSize = attributeCount * attributeCount

    // Rough memory estimation
    const memoryEstimate = `~${Math.round(
      (correlationMatrixSize * 8 + constraintCount * 32) / 1024
    )}KB`

    return {
      constraintCount,
      attributeCount,
      correlationMatrixSize,
      memoryEstimate,
    }
  }
}
