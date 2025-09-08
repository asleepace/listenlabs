import type {
  GameState,
  GameStatus,
  PersonAttributes,
  PersonAttributesScenario3,
} from './types'
import { Stats } from './stats'

type Frequencies<Attributes extends PersonAttributes> = {
  readonly [K in keyof Attributes]: number
}

type Correlations<Attributes extends PersonAttributes> = {
  readonly [K in keyof Attributes]: {
    readonly [J in keyof Attributes]: number
  }
}

type Constraint<Attributes extends PersonAttributes> = {
  readonly attribute: keyof Attributes
  readonly minCount: number
  count: number
}

type ConstraintProgress<Attributes extends PersonAttributes> = {
  readonly attribute: keyof Attributes
  readonly current: number
  readonly needed: number
  readonly progress: number
  readonly completed: boolean
}

type AttributeStats<Attributes extends PersonAttributes> = {
  readonly attribute: keyof Attributes
  readonly frequency: number
  readonly rarity: number
  readonly quotaRate: number
  readonly overdemanded: boolean
}

export class Metrics<
  Attributes extends PersonAttributes = PersonAttributesScenario3
> {
  private _constraints: Constraint<Attributes>[]
  private _frequencies: Frequencies<Attributes>
  private _correlations: Correlations<Attributes>
  private _attributeStats: Map<keyof Attributes, AttributeStats<Attributes>>

  constructor(public readonly gameData: GameState['game']) {
    this._constraints = gameData.constraints.map(
      (constraint): Constraint<Attributes> => ({
        attribute: constraint.attribute as keyof Attributes,
        minCount: constraint.minCount,
        count: 0,
      })
    )

    this._frequencies = gameData.attributeStatistics
      .relativeFrequencies as Frequencies<Attributes>
    this._correlations = gameData.attributeStatistics
      .correlations as Correlations<Attributes>
    this._attributeStats = this.preCalculateAttributeStats()
  }

  // Getters
  get constraints(): readonly Constraint<Attributes>[] {
    return this._constraints
  }

  get frequencies(): Frequencies<Attributes> {
    return this._frequencies
  }

  get correlations(): Correlations<Attributes> {
    return this._correlations
  }

  get attributeStats(): ReadonlyMap<
    keyof Attributes,
    AttributeStats<Attributes>
  > {
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

  // Pre-calculate useful statistics
  private preCalculateAttributeStats(): Map<
    keyof Attributes,
    AttributeStats<Attributes>
  > {
    const stats = new Map<keyof Attributes, AttributeStats<Attributes>>()

    for (const constraint of this._constraints) {
      const attr = constraint.attribute
      const frequency = this._frequencies[attr] || 0
      const quotaRate = constraint.minCount / 1000 // Assuming 1000 capacity

      stats.set(attr, {
        attribute: attr,
        frequency,
        rarity: 1 / Math.max(frequency, 0.01), // Higher = rarer
        quotaRate,
        overdemanded: quotaRate > frequency * 1.5, // Needs more than natural rate
      })
    }

    return stats
  }

  // Counting methods
  updateCount(attribute: keyof Attributes, increment: number = 1): void {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (constraint) {
      constraint.count += increment
    }
  }

  updateCounts(attributes: Partial<Attributes>): void {
    Object.entries(attributes).forEach(([attr, hasAttribute]) => {
      if (hasAttribute) {
        this.updateCount(attr as keyof Attributes)
      }
    })
  }

  getCount(attribute: keyof Attributes): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    return constraint?.count ?? 0
  }

  getNeeded(attribute: keyof Attributes): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (!constraint) return 0
    return Math.max(0, constraint.minCount - constraint.count)
  }

  getProgress(attribute: keyof Attributes): number {
    const constraint = this._constraints.find((c) => c.attribute === attribute)
    if (!constraint) return 0
    return Math.min(constraint.count / constraint.minCount, 1)
  }

  isCompleted(attribute: keyof Attributes): boolean {
    return this.getNeeded(attribute) === 0
  }

  // Analysis methods
  getConstraintProgress(): ConstraintProgress<Attributes>[] {
    return this._constraints.map((constraint) => ({
      attribute: constraint.attribute,
      current: constraint.count,
      needed: Math.max(0, constraint.minCount - constraint.count),
      progress: Math.min(constraint.count / constraint.minCount, 1),
      completed: constraint.count >= constraint.minCount,
    }))
  }

  getIncompleteConstraints(): ConstraintProgress<Attributes>[] {
    return this.getConstraintProgress().filter((cp) => !cp.completed)
  }

  getMostNeededAttributes(): (keyof Attributes)[] {
    return this.getIncompleteConstraints()
      .sort((a, b) => b.needed - a.needed)
      .map((cp) => cp.attribute)
  }

  getLeastProgressAttributes(): (keyof Attributes)[] {
    return this.getIncompleteConstraints()
      .sort((a, b) => a.progress - b.progress)
      .map((cp) => cp.attribute)
  }

  getOverdemandedAttributes(): (keyof Attributes)[] {
    return Array.from(this._attributeStats.entries())
      .filter(
        ([_, stats]) => stats.overdemanded && !this.isCompleted(stats.attribute)
      )
      .map(([attr, _]) => attr)
  }

  getRarestAttributes(): (keyof Attributes)[] {
    return Array.from(this._attributeStats.entries())
      .filter(([attr, _]) => !this.isCompleted(attr))
      .sort(([_, a], [__, b]) => b.rarity - a.rarity)
      .map(([attr, _]) => attr)
  }

  // Correlation analysis
  getPositivelyCorrelated(
    attribute: keyof Attributes,
    threshold: number = 0.3
  ): (keyof Attributes)[] {
    const correlations = this._correlations[attribute]
    if (!correlations) return []

    type C = (typeof correlations)[keyof Attributes]

    return Object.entries(correlations)
      .filter(([_, correlation]) => (correlation as number) > threshold)
      .map(([attr, _]) => attr as keyof Attributes)
      .filter((attr) => attr !== attribute)
  }

  getNegativelyCorrelated(
    attribute: keyof Attributes,
    threshold: number = -0.3
  ): (keyof Attributes)[] {
    const correlations = this._correlations[attribute]
    if (!correlations) return []

    return Object.entries(correlations)
      .filter(([_, correlation]) => (correlation as number) < threshold)
      .map(([attr, _]) => attr as keyof Attributes)
      .filter((attr) => attr !== attribute)
  }

  getCorrelation(attr1: keyof Attributes, attr2: keyof Attributes): number {
    return this._correlations[attr1]?.[attr2] ?? 0
  }

  // Utility methods for person evaluation
  getUsefulAttributes(
    personAttributes: Partial<Attributes>
  ): (keyof Attributes)[] {
    return Object.entries(personAttributes)
      .filter(
        ([attr, hasAttr]) =>
          hasAttr && !this.isCompleted(attr as keyof Attributes)
      )
      .map(([attr, _]) => attr as keyof Attributes)
  }

  countUsefulAttributes(personAttributes: Partial<Attributes>): number {
    return this.getUsefulAttributes(personAttributes).length
  }

  hasAttribute(
    personAttributes: Partial<Attributes>,
    attribute: keyof Attributes
  ): boolean {
    return Boolean(personAttributes[attribute])
  }

  hasAllAttributes(personAttributes: Partial<Attributes>): boolean {
    return this._constraints.every((constraint) =>
      Boolean(personAttributes[constraint.attribute])
    )
  }

  hasAnyNeededAttribute(personAttributes: Partial<Attributes>): boolean {
    return this._constraints.some(
      (constraint) =>
        !this.isCompleted(constraint.attribute) &&
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
      this.getProgress(c.attribute)
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
    outliers: (keyof Attributes)[]
  } {
    const progressValues = this._constraints.map((c) =>
      this.getProgress(c.attribute)
    )
    const mean = Stats.average(progressValues)
    const stdDev = Stats.stdDev(progressValues)
    const coefficient = mean > 0 ? stdDev / mean : 0

    // Find outliers (progress > 2 std devs from mean)
    const outliers = this._constraints
      .filter((c) => {
        const progress = this.getProgress(c.attribute)
        const zScore = Stats.zScore(progress, progressValues)
        return Math.abs(zScore) > 2
      })
      .map((c) => c.attribute)

    return {
      isBalanced: coefficient < 0.3, // Low coefficient of variation
      coefficient,
      outliers,
    }
  }

  getQuotaDifficulty(): Map<
    keyof Attributes,
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
      const stats = this._attributeStats.get(attr)!
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
      this.getProgress(c.attribute)
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

  getRiskAssessment(peopleRemaining: number): {
    criticalAttributes: (keyof Attributes)[]
    riskScore: number
    timeRemaining: number
    feasibilityScore: number
  } {
    const incomplete = this.getIncompleteConstraints()

    // Sort by difficulty/rarity to process hardest first
    const sortedIncomplete = incomplete.sort((a, b) => {
      const freqA = this._attributeStats.get(a.attribute)!.frequency
      const freqB = this._attributeStats.get(b.attribute)!.frequency
      return freqA - freqB // Rarest first
    })

    let availablePeople = peopleRemaining

    const riskFactors = sortedIncomplete.map((constraint) => {
      const stats = this._attributeStats.get(constraint.attribute)!
      const needed = constraint.needed
      const frequency = stats.frequency

      const expectedWithAttribute = availablePeople * frequency
      const riskRatio = needed / Math.max(expectedWithAttribute, 1)
      const scaledRisk = Stats.clamp(riskRatio * 3, 0, 10) // Define before logging

      console.log(
        `${String(
          constraint.attribute
        )}: need=${needed}, freq=${frequency}, expected=${expectedWithAttribute}, ratio=${riskRatio}, scaled=${scaledRisk}`
      )

      // More conservative subtraction for rarer attributes
      const peopleUsed = Math.min(needed, expectedWithAttribute * 0.8)
      availablePeople = Math.max(0, availablePeople - peopleUsed)

      return Stats.clamp(riskRatio * 3, 0, 10)
    })

    const riskScore = riskFactors.length > 0 ? Stats.average(riskFactors) : 0
    const criticalAttributes = incomplete
      .filter(
        (_, index) => riskFactors[index]! > Stats.percentile(riskFactors, 0.75)
      )
      .map((c) => c.attribute)

    return {
      criticalAttributes,
      riskScore: Stats.round(riskScore, 100),
      timeRemaining: 1 - this.totalProgress,
      feasibilityScore: Stats.round(Math.max(0, 1 - riskScore / 10), 100),
    }
  }

  getCorrelationInsights(): {
    strongPairs: Array<{
      attr1: keyof Attributes
      attr2: keyof Attributes
      correlation: number
      bothNeeded: boolean
    }>
    conflictPairs: Array<{
      attr1: keyof Attributes
      attr2: keyof Attributes
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
          constraint1.attribute,
          constraint2.attribute
        )
        const bothNeeded =
          !this.isCompleted(constraint1.attribute) &&
          !this.isCompleted(constraint2.attribute)

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
  getSummary(peopleInLineLeft: number): {
    totalConstraints: number
    completedConstraints: number
    totalProgress: number
    allMet: boolean
    mostNeeded: (keyof Attributes)[]
    leastProgress: (keyof Attributes)[]
    efficiency: number
    riskScore: number
    isBalanced: boolean
  } {
    const efficiency = this.getEfficiencyMetrics()
    const risk = this.getRiskAssessment(peopleInLineLeft)
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

  getDetailedAnalysis(peopleInLineLeft: number) {
    return {
      progress: this.getProgressDistribution(),
      variability: this.getProgressVariability(),
      difficulty: this.getQuotaDifficulty(),
      efficiency: this.getEfficiencyMetrics(),
      risk: this.getRiskAssessment(peopleInLineLeft),
      correlations: this.getCorrelationInsights(),
    } as const
  }
}
