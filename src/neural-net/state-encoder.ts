/** @file state-encoder.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, GameConstraints } from '../types' // Adjust import path as needed

export class StateEncoder {
  private attributeKeys: string[]
  private maxAdmitted = 1000
  private maxRejected = 20000

  constructor(private game: Game) {
    // Extract attribute keys from constraints or statistics
    this.attributeKeys = Object.keys(game.attributeStatistics.relativeFrequencies)
  }

  encode(status: GameStatusRunning<PersonAttributesScenario2>): number[] {
    const features: number[] = []

    // 1. Person attributes (4 binary values)
    const personFeatures = this.encodePersonAttributes(status.nextPerson.attributes)
    features.push(...personFeatures)

    // 2. Constraint satisfaction ratios (4 values, 0-1)
    const satisfactionRatios = this.getConstraintSatisfactionRatios(status)
    features.push(...satisfactionRatios)

    // 3. Constraint pressure scores (4 values, 0-1)
    // How urgently we need each attribute
    const pressureScores = this.getConstraintPressure(status)
    features.push(...pressureScores)

    // 4. Global state features
    features.push(
      // Progress toward capacity
      status.admittedCount / this.maxAdmitted,

      // Rejection rate
      status.rejectedCount / this.maxRejected,

      // Remaining capacity ratio
      (this.maxAdmitted - status.admittedCount) / this.maxAdmitted
    )

    // 5. Person-constraint alignment score
    // How well this person helps meet unsatisfied constraints
    const alignmentScore = this.getAlignmentScore(status)
    features.push(alignmentScore)

    // 6. Correlation features
    // How this person's attributes correlate with high-need attributes
    const correlationScore = this.getCorrelationScore(status)
    features.push(correlationScore)

    return features
  }

  private encodePersonAttributes(attributes: PersonAttributesScenario2): number[] {
    return this.attributeKeys.map((key) => (attributes[key] ? 1 : 0))
  }

  private getConstraintSatisfactionRatios(status: GameStatusRunning<PersonAttributesScenario2>): number[] {
    // Track current counts for each attribute
    const currentCounts = this.getCurrentAttributeCounts(status)

    return this.game.constraints.map((constraint) => {
      const current = currentCounts[constraint.attribute] || 0
      const required = constraint.minCount
      // Clamp to [0, 1]
      return Math.min(1, current / required)
    })
  }

  private getConstraintPressure(status: GameStatusRunning<PersonAttributesScenario2>): number[] {
    const remaining = this.maxAdmitted - status.admittedCount
    const currentCounts = this.getCurrentAttributeCounts(status)

    return this.game.constraints.map((constraint) => {
      const current = currentCounts[constraint.attribute] || 0
      const stillNeeded = Math.max(0, constraint.minCount - current)

      if (remaining === 0) return 0

      // Pressure = how many we still need / how many slots remain
      // Higher pressure means we need a higher percentage of remaining slots
      const pressure = stillNeeded / remaining

      // Apply exponential scaling for high pressure situations
      // This makes the network more sensitive to urgent constraints
      return Math.min(1, Math.pow(pressure, 0.7))
    })
  }

  private getAlignmentScore(status: GameStatusRunning<PersonAttributesScenario2>): number {
    const pressures = this.getConstraintPressure(status)
    const personAttrs = this.encodePersonAttributes(status.nextPerson.attributes)

    let score = 0
    let totalPressure = 0

    this.game.constraints.forEach((constraint, i) => {
      const attrIndex = this.attributeKeys.indexOf(constraint.attribute)
      if (attrIndex >= 0 && personAttrs[attrIndex] === 1) {
        score += pressures[i]
      }
      totalPressure += pressures[i]
    })

    return totalPressure > 0 ? score / totalPressure : 0
  }

  private getCorrelationScore(status: GameStatusRunning<PersonAttributesScenario2>): number {
    const pressures = this.getConstraintPressure(status)
    const person = status.nextPerson.attributes
    let score = 0

    // For each high-pressure constraint, check correlations
    this.game.constraints.forEach((constraint, i) => {
      if (pressures[i] > 0.5) {
        // High pressure threshold
        const targetAttr = constraint.attribute

        // Check how this person's attributes correlate with the needed attribute
        this.attributeKeys.forEach((attr) => {
          if (person[attr]) {
            const correlation = this.game.attributeStatistics.correlations[attr]?.[targetAttr] || 0
            score += correlation * pressures[i]
          }
        })
      }
    })

    // Normalize to [-1, 1]
    const maxPossible = pressures.filter((p) => p > 0.5).length
    return maxPossible > 0 ? score / maxPossible : 0
  }

  private getCurrentAttributeCounts(status: GameStatusRunning<PersonAttributesScenario2>): Record<string, number> {
    // In a real implementation, we'd track this incrementally
    // For now, estimate based on admitted count and expected frequencies
    const counts: Record<string, number> = {}

    this.attributeKeys.forEach((attr) => {
      const frequency = this.game.attributeStatistics.relativeFrequencies[attr]
      // This is an approximation - in practice you'd track actual counts
      counts[attr] = Math.floor(status.admittedCount * frequency)
    })

    return counts
  }

  // Get feature vector size
  getFeatureSize(): number {
    // 4 person attrs + 4 satisfaction ratios + 4 pressure scores +
    // 3 global features + 1 alignment + 1 correlation = 17
    return this.attributeKeys.length + 2 * this.game.constraints.length + 5
  }

  // Get feature names for debugging
  getFeatureNames(): string[] {
    const names: string[] = []

    // Person attributes
    this.attributeKeys.forEach((key) => names.push(`person_${key}`))

    // Constraint satisfaction
    this.game.constraints.forEach((c) => names.push(`satisfaction_${c.attribute}`))

    // Constraint pressure
    this.game.constraints.forEach((c) => names.push(`pressure_${c.attribute}`))

    // Global features
    names.push('progress_ratio', 'rejection_ratio', 'remaining_capacity', 'alignment_score', 'correlation_score')

    return names
  }

  // Normalize features to [-1, 1] or [0, 1] if needed
  normalize(features: number[]): number[] {
    // Most features are already in [0, 1]
    // Correlation score is in [-1, 1]
    // This method can be extended if different normalization is needed
    return features
  }
}

// Helper class to track actual counts during game
export class AttributeTracker {
  private counts: Map<string, number> = new Map()
  private admittedAttributes: Array<Set<string>> = []

  constructor(private attributeKeys: string[]) {
    attributeKeys.forEach((key) => this.counts.set(key, 0))
  }

  admit(attributes: Record<string, boolean>): void {
    const attrSet = new Set<string>()

    for (const [key, value] of Object.entries(attributes)) {
      if (value) {
        attrSet.add(key)
        this.counts.set(key, (this.counts.get(key) || 0) + 1)
      }
    }

    this.admittedAttributes.push(attrSet)
  }

  getCounts(): Record<string, number> {
    const result: Record<string, number> = {}
    this.counts.forEach((value, key) => {
      result[key] = value
    })
    return result
  }

  getCount(attribute: string): number {
    return this.counts.get(attribute) || 0
  }

  reset(): void {
    this.counts.clear()
    this.admittedAttributes = []
    this.attributeKeys.forEach((key) => this.counts.set(key, 0))
  }
}
