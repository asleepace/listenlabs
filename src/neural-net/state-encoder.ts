import type { Game, GameStatusRunning, ScenarioAttributes } from '../types'
import { Conf } from './config'

export class StateEncoder {
  private attributeKeys: string[]
  private maxAdmitted = Conf.MAX_ADMISSIONS
  private maxRejected = Conf.MAX_REJECTIONS

  constructor(private game: Game) {
    this.attributeKeys = Object.keys(game.attributeStatistics.relativeFrequencies)
  }

  /**
   * Encode the state.
   * @param status game status
   * @param countsOverride (optional) real counts of admitted attributes to use instead of estimates
   */
  encode(status: GameStatusRunning<ScenarioAttributes>, countsOverride: Record<string, number>): number[] {
    // Precompute once
    const personFeatures = this.encodePersonAttributes(status.nextPerson.attributes)
    const satisfactionRatios = this.getConstraintSatisfactionRatios(status, countsOverride)
    const pressureScores = this.getConstraintPressure(status, countsOverride)
    const alignmentScore = this.getAlignmentScore(status, countsOverride)
    const correlationScore = this.getCorrelationScore(status, countsOverride)

    const progressRatio = status.admittedCount / this.maxAdmitted
    const rejectionRatio = status.rejectedCount / this.maxRejected
    const remainingCapacity = (this.maxAdmitted - status.admittedCount) / this.maxAdmitted

    // Build the feature vector in the same order used by getFeatureNames()
    const features: number[] = [
      ...personFeatures, // |attributeKeys|
      ...satisfactionRatios, // |constraints|
      ...pressureScores, // |constraints|
      progressRatio, // 1
      rejectionRatio, // 1
      remainingCapacity, // 1
      alignmentScore, // 1
      correlationScore, // 1
    ]

    if (features.length !== this.getFeatureSize()) {
      console.warn('[StateEncoder] feature length mismatch:', { got: features.length, expected: this.getFeatureSize() })
    }

    return features
  }

  private encodePersonAttributes(attributes: ScenarioAttributes): number[] {
    return this.attributeKeys.map((key) => (attributes[key] ? 1 : 0))
  }

  private getConstraintSatisfactionRatios(
    status: GameStatusRunning<ScenarioAttributes>,
    countsOverride?: Record<string, number>
  ): number[] {
    const currentCounts = countsOverride ?? this.getEstimatedCounts(status)
    return this.game.constraints.map((constraint) => {
      const current = currentCounts[constraint.attribute] || 0
      const required = constraint.minCount
      return Math.min(1, current / required)
    })
  }

  private getConstraintPressure(
    status: GameStatusRunning<ScenarioAttributes>,
    countsOverride?: Record<string, number>
  ): number[] {
    const remaining = this.maxAdmitted - status.admittedCount
    const currentCounts = countsOverride ?? this.getEstimatedCounts(status)
    return this.game.constraints.map((constraint) => {
      const current = currentCounts[constraint.attribute] || 0
      const stillNeeded = Math.max(0, constraint.minCount - current)
      if (remaining === 0) return 0
      const pressure = stillNeeded / remaining
      return Math.min(1, Math.pow(pressure, 0.7))
    })
  }

  private getAlignmentScore(
    status: GameStatusRunning<ScenarioAttributes>,
    countsOverride?: Record<string, number>
  ): number {
    const pressures = this.getConstraintPressure(status, countsOverride)
    const personAttrs = this.encodePersonAttributes(status.nextPerson.attributes)
    let score = 0
    let total = 0
    this.game.constraints.forEach((constraint, i) => {
      const idx = this.attributeKeys.indexOf(constraint.attribute)
      if (idx >= 0 && personAttrs[idx] === 1) score += pressures[i]
      total += pressures[i]
    })
    return total > 0 ? score / total : 0
  }

  private getCorrelationScore(
    status: GameStatusRunning<ScenarioAttributes>,
    countsOverride?: Record<string, number> // kept for symmetry
  ): number {
    const pressures = this.getConstraintPressure(status, countsOverride)
    const person = status.nextPerson.attributes
    let score = 0
    this.game.constraints.forEach((constraint, i) => {
      if (pressures[i] > 0.5) {
        const targetAttr = constraint.attribute
        this.attributeKeys.forEach((attr) => {
          if (person[attr]) {
            const corr = this.game.attributeStatistics.correlations[attr]?.[targetAttr] || 0
            score += corr * pressures[i]
          }
        })
      }
    })
    const maxP = pressures.filter((p) => p > 0.5).length
    return maxP > 0 ? score / maxP : 0
  }

  /** Old heuristic: keep for fallback when no override is provided */
  private getEstimatedCounts(status: GameStatusRunning<ScenarioAttributes>): Record<string, number> {
    const counts: Record<string, number> = {}
    this.attributeKeys.forEach((attr) => {
      const freq = this.game.attributeStatistics.relativeFrequencies[attr]
      counts[attr] = Math.floor(status.admittedCount * freq)
    })
    return counts
  }

  getFeatureSize(): number {
    return this.attributeKeys.length + 2 * this.game.constraints.length + 5
  }

  getFeatureNames(): string[] {
    const names: string[] = []
    this.attributeKeys.forEach((k) => names.push(`person_${k}`))
    this.game.constraints.forEach((c) => names.push(`satisfaction_${c.attribute}`))
    this.game.constraints.forEach((c) => names.push(`pressure_${c.attribute}`))
    names.push('progress_ratio', 'rejection_ratio', 'remaining_capacity', 'alignment_score', 'correlation_score')
    return names
  }

  normalize(features: number[]): number[] {
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
