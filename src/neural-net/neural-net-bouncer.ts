/** @file neural-net-bouncer.ts */

import type { BerghainBouncer } from '../berghain'

import type {
  Game,
  GameStatusRunning,
  GameStatusCompleted,
  GameStatusFailed,
  PersonAttributesScenario2,
} from '../types'

import { NeuralNet, createBerghainNet } from './neural-net'
import { StateEncoder, AttributeTracker } from './state-encoder'

export interface NeuralNetBouncerConfig {
  baseThreshold?: number
  minThreshold?: number
  maxThreshold?: number
  urgencyFactor?: number
  explorationRate?: number
  decayRate?: number
}

export class NeuralNetBouncer implements BerghainBouncer {
  private net: NeuralNet
  private encoder: StateEncoder
  private tracker: AttributeTracker
  private game: Game

  // Decision thresholds
  private baseThreshold: number
  private minThreshold: number
  private maxThreshold: number
  private urgencyFactor: number

  // Exploration for training
  private explorationRate: number
  private decayRate: number

  // Tracking for analysis
  private decisions: Array<{
    features: number[]
    probability: number
    admitted: boolean
    threshold: number
  }> = []

  private admissionCount = 0
  private rejectionCount = 0

  constructor(game: Game, config: NeuralNetBouncerConfig = {}) {
    this.game = game
    this.encoder = new StateEncoder(game)
    this.tracker = new AttributeTracker(Object.keys(game.attributeStatistics.relativeFrequencies))

    // Initialize or load neural network
    this.net = createBerghainNet(this.encoder.getFeatureSize())

    // Configure thresholds
    this.baseThreshold = config.baseThreshold ?? 0.5
    this.minThreshold = config.minThreshold ?? 0.3
    this.maxThreshold = config.maxThreshold ?? 0.7
    this.urgencyFactor = config.urgencyFactor ?? 2.0
    this.explorationRate = config.explorationRate ?? 0.1
    this.decayRate = config.decayRate ?? 0.995
  }

  admit(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    // Encode current state
    const features = this.encoder.encode(status)

    // Get neural network prediction
    const output = this.net.forward(features)
    const probability = output[0]

    // Calculate dynamic threshold
    const threshold = this.calculateDynamicThreshold(status)

    // Exploration vs exploitation
    let decision: boolean
    if (Math.random() < this.explorationRate) {
      // Exploration: make random decision weighted by constraints
      decision = this.makeExploratoryDecision(status)
    } else {
      // Exploitation: use neural network
      decision = probability > threshold
    }

    // Track the decision
    this.decisions.push({
      features,
      probability,
      admitted: decision,
      threshold,
    })

    // Update tracking
    if (decision) {
      this.admissionCount++
      this.tracker.admit(status.nextPerson.attributes)
    } else {
      this.rejectionCount++
    }

    // Decay exploration rate
    this.explorationRate *= this.decayRate

    return decision
  }

  private calculateDynamicThreshold(status: GameStatusRunning<PersonAttributesScenario2>): number {
    const remaining = 1000 - status.admittedCount
    const counts = this.tracker.getCounts()

    // Calculate urgency for each constraint
    let maxUrgency = 0
    let avgUrgency = 0
    let urgentConstraints = 0

    for (const constraint of this.game.constraints) {
      const current = counts[constraint.attribute] || 0
      const needed = constraint.minCount - current

      if (needed > 0 && remaining > 0) {
        const urgency = needed / remaining
        maxUrgency = Math.max(maxUrgency, urgency)
        avgUrgency += urgency

        if (urgency > 0.8) {
          urgentConstraints++
        }
      }
    }

    avgUrgency /= this.game.constraints.length

    // Base threshold adjustment
    let threshold = this.baseThreshold

    // Adjust based on urgency
    if (maxUrgency > 0.9) {
      // Critical: we need almost everyone to have certain attributes
      threshold = this.minThreshold
    } else if (maxUrgency > 0.7) {
      // High urgency: lower threshold proportionally
      const factor = 1 - (maxUrgency - 0.7) * this.urgencyFactor
      threshold *= factor
    } else if (remaining < 100) {
      // Near the end: be more selective if we're meeting constraints
      threshold = this.maxThreshold
    }

    // Check if person has highly needed attributes
    const person = status.nextPerson.attributes
    let hasUrgentAttribute = false

    for (const constraint of this.game.constraints) {
      const current = counts[constraint.attribute] || 0
      const needed = constraint.minCount - current
      const urgency = remaining > 0 ? needed / remaining : 0

      if (urgency > 0.7 && person[constraint.attribute]) {
        hasUrgentAttribute = true
        break
      }
    }

    // Lower threshold if person has urgent attributes
    if (hasUrgentAttribute) {
      threshold *= 0.7
    }

    // Clamp to valid range
    return Math.max(this.minThreshold, Math.min(this.maxThreshold, threshold))
  }

  private makeExploratoryDecision(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    const remaining = 1000 - status.admittedCount
    const counts = this.tracker.getCounts()
    const person = status.nextPerson.attributes

    // Calculate value score for this person
    let value = 0
    let totalWeight = 0

    for (const constraint of this.game.constraints) {
      const current = counts[constraint.attribute] || 0
      const needed = Math.max(0, constraint.minCount - current)
      const weight = needed / Math.max(1, remaining)

      if (person[constraint.attribute]) {
        value += weight
      }
      totalWeight += weight
    }

    // Normalize value
    const normalizedValue = totalWeight > 0 ? value / totalWeight : 0.5

    // Make probabilistic decision based on value
    return Math.random() < 0.3 + 0.4 * normalizedValue
  }

  getProgress(): any {
    const counts = this.tracker.getCounts()
    const constraints = this.game.constraints.map((c) => ({
      attribute: c.attribute,
      required: c.minCount,
      current: counts[c.attribute] || 0,
      satisfied: (counts[c.attribute] || 0) >= c.minCount,
    }))

    return {
      admitted: this.admissionCount,
      rejected: this.rejectionCount,
      constraints,
      explorationRate: this.explorationRate,
      decisions: this.decisions.length,
      allSatisfied: constraints.every((c) => c.satisfied),
    }
  }

  getOutput(lastStatus: GameStatusCompleted | GameStatusFailed): any {
    const progress = this.getProgress()

    return {
      status: lastStatus.status,
      finalRejections: lastStatus.status === 'completed' ? lastStatus.rejectedCount : -1,
      ...progress,
      averageProbability: this.decisions.reduce((sum, d) => sum + d.probability, 0) / this.decisions.length,
      averageThreshold: this.decisions.reduce((sum, d) => sum + d.threshold, 0) / this.decisions.length,
      networkParams: this.net.getParameterCount(),
      decisions: this.decisions.slice(-10), // Last 10 decisions for debugging
    }
  }

  // Training methods

  getDecisions(): typeof this.decisions {
    return this.decisions
  }

  loadWeights(weights: any): void {
    this.net = NeuralNet.fromJSON(weights)
  }

  getWeights(): any {
    return this.net.toJSON()
  }

  reset(): void {
    this.tracker.reset()
    this.decisions = []
    this.admissionCount = 0
    this.rejectionCount = 0
  }

  // Set network directly (for training)
  setNetwork(net: NeuralNet): void {
    this.net = net
  }
}

// Factory function with pre-trained weights (if available)
export function createTrainedBouncer(game: Game, weights?: any, config?: NeuralNetBouncerConfig): NeuralNetBouncer {
  const bouncer = new NeuralNetBouncer(game, config)

  if (weights) {
    bouncer.loadWeights(weights)
  }

  return bouncer
}
