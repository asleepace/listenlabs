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
  decayRate?: number // unused now (no per-step decay)
}

export class NeuralNetBouncer implements BerghainBouncer {
  private net: NeuralNet
  private encoder: StateEncoder
  private tracker: AttributeTracker
  private game: Game

  private baseThreshold: number
  private minThreshold: number
  private maxThreshold: number
  private urgencyFactor: number

  private explorationRate: number

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
    this.net = createBerghainNet(this.encoder.getFeatureSize())

    this.baseThreshold = config.baseThreshold ?? 0.5
    this.minThreshold = config.minThreshold ?? 0.3
    this.maxThreshold = config.maxThreshold ?? 0.7
    this.urgencyFactor = config.urgencyFactor ?? 2.0
    this.explorationRate = config.explorationRate ?? 0.1
  }

  admit(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    const features = this.encoder.encode(status)
    const output = this.net.forward(features)
    const probability = output[0]
    const threshold = this.calculateDynamicThreshold(status)

    // --- Urgency override: if critically behind on any attribute and this person helps, force admit
    const remaining = 1000 - status.admittedCount
    if (remaining > 0) {
      for (const c of this.game.constraints) {
        const have = this.tracker.getCount(c.attribute)
        const need = Math.max(0, c.minCount - have)
        const urgency = need / remaining
        if (urgency >= 0.9 && status.nextPerson.attributes[c.attribute]) {
          this.decisions.push({ features, probability, admitted: true, threshold })
          this.admissionCount++
          this.tracker.admit(status.nextPerson.attributes)
          return true
        }
      }
    }

    // Stall breaker: if we're completely stuck early, admit with high probability
    if (this.admissionCount === 0 && this.rejectionCount >= 500) {
      if (Math.random() < 0.9) {
        this.decisions.push({ features, probability, admitted: true, threshold })
        this.admissionCount++
        this.tracker.admit(status.nextPerson.attributes)
        return true
      }
    }

    // Exploration vs exploitation
    let decision: boolean
    if (Math.random() < this.explorationRate) {
      decision = this.makeExploratoryDecision(status)
    } else {
      decision = probability > threshold
    }

    this.decisions.push({ features, probability, admitted: decision, threshold })

    if (decision) {
      this.admissionCount++
      this.tracker.admit(status.nextPerson.attributes)
    } else {
      this.rejectionCount++
    }

    // NOTE: no per-step exploration decay here

    return decision
  }

  private calculateDynamicThreshold(status: GameStatusRunning<PersonAttributesScenario2>): number {
    const remaining = 1000 - status.admittedCount
    const counts = this.tracker.getCounts()

    let maxUrgency = 0
    for (const constraint of this.game.constraints) {
      const current = counts[constraint.attribute] || 0
      const needed = constraint.minCount - current
      if (needed > 0 && remaining > 0) {
        const urgency = needed / remaining
        maxUrgency = Math.max(maxUrgency, urgency)
      }
    }

    let threshold = this.baseThreshold
    if (maxUrgency > 0.9) {
      threshold = this.minThreshold
    } else if (maxUrgency > 0.7) {
      const factor = 1 - (maxUrgency - 0.7) * this.urgencyFactor
      threshold *= factor
    } else if (remaining < 100) {
      threshold = this.maxThreshold
    }

    // Favor a person who has any highly urgent attribute
    const person = status.nextPerson.attributes
    for (const c of this.game.constraints) {
      const current = counts[c.attribute] || 0
      const needed = c.minCount - current
      const urgency = remaining > 0 ? needed / remaining : 0
      if (urgency > 0.7 && person[c.attribute]) {
        threshold *= 0.7
        break
      }
    }

    return Math.max(this.minThreshold, Math.min(this.maxThreshold, threshold))
  }

  private makeExploratoryDecision(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    const remaining = 1000 - status.admittedCount
    const counts = this.tracker.getCounts()
    const person = status.nextPerson.attributes

    let value = 0
    let totalWeight = 0
    for (const c of this.game.constraints) {
      const current = counts[c.attribute] || 0
      const needed = Math.max(0, c.minCount - current)
      const weight = needed / Math.max(1, remaining)
      if (person[c.attribute]) value += weight
      totalWeight += weight
    }
    const normalizedValue = totalWeight > 0 ? value / totalWeight : 0.5

    const base = 0.65
    const admitProb = Math.min(0.98, base + 0.35 * normalizedValue)
    return Math.random() < admitProb
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
      averageProbability: this.decisions.reduce((s, d) => s + d.probability, 0) / Math.max(1, this.decisions.length),
      averageThreshold: this.decisions.reduce((s, d) => s + d.threshold, 0) / Math.max(1, this.decisions.length),
      networkParams: this.net.getParameterCount(),
      decisions: this.decisions.slice(-10),
    }
  }

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

  setNetwork(net: NeuralNet): void {
    this.net = net
  }
}

export function createTrainedBouncer(game: Game, weights?: any, config?: NeuralNetBouncerConfig): NeuralNetBouncer {
  const bouncer = new NeuralNetBouncer(game, config)
  if (weights) bouncer.loadWeights(weights)
  return bouncer
}
