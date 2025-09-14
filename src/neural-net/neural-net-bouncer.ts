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
}

function clamp(min: number, curr: number, max: number): number {
  return Math.max(min, Math.min(curr, max))
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

  /** Legacy one-shot decision with side-effects (kept for runtime use). */
  admit(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    const features = this.encoder.encode(status)
    const probability = this.net.forward(features)[0]
    const threshold = this.computeThreshold(status)

    // console.log('explore:', this.explorationRate)

    let decision: boolean
    if (Math.random() < this.explorationRate) {
      decision = this.exploratoryAdmit(status)
    } else {
      decision = probability > threshold
    }

    this.applyFinalDecision(decision, status.nextPerson.attributes)
    this.decisions.push({ features, probability, admitted: decision, threshold })
    return decision
  }

  /** Side-effect free forward pass. */
  predictProbability(features: number[]): number {
    return this.net.forward(features)[0]
  }

  /** Public wrapper for dynamic threshold (side-effect free). */
  computeThreshold(status: GameStatusRunning<PersonAttributesScenario2>): number {
    return this.calculateDynamicThreshold(status)
  }

  /** Side-effect free exploration heuristic used by training. */
  exploratoryAdmit(status: GameStatusRunning<PersonAttributesScenario2>): boolean {
    const remaining = Math.max(1, 1000 - this.admissionCount)
    const counts = this.tracker.getCounts()
    const person = status.nextPerson.attributes

    let value = 0
    let totalWeight = 0
    for (const c of this.game.constraints) {
      const current = counts[c.attribute] || 0
      const needed = Math.max(0, c.minCount - current)
      const weight = needed / remaining
      if (person[c.attribute]) value += weight
      totalWeight += weight
    }
    const normalizedValue = totalWeight > 0 ? value / totalWeight : 0.5
    const base = 0.65
    const admitProb = Math.min(0.98, base + 0.35 * normalizedValue)
    return Math.random() < admitProb
  }

  /** Apply the final chosen decision exactly once to internal counters + tracker. */
  applyFinalDecision(admit: boolean, personAttrs: Record<string, boolean>): void {
    if (admit) {
      this.admissionCount++
      this.tracker.admit(personAttrs)
    } else {
      this.rejectionCount++
      // no change to tracker on rejection
    }
  }

  // Put this helper somewhere in the file
  clampNum = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

  /** Dynamic threshold: stricter when urgency is high; lenient only if the person helps. */
  private calculateDynamicThreshold(status: GameStatusRunning<PersonAttributesScenario2>): number {
    const remaining = Math.max(0, 1000 - this.admissionCount)
    const counts = this.tracker.getCounts()

    // compute maximum urgency over constraints
    let maxUrgency = 0
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const need = Math.max(0, c.minCount - cur)
      if (remaining > 0) {
        const u = need / remaining
        if (u > maxUrgency) maxUrgency = u
      }
    }

    // Base: go stricter as urgency rises
    let threshold = this.baseThreshold
    if (maxUrgency >= 0.8) {
      threshold = this.maxThreshold // very strict
    } else if (maxUrgency >= 0.6) {
      threshold = this.baseThreshold + 0.7 * (this.maxThreshold - this.baseThreshold)
    } else if (maxUrgency >= 0.4) {
      threshold = this.baseThreshold + 0.4 * (this.maxThreshold - this.baseThreshold)
    } else if (remaining < 100) {
      threshold = this.baseThreshold + 0.3 * (this.maxThreshold - this.baseThreshold)
    }

    // If the person has a highly urgent attribute, sweeten the threshold
    const person = status.nextPerson.attributes
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const need = Math.max(0, c.minCount - cur)
      const u = remaining > 0 ? need / remaining : 0
      if (u >= 0.6 && person[c.attribute]) {
        threshold = Math.max(this.minThreshold, threshold * 0.6) // easier if they help
        break
      }
    }

    return Math.max(this.minThreshold, Math.min(this.maxThreshold, threshold))
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
