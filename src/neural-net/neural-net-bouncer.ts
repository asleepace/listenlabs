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

  /** Dynamic threshold based on expected shortfall and remaining seats. */
  private calculateDynamicThreshold(status: GameStatusRunning<PersonAttributesScenario2>): number {
    // Avoid div-by-zero; use the tracker’s true counts
    const remaining = Math.max(1, 1000 - this.admissionCount)
    const counts = this.tracker.getCounts()

    // 1) Expected-feasibility worst gap across constraints
    let worstGap = 0
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const f = this.game.attributeStatistics.relativeFrequencies[c.attribute] || 0
      const expectedFinalIfDoNothing = cur + f * remaining
      const gap = Math.max(0, c.minCount - expectedFinalIfDoNothing)
      if (gap > worstGap) worstGap = gap
    }

    // Normalize gap to [0, 1+] by remaining seats
    const g = worstGap / remaining

    // 2) Map gap -> interpolation factor with hyperparameter k
    //    k ≈ 2.0 works well as a starting point.
    const k = 2.0
    const t = this.clampNum(k * g, 0, 1) // 0=no gap -> base, 1=big gap -> min

    // 3) Interpolate threshold toward minThreshold as gap grows
    let threshold = this.baseThreshold - (this.baseThreshold - this.minThreshold) * t

    // 4) Endgame: if we have very few seats left and no gap, be pickier
    if (remaining < 100 && g === 0) {
      const endgameT = (100 - remaining) / 100 // 0..1
      threshold = threshold + (this.maxThreshold - threshold) * endgameT * 0.5 // blend up to max a bit
    }

    // 5) Sweeten if person helps a currently urgent attribute (based on NEED/remaining)
    const person = status.nextPerson.attributes
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const need = Math.max(0, c.minCount - cur)
      const urgency = need / remaining // not expected-gap; just "need now"
      if (urgency > 0.7 && person[c.attribute]) {
        threshold *= 0.7 // nudge down
        break
      }
    }

    // 6) Final clamp to absolute bounds
    return this.clampNum(threshold, this.minThreshold, this.maxThreshold)
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
