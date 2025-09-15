/** @file neural-net-bouncer.ts */

import type {
  BerghainBouncer,
  Game,
  GameStatusCompleted,
  GameStatusFailed,
  GameStatusRunning,
  ScenarioAttributes,
} from '../types'
import { Conf } from './config'
import { NeuralNet } from './neural-net'
import { StateEncoder } from './state-encoder'
import { sum } from './util'

export class NeuralNetBouncer implements BerghainBouncer {
  private encoder: StateEncoder
  private net?: NeuralNet
  private counts: Record<string, number> = {}

  constructor(
    private game: Game,
    private cfg: {
      explorationRate?: number
      baseThreshold?: number
      minThreshold?: number
      maxThreshold?: number
      urgencyFactor?: number
      optimism?: number // 0..1 higher = more optimistic (default 0.7)
    } = {}
  ) {
    this.encoder = new StateEncoder(game)
  }

  public setNetwork(net: NeuralNet) {
    this.net = net
  }
  public setCounts(counts: Record<string, number>) {
    this.counts = { ...counts }
  }

  private sigmoid(z: number): number {
    if (z >= 0) {
      const ez = Math.exp(-z)
      return 1 / (1 + ez)
    }
    const ez = Math.exp(z)
    return ez / (1 + ez)
  }

  /** Convert various NN outputs to a single admit probability in [0,1] */

  private toProbability(raw: unknown): number {
    if (typeof raw === 'number') return raw < 0 || raw > 1 ? this.sigmoid(raw) : raw
    if (Array.isArray(raw)) {
      const arr = raw as number[]
      if (arr.length === 0) return 0.5
      if (arr.length === 1) {
        const v = arr[0]
        return v < 0 || v > 1 ? this.sigmoid(v) : v
      }
      const [a, b] = arr
      const s = a + b
      if (s > 0.999 && s < 1.001 && a >= 0 && b >= 0) return b
      const m = Math.max(a, b)
      const ea = Math.exp(a - m),
        eb = Math.exp(b - m)
      return eb / (ea + eb)
    }
    if (raw && typeof raw === 'object') {
      const o = raw as Record<string, any>
      if (typeof o.prob === 'number') return clamp01(o.prob)
      if (typeof o.p === 'number') return clamp01(o.p)
      if (typeof o.logit === 'number') return this.sigmoid(o.logit)
      if (Array.isArray(o.probs) && o.probs.length >= 2) return clamp01(o.probs[1])
      if (Array.isArray(o.logits) && o.logits.length >= 2) {
        const a = o.logits[0],
          b = o.logits[1]
        const m = Math.max(a, b)
        const ea = Math.exp(a - m),
          eb = Math.exp(b - m)
        return eb / (ea + eb)
      }
    }
    return 0.5
    function clamp01(x: number) {
      return Math.max(0, Math.min(1, x))
    }
  }

  /** Progress-aware threshold with safe bounds */
  private dynamicThreshold(status: GameStatusRunning<any>): number {
    const base = this.cfg.baseThreshold ?? 0.28
    const minT = this.cfg.minThreshold ?? 0.18
    const maxT = this.cfg.maxThreshold ?? 0.6
    const optimism = this.cfg.optimism ?? 0.7 // higher = more lenient
    const progress = Math.min(1, status.admittedCount / Math.max(1, Conf.MAX_ADMISSIONS))
    const slope = 0.1 * (1 - 0.5 * (optimism - 0.7)) // small effect of optimism on slope
    const theta = base + slope * progress
    return Math.max(minT, Math.min(maxT, theta))
  }

  /** Remaining need per constraint (only unmet). */
  private unmetNeeds(counts: Record<string, number>): Record<string, number> {
    const out: Record<string, number> = {}
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const need = Math.max(0, c.minCount - cur)
      if (need > 0) out[c.attribute] = need
    }
    return out
  }

  /**
   * Returns a tuple containing [required, critical] attributes, which should be
   * used as hard limits on letting people in the venue.
   *
   *  - required: next admitted person must have all of these attributes
   *  - critical: next admitted person must have some of these attributes
   *
   * NOTE: Often times two negatively correlated attributes will become the bottleneck
   * at the end and so each turn this will pick an odd or even strategy for filtering
   * critical attributes, this way they don't become stuck.
   */
  private getSafetyGates(
    status: GameStatusRunning<ScenarioAttributes>,
    needed: Record<string, number>
  ): [required: string[], critical: string[]] {
    const attrs = Object.keys(needed)
    if (attrs.length === 0) return [[], []]

    // Add a small buffer which oscillates to prevent any one attributes from becoming stuck
    // as required.
    const wiggleRoom = attrs.length * (status.admittedCount % 2 === 1 ? 2 : 1)
    const seatsLeft = Math.max(0, Conf.MAX_ADMISSIONS - status.admittedCount - wiggleRoom)
    if (seatsLeft <= 0) return [[], []]

    if (attrs.length === 1) {
      // Only one unmet quota → every remaining seat must help it
      return [[attrs[0]], [attrs[0]]]
    }

    const totalNeed = sum(Object.values(needed))
    const topAttr = attrs.reduce((best, a) => (needed[a] > needed[best] ? a : best), attrs[0])
    const maxNeed = needed[topAttr]

    // If the top-need alone equals/exceeds remaining seats, we must target it.
    if (maxNeed >= seatsLeft) {
      return [[topAttr], attrs] // require top; critical = any unmet
    }

    // If total unmet equals/exceeds remaining seats, every seat must hit some unmet attr.
    if (totalNeed >= seatsLeft) {
      return [[], attrs] // any unmet attribute is acceptable, but at least one is required
    }

    return [[], []]
  }

  /**
   * IMPORTANT: countsOverride lets the trainer pass the same counts it used to
   * encode the state, so train/test features match.
   */
  public admit(status: GameStatusRunning<ScenarioAttributes>, countsOverride?: Record<string, number>): boolean {
    const counts = countsOverride ?? this.counts

    if (!NeuralNet.isNeuralNet(this.net)) {
      throw new Error('NeuralNetBouncer: Neural net is not defined!')
    }
    if (!counts) {
      throw new Error('NeuralNetBouncer: Missing counts!')
    }

    const guest = status.nextPerson.attributes

    // Encode state & get model probability
    const x = this.encoder.encode(status, counts)
    const raw = this.net.forward?.(x) ?? this.net.infer(x)
    const p = this.toProbability(raw)
    const theta = this.dynamicThreshold(status)

    // Quota-aware safety gates
    const needed = this.unmetNeeds(counts)

    // If all quotas are satisfied, be optimistic — admit to finish quickly.
    if (Object.keys(needed).length === 0) return true

    const [required, critical] = this.getSafetyGates(status, needed)

    // Softer REQUIRED: guest must match *any one* of the required attrs (not all).
    if (required.length) {
      const hasAnyRequired = required.some((a) => guest[a])
      if (!hasAnyRequired) return false
    }

    // CRITICAL: guest must match at least one of the critical attrs if present.
    if (critical.length) {
      const hasAnyCritical = critical.some((a) => guest[a])
      if (!hasAnyCritical) return false
    }

    // Optional exploration (usually 0 in prod)
    const eps = this.cfg.explorationRate ?? 0
    if (eps > 0 && Math.random() < eps) return Math.random() < 0.5

    // Model decision
    return p >= theta
  }

  // required by interface...

  public getOutput(_lastStatus: GameStatusCompleted | GameStatusFailed) {
    return {}
  }

  getProgress() {
    return {}
  }
}
