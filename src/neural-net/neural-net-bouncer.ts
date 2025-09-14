/** @file neural-net-bouncer.ts */

import type { BerghainBouncer, Game, GameStatusCompleted, GameStatusFailed, GameStatusRunning } from '../types'
import { Conf } from './config'
import { NeuralNet } from './neural-net'
import { StateEncoder } from './state-encoder'

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
    // stable-ish sigmoid
    if (z >= 0) {
      const ez = Math.exp(-z)
      return 1 / (1 + ez)
    } else {
      const ez = Math.exp(z)
      return ez / (1 + ez)
    }
  }

  /** Convert various NN outputs to a single admit probability in [0,1] */
  private toProbability(raw: unknown): number {
    // 1) scalar number (either prob or logit)
    if (typeof raw === 'number') {
      // if outside [0,1], treat as logit
      return raw < 0 || raw > 1 ? this.sigmoid(raw) : raw
    }

    // 2) array of numbers
    if (Array.isArray(raw)) {
      const arr = raw as number[]
      if (arr.length === 0) return 0.5
      if (arr.length === 1) {
        const v = arr[0]
        return v < 0 || v > 1 ? this.sigmoid(v) : v
      }
      // assume 2-way softmax [p(reject), p(admit)] or [logit0, logit1]
      const [a, b] = arr
      // if they already look like probs that sum≈1, use b
      const s = a + b
      if (s > 0.999 && s < 1.001 && a >= 0 && b >= 0) return b
      // otherwise treat as two logits -> softmax(1)
      const m = Math.max(a, b)
      const ea = Math.exp(a - m),
        eb = Math.exp(b - m)
      return eb / (ea + eb)
    }

    // 3) object-shaped outputs (rare) — try common fields
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

    // fallback
    return 0.5

    function clamp01(x: number) {
      return Math.max(0, Math.min(1, x))
    }
  }

  /** Progress-aware threshold with safe bounds */
  private dynamicThreshold(status: GameStatusRunning<any>): number {
    const base = this.cfg.baseThreshold ?? 0.32
    const minT = this.cfg.minThreshold ?? 0.22
    const maxT = this.cfg.maxThreshold ?? 0.62
    const urgency = this.cfg.urgencyFactor ?? 1.0
    const progress = Math.min(1, status.admittedCount / Math.max(1, Conf.MAX_ADMISSIONS))
    const theta = base + 0.18 * urgency * progress
    return Math.max(minT, Math.min(maxT, theta))
  }

  /**
   * IMPORTANT: countsOverride lets the trainer pass the same counts it used to
   * encode the state, so train/test features match.
   */
  public admit(status: GameStatusRunning<any>, countsOverride?: Record<string, number>): boolean {
    const counts = countsOverride ?? this.counts

    if (!counts) {
      throw new Error('NeuralNetBouncer: Missing counts!')
    }

    const x = this.encoder.encode(status, counts)

    if (!NeuralNet.isNeuralNet(this.net)) {
      throw new Error('NeuralNetBouncer: Neural net is not defined!')
    }

    // Call your NN. Prefer `forward`, else fall back to `inference`.
    const raw = this.net.forward?.(x) ?? this.net.infer(x)
    const p = this.toProbability(raw)
    const theta = this.dynamicThreshold(status)

    // epsilon-greedy exploration (if you keep it here)
    const eps = this.cfg.explorationRate ?? 0
    if (eps > 0 && Math.random() < eps) {
      return Math.random() < 0.5
    }
    return p >= theta
  }

  // required by interface...

  public getOutput(lastStatus: GameStatusCompleted | GameStatusFailed) {
    return {}
  }

  getProgress() {
    return {}
  }
}
