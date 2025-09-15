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
      isProduction?: boolean
      softGates?: boolean
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
    const base = this.cfg.baseThreshold ?? 0.27 // was 0.28
    const minT = this.cfg.minThreshold ?? 0.16 // was 0.18
    const maxT = this.cfg.maxThreshold ?? 0.58 // was 0.60
    const optimism = this.cfg.optimism ?? 0.8 // 0..1

    const progress = Math.min(1, status.admittedCount / Math.max(1, Conf.MAX_ADMISSIONS))
    const slope = 0.09 // was 0.10 — gentler tightening

    // optimism nudges the threshold downward up to ~0.04
    const adj = (optimism - 0.5) * 0.08
    const theta = base + slope * progress - adj

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

    // --- Preconditions
    if (!NeuralNet.isNeuralNet(this.net)) {
      throw new Error('NeuralNetBouncer: Neural net is not defined!')
    }
    if (!counts) {
      throw new Error('NeuralNetBouncer: Missing counts!')
    }

    const guest = status.nextPerson.attributes

    // --- Model score + dynamic threshold
    const x = this.encoder.encode(status, counts)
    const raw = (this.net as any).forward?.(x) ?? (this.net as any).infer(x)
    const p = this.toProbability(raw)
    const thetaBase = this.dynamicThreshold(status)

    // --- Quota bookkeeping
    const needed = this.unmetNeeds(counts)
    const neededKeys = Object.keys(needed)
    const seatsLeft = Math.max(0, Conf.MAX_ADMISSIONS - status.admittedCount)

    const needSum = neededKeys.reduce((s, k) => s + (needed[k] || 0), 0)
    const hits = neededKeys.filter((a) => guest[a]).length

    // --- SOFT-GATES MODE: admit aggressively, tiny/endgame windows only
    if (this.cfg.softGates) {
      if (neededKeys.length === 0) return true

      // If we are badly behind on quotas, only block total neutrals to save seats.
      const extremeDeficit = needSum > seatsLeft * 1.5
      if (extremeDeficit && hits === 0) return false

      // Micro endgame window (much smaller than normal)
      const FEW_CUTOFF = 3
      const ENDGAME_SEATS = 2
      const stillNeeded = neededKeys.filter((a) => needed[a] > 0)
      const nearCritical = stillNeeded.filter((a) => needed[a] <= FEW_CUTOFF)
      const fewNeed = nearCritical.reduce((s, a) => s + needed[a], 0)
      if (nearCritical.length && seatsLeft <= fewNeed) {
        if (!nearCritical.some((a) => guest[a])) return false
      }

      // Loosen threshold to push rejections down
      const theta = Math.max(0, thetaBase - 0.1 - 0.02 * Math.min(2, hits))
      return p >= theta
    }

    // If all quotas are satisfied, be optimistic — admit to finish quickly.
    if (neededKeys.length === 0) return true

    // Safety gates
    const [required, critical] = this.getSafetyGates(status, needed)

    // REQUIRED: must match *any one* required attr.
    if (required.length && !required.some((a) => guest[a])) return false

    // CRITICAL: must match *any one* critical attr.
    if (critical.length && !critical.some((a) => guest[a])) return false

    // --- Endgame pressure rules
    // Tunables up top for clarity
    const FEW_CUTOFF = 7 // treat need <= 7 as "near-critical"
    const ENDGAME_SEATS = 5 // final-seat gating begins here
    const ENDGAME_SLACK = 0 // zero slack when near quotas
    const THRESH_HIT_BONUS = 0.02 // per unmet-attr hit
    const THRESH_MAX_HITS = 2 // cap for hit bonuses
    const THRESH_TIGHT_BONUS = 0.02 // extra relief when seats ~= needs

    // Compute “near-critical” set and totals once
    const stillNeeded = neededKeys.filter((a) => needed[a] > 0)
    const nearCritical = stillNeeded.filter((a) => needed[a] <= FEW_CUTOFF)
    const fewNeed = nearCritical.reduce((s, a) => s + needed[a], 0)
    const totalNeed = stillNeeded.reduce((s, a) => s + needed[a], 0)

    // If seats are just enough to cover the last few, require a hit on those.
    if (nearCritical.length && seatsLeft <= fewNeed + ENDGAME_SLACK) {
      if (!nearCritical.some((a) => guest[a])) return false
    }

    // Production endgame: with very few seats, require help on *still-needed* attrs.
    if ((this.cfg as any).isProduction) {
      const endgameAttrs = nearCritical.length ? nearCritical : critical
      if (endgameAttrs.length && seatsLeft <= ENDGAME_SEATS) {
        if (!endgameAttrs.some((a) => guest[a])) return false
      }
    }

    // --- Optional exploration (usually 0 in prod)
    const eps = this.cfg.explorationRate ?? 0
    if (eps > 0 && Math.random() < eps) return Math.random() < 0.5

    // --- Threshold relief for unmet-attr hits (helps avoid “miss-by-1” endings)
    const hitsUnmet = stillNeeded.reduce((n, a) => n + (guest[a] ? 1 : 0), 0)
    let thetaAdj = 0

    if (hitsUnmet > 0) {
      thetaAdj += THRESH_HIT_BONUS * Math.min(THRESH_MAX_HITS, hitsUnmet) // up to ~0.04
    }
    if (seatsLeft > 0 && totalNeed > 0 && seatsLeft <= totalNeed + 1) {
      thetaAdj += THRESH_TIGHT_BONUS // tiny extra nudge in tight endgames
    }

    const theta = Math.max(0, thetaBase - thetaAdj)

    // --- Final decision
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
