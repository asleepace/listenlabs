/* eslint-disable max-classes-per-file */

import type { BerghainBouncer } from './berghain'
import type { GameState, GameStatusCompleted, GameStatusFailed, GameStatusRunning, ScenarioAttributes } from './types'
import { Disk } from './utils/disk'
import { dump } from './utils/dump'

/* =========================
   ✅ TUNING CONFIG (all knobs)
   ========================= */
const CFG = {
  // Included on persisted data to identify models and biases
  MODEL_VERSION: 2.3,

  // Display / reporting
  UI: {
    PROGRESS_DENOM: 1_000, // denominator for overall progress string
    TOPK_ATTRS: 3, // top scarcity entries in progress
  },

  // Capacity / schedule (flattened target → keeps admit rate steady)
  TARGET_RATE_BASE: { early: 0.18, mid: 0.18, late: 0.18 },
  TARGET_RATE_MIN: 0.16,

  // Price model (shadow prices)
  PRICE: {
    priorTrue: 2, // Beta prior for frequency (alpha)
    priorTotal: 8, // Beta prior total (alpha+beta)
    k0: 0.8, // optimism scale at start (UCB), fades with used (range 0.6–1.2)
    slope: 16.0, // squashing slope → price jump as need > supply
    synergy: 0.12, // small bonus for covering multiple urgent attrs
    paceSlack: 0.02, // allow a tiny progress gap before any pace nudges
    paceBrake: 0.6, // reduce price when ahead of pace (keeps rate flat)
    paceBoost: 0.95, // increase price when behind pace (keeps bars together)
    aheadPenalty: { scale: 0.8, max: 1.1 }, // admit-time nudge vs ahead-of-pace even before satisfied
    overshootPenaltyMax: 4.0, // cap on overshoot tax per attribute (per-decision)
    rareScarcityScale: 1.5, // multiplier for scarcity feature strength
  },

  PACE: {
    lagSlack: 0.02, // ignore tiny lag noise
    worstLagGateStart: [
      // scarcity-like gate but by worst pace lag ratio
      { ratio: 3.0, start: 0.7 },
      { ratio: 1.5, start: 0.8 },
      { ratio: 0.7, start: 0.9 },
      { ratio: 0.0, start: 0.95 },
    ],
    helperBonus: 0.55, // raw-value bonus if helps worst-lag attr
    nonHelperMalus: 0.45, // raw-value malus if doesn’t help worst-lag attr late
    lateCutoverUsed: 0.6, // start applying helper/malus around here
    gateLagSlack: 0.02, // consider "helps lagging" if progress < used - slack
    scarcityCap: 10.0, // cap for scarcity/ratio to avoid runaway prices
  },

  // Linear bandit (dimension is determined dynamically)
  BANDIT: {
    warmStartN: 100, // was hardcoded slice(-100)
    noiseBase: 0.2, // exploration noise base
    noiseDecaySteps: 400, // steps until half-decayed

    eta: 0.15, // learning rate
    hintEta: 0.2, // positive hint learning rate
    emaBeta: 0.035, // admit-rate EMA
    iBeta: 0.002, // integral smoothing
    weightClamp: [-5, 5] as const,
    capacityFloor: -0.1, // capacity weight ≤ this (cap at -0.1 so it can't drift positive)
    scarcityFloor: 0.0, // scarcity weight ≥ this
    initRecentValues: { n: 60, start: 0.3, step: 0.02 },
    recentCap: 500,
    earlyNoiseBoostDecisions: 1200, // more exploration early
    indicatorPrior: 1.8, // prior for each constraint indicator
    capacityPrior: -0.8, // prior for capacity feature
    scarcityPrior: 1.2, // prior for scarcity feature
    warmStartDecay: 0.9, // decay for history warm start
    updateClamp: [-50, 50] as const, // clamp for per-update reward
    softClip: 8, // tanh soft clip for decision series
    priorScale: { min: 0.3, max: 1.4 }, // clamps how much we scale up/down priors
  },

  // Thresholding (robust quantile + PI controller)
  THRESH: {
    sigmaFloor: 0.2, // avoid too-tight thresholds early
    madToSigma: 1.4826, // MAD→sigma constant
    clipSigma: 3.0, // bound thresholds within ±clipSigma σ
    warmupDecisions: 300,
    warmupErrCap: 0.25,
    floorBumpBase: 0.25,
    floorBumpSlope: 1.25,
    capacityBiasScale: { early: 0.32, late: 0.51 }, // * used * sigma
    ctrlGains: { kP: 1.2, kI: 0.4, boostEdge: 0.25, boostFactor: 1.15 },
  },

  // Feasibility-based late gate
  GATE: {
    // choose first where maxRatio > ratio, else fallback to last
    SEQUENCE: [
      { ratio: 300, start: 0.7 },
      { ratio: 150, start: 0.8 },
      { ratio: 60, start: 0.9 },
      { ratio: -Infinity, start: 0.95 },
    ],
    lagSlack: 0.02, // consider helping if attr progress < used - lagSlack
  },

  // Lagger-first safety: never reject someone who closes the most-behind unmet constraint
  LAGGER_FORCE: {
    enable: true,
    minLag: 0.03, // minimum lag vs used to trigger force admit
  },

  // Reward clamp (final safety)
  REWARD: {
    clamp: [-2, 6] as const,
  },

  // Feature engineering
  FEATURES: {
    capacityExp: 0.8, // exponent on used fraction
    scarcityCap: 5.0, // cap on scarcity feature before scaling
  },

  HINTS: {
    synergyPairCap: 3, // pairs to include for synergy
    antiHintScale: 0.9, // strength for ahead-of-pace anti-hint
    abundance: { scale: 0.25, max: 0.6 }, // mild nudge against over-abundant attrs
  },

  STATS: {
    softClipS: 8, // tanh soft clip scale for recentValues
  },

  // Fill-to-capacity helper when all constraints are satisfied
  FILL: {
    enableAtUsed: 0.9, // once ≥90% used and constraints met -> just admit
    learn: false, // don't train on pure fill admits
  },

  // Small endgame helper (only “gate” we keep)
  FINISH: {
    enableAtUsed: 0.9,
    maxShortfall: 10, // admit helpful people when smallest gap ≤ 10
    ratioMin: 3, // OR remainingSlots / smallestShortfall ≥ 3
  },

  // Learning warmup
  WARMUP: {
    usedMax: 0.1,
    minEma: 0.02,
    negClampAfter: -0.3, // clamp negative reward updates after warmup
  },

  // Optional epsilon-admit very early (break stalemates)
  EXPLORE: { epsAdmit: 0.03, epsUntilUsed: 0.08 },

  // Debug / logging
  DEBUG: {
    keepLastLogs: 2,
    includeThresholdBlock: false,
  },

  // Final score shaping (global)
  SCORE: {
    overshootSlack: 5, // free buffer per attribute (ignores small drift)
    overshootL1: 0.75, // linear cost per extra admit beyond the slack
    overshootL2: 0.02, // gentle quadratic kicker (keeps huge overshoots costly)
    weightByScarcity: true, // weight overshoot by current scarcity of that attribute
  },
}

/* ================
   Types & Interfaces
   ================ */
interface GameResult {
  gameId: string
  finalScore: number
  timestamp: Date
  constraints: any[]
  decisions: DecisionRecord[]
  snapshot?: BanditSnapshot
}

interface DecisionRecord {
  context: Context
  action: 'admit' | 'reject'
  person: Record<string, boolean>
  reward: number
  features: number[]
  banditValue: number
  heuristicValue: number
}

interface Context {
  admittedCount: number
  remainingSlots: number
  constraintProgress: number[]
  constraintShortfalls: number[]
}

interface Config {
  TOTAL_PEOPLE: number
  TARGET_RANGE: number
  MAX_CAPACITY: number
}

type Person<T> = Record<keyof T, boolean>

type BanditSnapshot = {
  A: number[][]
  b: number[]
  weights: number[]
  admitRateEma: number
  recentValues: number[]
  featureDim: number
  maxCapacity: number
  indicatorCount?: number // for compatibility with older snapshots
}

interface Statistics<T> {
  correlations: Record<keyof T, Record<keyof T, number>>
  relativeFrequencies: Record<keyof T, number>
}

/* ==================
  Helpers
   ================== */

// Format a fraction in [0,1] as a percentage 0–100 with rounding.
function pct(n: number, decimals = 1): number {
  const v = Math.max(0, Math.min(100, 100 * n))
  return +v.toFixed(decimals)
}

/* ==================
   Constraint (with EF)
   ================== */
class Constraint<T> {
  public admitted = 0
  public rejected = 0
  private seenTotal = 0
  private seenTrue = 0

  constructor(
    public attribute: keyof T,
    public minRequired: number,
    public frequency: number, // prior
    public config: Config
  ) {}

  update(person: Person<T>, admitted: boolean): void {
    this.seenTotal++
    if (person[this.attribute]) {
      this.seenTrue++
      if (admitted) this.admitted++
      else this.rejected++
    }
  }

  // Expose counters for Beta posterior
  getSeenTrue(): number {
    return this.seenTrue
  }
  getSeenTotal(): number {
    return this.seenTotal
  }

  // in Constraint<T>
  getTargetShare(): number {
    return this.minRequired / this.config.MAX_CAPACITY
  }
  getSupplySkew(): number {
    const ef = this.getEmpiricalFrequency() || this.frequency || 1e-6
    return ef / Math.max(1e-6, this.getTargetShare()) // >1 means over-abundant
  }

  getEmpiricalFrequency(): number {
    const priorTrue = CFG.PRICE.priorTrue
    const priorTotal = CFG.PRICE.priorTotal
    const num = this.seenTrue + priorTrue
    const den = this.seenTotal + priorTotal
    return Math.max(1e-6, num / Math.max(1, den))
  }

  getProgress(): number {
    return Math.min(1.0, this.admitted / Math.max(1, this.minRequired))
  }

  getShortfall(): number {
    return Math.max(0, this.minRequired - this.admitted)
  }

  isSatisfied(): boolean {
    return this.admitted >= this.minRequired
  }

  getScarcity(remainingSlots: number): number {
    if (this.isSatisfied()) return 0
    const f = this.getEmpiricalFrequency() || this.frequency
    const expectedAvailable = remainingSlots * f
    return this.getShortfall() / Math.max(0.1, expectedAvailable)
  }
}

/* ===============
   Linear Bandit
   =============== */
class LinearBandit {
  static randomInitialValues(n = CFG.BANDIT.initRecentValues.n) {
    const { start, step } = CFG.BANDIT.initRecentValues
    return Array.from({ length: n }, (_, i) => start + step * i)
  }

  static fromSnapshot(s: BanditSnapshot, indicatorCountGuess: number) {
    const ic = s.indicatorCount ?? Math.max(1, Math.min(s.featureDim - 2, indicatorCountGuess))
    const lb = new LinearBandit(s.featureDim, s.maxCapacity, ic, undefined, s.weights)
    lb.A = s.A
    lb.b = s.b
    lb.weights = s.weights
    lb.recentValues = LinearBandit.randomInitialValues()
    lb.admitRateEma = lb.getTargetAdmitRate()
    return lb
  }

  private weights!: number[]
  private A!: number[][]
  private b!: number[]
  private featureDim: number
  private indicatorCount: number
  private capIdx: number
  private scarIdx: number

  private decisionCount = 0
  private lastThreshold = 9
  private lastRawValue = 9

  public totalAdmitted = 0
  public maxCapacity: number
  public maxFeasRatio = 0 // unused now, kept for compatibility

  private recentValues: number[] = []
  private recentCap = CFG.BANDIT.recentCap

  public targetAdmitRate = 0.18
  private eta = CFG.BANDIT.eta
  public admitRateEma = this.targetAdmitRate
  private emaBeta = CFG.BANDIT.emaBeta

  private rateErrI = 0
  private iBeta = CFG.BANDIT.iBeta

  private lastThrDbg = {
    med: 0,
    mad: 0,
    sigma: 0,
    z: 0,
    quantileEstimate: 0,
    capacityBias: 0,
    rateAdjustment: 0,
    urgencyAdj: 0,
    lo: 0,
    hi: 0,
    final: 0,
    target: 0,
    err: 0,
    tail: 0,
  }

  constructor(
    featureDimension: number,
    maxCapacity: number,
    indicatorCount: number,
    priorWeights?: number[],
    snapshotWeights?: number[]
  ) {
    this.featureDim = featureDimension
    this.maxCapacity = maxCapacity
    this.indicatorCount = indicatorCount
    this.capIdx = indicatorCount
    this.scarIdx = indicatorCount + 1
    this.reset(priorWeights, snapshotWeights)
  }

  /** allow warm start without exposing a private method */
  public warmStartFromHistory(decisions: DecisionRecord[]) {
    const valid = decisions.filter((d) => d.features?.length === this.featureDim).slice(-CFG.BANDIT.warmStartN)
    valid.forEach((d, idx) => {
      const age = valid.length - idx
      const w = Math.pow(CFG.BANDIT.warmStartDecay, age)
      this.updateModel(d.features, d.reward * w)
    })
    if (valid.length) this.updateWeights()
  }

  private makeDefaultPriors(dim: number, indicatorCount: number): number[] {
    const p: number[] = []
    for (let i = 0; i < indicatorCount; i++) p.push(CFG.BANDIT.indicatorPrior)
    p.push(CFG.BANDIT.capacityPrior) // capacity
    p.push(CFG.BANDIT.scarcityPrior) // scarcity feature
    while (p.length < dim) p.push(0) // defensive
    return p.slice(0, dim)
  }

  private reset(priorWeights?: number[], snapshotWeights?: number[]) {
    const lambda = 0.1
    this.A = Array.from({ length: this.featureDim }, (_, i) =>
      Array.from({ length: this.featureDim }, (_, j) => (i === j ? lambda : 0))
    )
    const priors =
      snapshotWeights && snapshotWeights.length === this.featureDim
        ? snapshotWeights
        : priorWeights && priorWeights.length === this.featureDim
        ? priorWeights
        : this.makeDefaultPriors(this.featureDim, this.indicatorCount)

    this.b = priors.map((w) => lambda * w)
    this.weights = [...priors]
    this.recentValues = LinearBandit.randomInitialValues()
    this.admitRateEma = this.getTargetAdmitRate()
  }

  toSnapshot(): BanditSnapshot {
    return {
      A: this.A,
      b: this.b,
      weights: this.weights,
      admitRateEma: this.admitRateEma,
      recentValues: this.recentValues.slice(-this.recentCap),
      featureDim: this.featureDim,
      maxCapacity: this.maxCapacity,
      indicatorCount: this.indicatorCount,
    }
  }

  public getTargetAdmitRate(): number {
    const used = this.totalAdmitted / Math.max(1, this.maxCapacity)
    const base =
      used < 0.33 ? CFG.TARGET_RATE_BASE.early : used < 0.66 ? CFG.TARGET_RATE_BASE.mid : CFG.TARGET_RATE_BASE.late
    return Math.max(CFG.TARGET_RATE_MIN, base)
  }

  getScoreSummary() {
    const s = [...this.recentValues].sort((a, b) => a - b)
    const pick = (p: number) => (s.length ? s[Math.floor(p * (s.length - 1))] : 0)
    return { n: s.length, p10: pick(0.1), p50: pick(0.5), p90: pick(0.9) }
  }

  updateController(admitted: boolean) {
    this.admitRateEma = this.admitRateEma * (1 - this.emaBeta) + (admitted ? 1 : 0) * this.emaBeta
  }

  private median(xs: number[]): number {
    if (!xs.length) return 0
    const a = [...xs].sort((x, y) => x - y)
    const m = Math.floor(a.length / 2)
    return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m])
  }
  private mad(xs: number[], med: number): number {
    if (!xs.length) return 1e-6
    const dev = xs.map((v) => Math.abs(v - med)).sort((x, y) => x - y)
    const m = Math.floor(dev.length / 2)
    const mad = dev.length % 2 ? dev[m] : 0.5 * (dev[m - 1] + dev[m])
    return Math.max(mad, 1e-6)
  }

  // Inverse normal CDF approximation (A&S 26.2.23)
  private invNorm(p: number): number {
    const pp = Math.max(1e-9, Math.min(1 - 1e-9, p))
    const a1 = -39.69683028665376,
      a2 = 220.9460984245205,
      a3 = -275.9285104469687,
      a4 = 138.357751867269,
      a5 = -30.66479806614716,
      a6 = 2.506628277459239
    const b1 = -54.47609879822406,
      b2 = 161.5858368580409,
      b3 = -155.6989798598866,
      b4 = 66.80131188771972,
      b5 = -13.28068155288572
    const c1 = -0.007784894002430293,
      c2 = -0.3223964580411365,
      c3 = -2.400758277161838,
      c4 = -2.549732539343734,
      c5 = 4.374664141464968,
      c6 = 2.938163982698783
    const d1 = 0.007784695709041462,
      d2 = 0.3224671290700398,
      d3 = 2.445134137142996,
      d4 = 3.754408661907416

    let q: number, r: number
    if (pp < 0.02425) {
      q = Math.sqrt(-2 * Math.log(pp))
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    } else if (pp > 1 - 0.02425) {
      q = Math.sqrt(-2 * Math.log(1 - pp))
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    }
    q = pp - 0.5
    r = q * q
    return (
      ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
      (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    )
  }
  private zForTail(tail: number): number {
    return this.invNorm(tail)
  }

  private calculateAdaptiveThreshold(): number {
    const arr = this.recentValues.length ? this.recentValues : [0]
    const med = this.median(arr)
    const mad = this.mad(arr, med)
    const sigma = Math.max(CFG.THRESH.madToSigma * mad, CFG.THRESH.sigmaFloor)

    const target = this.getTargetAdmitRate()
    const err = this.admitRateEma - target
    const tail = 1 - target
    const z = this.zForTail(tail)
    const quantileEstimate = med + z * sigma

    const used = this.totalAdmitted / Math.max(1, this.maxCapacity)
    const capK = used < 0.5 ? CFG.THRESH.capacityBiasScale.early : CFG.THRESH.capacityBiasScale.late
    const capacityBias = capK * used * sigma

    // PI controller
    const { kP, kI, boostEdge, boostFactor } = CFG.THRESH.ctrlGains
    this.rateErrI = (1 - this.iBeta) * this.rateErrI + this.iBeta * err
    const boost = err > boostEdge ? boostFactor : 1
    const rateAdjustment = kP * boost * err + kI * boost * this.rateErrI

    let thr = quantileEstimate + capacityBias + rateAdjustment

    // floor vs err (more rejections if over target)
    if (err > 0) {
      const warm = this.decisionCount < CFG.THRESH.warmupDecisions
      const cap = warm ? CFG.THRESH.warmupErrCap : 0.6
      const bump = CFG.THRESH.floorBumpBase + CFG.THRESH.floorBumpSlope * Math.min(cap, Math.max(0, err))
      thr = Math.max(thr, med + bump * sigma)
    }

    const lo = med - CFG.THRESH.clipSigma * sigma
    const hi = med + CFG.THRESH.clipSigma * sigma
    if (!Number.isFinite(thr)) thr = med
    const final = Math.max(lo, Math.min(hi, thr))

    this.lastThrDbg = {
      med,
      mad,
      sigma,
      z,
      quantileEstimate,
      capacityBias,
      rateAdjustment,
      urgencyAdj: 0,
      lo,
      hi,
      final,
      target,
      err,
      tail,
    }
    return final
  }

  private assertFeatureDim(f: number[]) {
    if (f.length !== this.indicatorCount + 2) {
      throw new Error(`Feature length ${f.length} != expected ${this.indicatorCount + 2}`)
    }
  }

  // in class LinearBandit
  selectAction(features: number[], bias = 0) {
    this.assertFeatureDim(features)
    this.decisionCount++
    this.updateWeights()

    const rawValue = this.predictValue(features) + bias // <-- add bias here
    const base = CFG.BANDIT.noiseBase
    const decay = Math.min(1, this.decisionCount / CFG.BANDIT.noiseDecaySteps)
    const boost = this.decisionCount < CFG.BANDIT.earlyNoiseBoostDecisions ? 1.5 : 1.0
    const noise = (Math.random() - 0.5) * (base * boost * (1 - 0.5 * decay))
    const decisionVar = rawValue + noise
    this.pushRaw(decisionVar)
    const threshold = this.calculateAdaptiveThreshold()

    this.lastThreshold = threshold
    this.lastRawValue = rawValue

    const action = decisionVar > threshold ? 'admit' : 'reject'
    return { action, value: rawValue } as const
  }

  private predictValue(features: number[]): number {
    let v = 0
    const n = Math.min(features.length, this.weights.length)
    for (let i = 0; i < n; i++) v += this.weights[i] * features[i]
    return v
  }

  private updateWeights() {
    // diagonal-only update → weights[i] = b[i]/A[i][i]
    for (let i = 0; i < this.featureDim; i++) if (this.A[i][i] > 1e-6) this.weights[i] = this.b[i] / this.A[i][i]

    const clamp = (x: number, [lo, hi]: readonly [number, number]) => Math.max(lo, Math.min(hi, x))

    // global clamp
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = clamp(this.weights[i], CFG.BANDIT.weightClamp)

    // keep capacity weight ≤ capacityFloor (cap at -0.1 so it can't drift positive)
    if (this.capIdx < this.weights.length) {
      this.weights[this.capIdx] = Math.min(this.weights[this.capIdx], CFG.BANDIT.capacityFloor)
    }
    if (this.scarIdx < this.weights.length) {
      this.weights[this.scarIdx] = Math.max(CFG.BANDIT.scarcityFloor, this.weights[this.scarIdx])
    }

    // warmup: prevent indicators from going negative early
    const used = this.totalAdmitted / Math.max(1, this.maxCapacity)
    const warmup = used < CFG.WARMUP.usedMax || this.admitRateEma < CFG.WARMUP.minEma
    if (warmup) {
      for (let i = 0; i < this.indicatorCount; i++) this.weights[i] = Math.max(0, this.weights[i])
    }
  }

  updateModel(features: number[], reward: number) {
    this.assertFeatureDim(features)
    const [lo, hi] = CFG.BANDIT.updateClamp
    const r = Math.max(lo, Math.min(hi, reward))
    const n = Math.min(features.length, this.weights.length)
    for (let i = 0; i < n; i++) {
      const f = features[i]
      this.A[i][i] += this.eta * (f * f)
      this.b[i] += this.eta * (r * f)
    }
  }

  getStats() {
    const absAvg = this.weights.reduce((a, b) => a + Math.abs(b), 0) / Math.max(1, this.weights.length)
    return {
      weights: this.weights.slice(0, Math.min(6, this.weights.length)),
      decisionCount: this.decisionCount,
      capacityUsed: +(this.totalAdmitted / Math.max(1, this.maxCapacity)).toFixed(3),
      threshold: +this.lastThreshold.toFixed(2),
      avgWeight: +absAvg.toFixed(2),
    }
  }

  getLastRawValue() {
    return this.lastRawValue
  }
  getLastThreshold() {
    return this.lastThreshold
  }
  getThresholdDebug() {
    return this.lastThrDbg
  }

  private pushRaw(v: number) {
    const s = CFG.BANDIT.softClip
    const soft = s * Math.tanh(v / s)
    this.recentValues.push(soft)
    if (this.recentValues.length > this.recentCap) this.recentValues.shift()
  }
}

/* =================
   BanditBouncer<T>
   ================= */
export class BanditBouncer<T> implements BerghainBouncer {
  public constraints = new Map<keyof T, Constraint<T>>()
  public totalAdmitted = 0
  public totalRejected = 0
  private bandit!: LinearBandit
  private decisions: DecisionRecord[] = []
  private logs: string[] = []
  private indicatorCount = 0 // number of constraint indicators used in features
  private pretrained = false

  constructor(public state: GameState, public config: Config) {
    this.initializeConstraints()
  }

  /* --- constraint helpers --- */
  private getConstraints(): Constraint<T>[] {
    return Array.from(this.constraints.values())
  }
  private remaining() {
    return this.config.MAX_CAPACITY - this.totalAdmitted
  }
  private usedFrac() {
    return this.totalAdmitted / this.config.MAX_CAPACITY
  }

  private getFeasibilityRatios() {
    const used = this.usedFrac()
    const remaining = this.remaining()
    const ratios = this.getConstraints()
      .filter((c) => !c.isSatisfied())
      .map((c) => {
        const ef = c.getEmpiricalFrequency() || c.frequency || 1e-6
        const expect = Math.max(1e-6, remaining * ef)
        const lag = this.paceLag(c, used) // ← pace lag (target - admitted at current used)
        return {
          attr: String(c.attribute),
          ratio: lag / expect, // pace-based ratio
          lag, // absolute lag (people)
          freq: ef,
        }
      })
      .sort((a, b) => b.ratio - a.ratio)

    const max = ratios.length ? ratios[0].ratio : 0
    const most = ratios.length ? (ratios[0].attr as string) : undefined
    return { max, mostCritical: most, ratios }
  }

  /* --- small utilities --- */
  private log(line: string) {
    this.logs.push(line)
    const keep = CFG.DEBUG.keepLastLogs
    if (this.logs.length > keep) this.logs.splice(0, this.logs.length - keep)
  }

  private recordDecision(
    person: Person<T>,
    action: 'admit' | 'reject',
    reward: number,
    value: number,
    features: number[]
  ) {
    const ctx = this.buildContext()
    this.decisions.push({
      context: ctx,
      action,
      person,
      reward,
      features,
      banditValue: value,
      heuristicValue: 0,
    })
    const usefulAttrs = this.getConstraints()
      .filter((c) => person[c.attribute] && !c.isSatisfied())
      .map((c) => String(c.attribute))
    this.log(`${action.toUpperCase()}: r=${reward.toFixed(1)} v=${value.toFixed(1)} useful=[${usefulAttrs.join(',')}]`)
  }

  /* --- initialization --- */
  initializeConstraints(): void {
    for (const gc of this.state.game.constraints) {
      const attribute = gc.attribute as keyof T
      this.constraints.set(attribute, new Constraint(attribute, gc.minCount, this.getFrequency(attribute), this.config))
    }
    // identify rarest attribute by prior frequency (fallback to first)
    const all = this.getConstraints()
    const rare = all.slice().sort((a, b) => (a.frequency ?? 1) - (b.frequency ?? 1))[0]
    this.indicatorCount = all.length
  }

  async initializeLearningData() {
    const prev = await this.getPreviousGameResults()
    const all = prev.flatMap((g) => g.decisions || [])
    const snap = prev.at(0)?.snapshot

    const dim = this.indicatorCount + 2 // [indicators] + capacity + scarcity
    const priors = this.buildPriorWeights(dim, this.indicatorCount)

    if (snap && snap.featureDim === dim) {
      this.bandit = LinearBandit.fromSnapshot(snap, this.indicatorCount)
      this.pretrained = true
    } else {
      this.bandit = new LinearBandit(dim, this.config.MAX_CAPACITY, this.indicatorCount, priors)
      this.pretrained = false
    }
    if (!snap || snap.featureDim !== dim) {
      // warm start from recent decisions only when we rebuilt
      this.bandit.warmStartFromHistory(all)
    }
  }

  private buildPriorWeights(dim: number, indicatorCount: number): number[] {
    const p: number[] = []
    const cs = this.getConstraints()
    for (let i = 0; i < indicatorCount; i++) {
      const c = cs[i]
      const skew = c?.getSupplySkew?.() ?? 1
      // if skew > 1 (over-abundant), scale prior down; if < 1, scale up a bit
      const scale = Math.max(CFG.BANDIT.priorScale.min, Math.min(CFG.BANDIT.priorScale.max, 1 / skew))
      p.push(CFG.BANDIT.indicatorPrior * scale)
    }
    p.push(CFG.BANDIT.capacityPrior)
    p.push(CFG.BANDIT.scarcityPrior)
    while (p.length < dim) p.push(0)
    return p.slice(0, dim)
  }

  get statistics(): Statistics<T> {
    return this.state.game.attributeStatistics as any
  }
  getFrequency(attribute: keyof T): number {
    return (this.statistics?.relativeFrequencies?.[attribute] as number) ?? 0.001
  }

  /* --- shadow prices --- */
  private computeShadowPrices() {
    const used = this.usedFrac()
    const remaining = Math.max(1, this.remaining())
    const k = CFG.PRICE.k0 * (1 - used)
    const prices: Record<string, number> = {}
    const S = (x: number) => 1 / (1 + Math.exp(-CFG.PRICE.slope * x))

    for (const c of this.getConstraints()) {
      if (c.isSatisfied()) {
        prices[String(c.attribute)] = 0
        continue
      }
      const seenTrue = c.getSeenTrue()
      const seenTotal = c.getSeenTotal()
      const alpha0 = CFG.PRICE.priorTrue
      const beta0 = Math.max(0, CFG.PRICE.priorTotal - CFG.PRICE.priorTrue)
      const alpha = alpha0 + seenTrue
      const beta = beta0 + Math.max(0, seenTotal - seenTrue)
      const mean = alpha / Math.max(1, alpha + beta)
      const vRaw = (alpha * beta) / (Math.pow(alpha + beta, 2) * Math.max(1, alpha + beta + 1))
      const sd = Math.sqrt(Math.max(1e-6, vRaw))
      const pUCB = Math.max(0, Math.min(1, mean + k * sd))

      // pace-based need: lag vs expected supply
      const lag = this.paceLag(c, used)
      const ef = c.getEmpiricalFrequency() || c.frequency || 1e-6
      const expect = Math.max(1e-6, remaining * ef)
      const gap = Math.max(0, lag / expect - pUCB) // optimistic supply vs lag need

      // pace brake / boost
      const paceLead = Math.max(0, c.getProgress() - used - CFG.PRICE.paceSlack)
      const paceBrake = CFG.PRICE.paceBrake * paceLead
      const paceLag = Math.max(0, used - c.getProgress() - CFG.PRICE.paceSlack)
      const paceBoost = CFG.PRICE.paceBoost * paceLag

      const price = Math.max(0, Math.min(1, S(gap) - paceBrake + paceBoost))
      prices[String(c.attribute)] = price
    }
    return prices
  }

  /* --- features (simple & stable) ---
     [ one-hot indicators per constraint (unrelated to unmet; keeps stationarity)
       capacity used^capacityExp
       dynamic scarcity feature: strongest scarcity among unmet attrs the person covers ]
  */
  private extractFeatures(person: Person<T>): number[] {
    const cs = this.getConstraints()
    const feats: number[] = []

    // indicators
    cs.forEach((c) => feats.push(person[c.attribute] ? 1 : 0))

    // capacity
    feats.push(Math.pow(this.usedFrac(), CFG.FEATURES.capacityExp))

    // dynamic scarcity among unmet attrs this person helps, pace-based
    const used = this.usedFrac()
    let scarcityBoost = 0
    for (const c of this.getConstraints()) {
      if (!c.isSatisfied() && person[c.attribute]) {
        const ef = c.getEmpiricalFrequency() || c.frequency || 1e-6
        const expect = Math.max(1e-6, this.remaining() * ef)
        const lag = this.paceLag(c, used)
        const r = Math.min(CFG.FEATURES.scarcityCap, lag / expect)
        scarcityBoost = Math.max(scarcityBoost, r)
      }
    }
    feats.push(CFG.PRICE.rareScarcityScale * scarcityBoost)
    return feats
  }

  /* --- price-driven reward (no policy gates) --- */
  private calculateReward(person: Person<T>, admitted: boolean, prices?: Record<string, number>): number {
    const p = prices ?? this.computeShadowPrices()
    const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

    if (!admitted) {
      const maxPrice = Math.max(0, ...Object.values(p))
      // mild rejection penalty scaled by global urgency (keeps learning signal without spikes)
      return clamp(-0.15 - 0.5 * maxPrice, CFG.REWARD.clamp[0], 0)
    }

    // coverage value
    const attrs = this.getConstraints().map((c) => String(c.attribute))
    const covers = attrs.filter((a) => person[a as keyof T])
    let reward = covers.reduce((s, a) => s + (p[a] || 0), 0)

    // small synergy for multi-urgent coverage
    if (covers.length >= 2) {
      const pr = covers.map((a) => p[a] || 0).sort((x, y) => y - x)
      const pairs = Math.min(CFG.HINTS.synergyPairCap, pr.length - 1)
      for (let i = 0; i < pairs; i++) reward += CFG.PRICE.synergy * Math.min(pr[0], pr[i + 1])
    }

    // gentle overshoot & ahead-of-pace taxes
    for (const c of this.getConstraints()) {
      if (person[c.attribute] && c.isSatisfied()) {
        const over = (c.admitted - c.minRequired) / Math.max(1, c.minRequired)
        reward -= Math.min(CFG.PRICE.overshootPenaltyMax, 1.0 + Math.max(0, over))
      } else if (person[c.attribute]) {
        const usedNow = this.usedFrac()
        const paceLead = Math.max(0, c.getProgress() - usedNow - CFG.PRICE.paceSlack)
        if (paceLead > 0) {
          const { scale, max } = CFG.PRICE.aheadPenalty
          reward -= Math.min(max, scale * paceLead)
        }
      }
    }

    return clamp(reward, CFG.REWARD.clamp[0], CFG.REWARD.clamp[1])
  }

  // --- helpers to mutate state consistently ---
  private applyAdmit(person: Person<T>, features: number[], prices: Record<string, number>, valueForLog = 0) {
    const reward = this.calculateReward(person, true, prices)
    this.bandit.updateModel(features, reward)
    this.getConstraints().forEach((k) => k.update(person, true))
    this.totalAdmitted++
    this.bandit.totalAdmitted++
    this.recordDecision(person, 'admit', reward, valueForLog, features)
    this.bandit.updateController(true)
    return true
  }
  private applyReject(
    person: Person<T>,
    features: number[],
    prices: Record<string, number>,
    valueForLog = 0,
    maskIndicators = true
  ) {
    const reward = this.calculateReward(person, false, prices)
    if (maskIndicators) {
      const masked = features.slice()
      for (let i = 0; i < this.indicatorCount; i++) masked[i] = 0
      this.bandit.updateModel(masked, Math.max(CFG.WARMUP.negClampAfter, reward))
    } else {
      this.bandit.updateModel(features, Math.max(CFG.WARMUP.negClampAfter, reward))
    }
    this.getConstraints().forEach((c) => c.update(person, false))
    this.totalRejected++
    this.recordDecision(person, 'reject', reward, valueForLog, features)
    this.bandit.updateController(false)
    return false
  }

  // --- force-finish helper (late-game admit if a person helps close any small gap)
  private tryForceFinish(person: Person<T>, features: number[], prices: Record<string, number>): boolean {
    const used = this.usedFrac()
    if (used < CFG.FINISH.enableAtUsed) return false

    const unmet = this.getConstraints().filter((c) => !c.isSatisfied())
    if (!unmet.length) return false

    const remaining = this.remaining()

    // Compute eligibility for each unmet constraint
    const info = unmet.map((c) => {
      const short = c.getShortfall()
      const roomy = remaining / Math.max(1, short) >= CFG.FINISH.ratioMin
      const eligible = short > 0 && (short <= CFG.FINISH.maxShortfall || roomy)
      return { c, short, roomy, eligible }
    })

    // If exactly one unmet, be decisive
    if (info.length === 1) {
      const { c } = info[0]
      if (person[c.attribute]) return this.applyAdmit(person, features, prices, this.bandit.getLastRawValue?.() ?? 0)
      return false
    }

    // Otherwise any eligible → admit the first that helps (smallest shortfall first)
    const eligible = info.filter((x) => x.eligible).sort((a, b) => a.short - b.short)
    for (const e of eligible) {
      if (person[e.c.attribute]) return this.applyAdmit(person, features, prices, this.bandit.getLastRawValue?.() ?? 0)
    }
    return false
  }

  private paceLag(c: Constraint<T>, used: number) {
    // how far behind the *target* this attr is at current used
    const target = c.minRequired * used
    return Math.max(0, target - c.admitted)
  }

  private worstLagInfo() {
    const used = this.usedFrac()
    const remaining = this.remaining()
    const unmet = this.getConstraints().filter((c) => !c.isSatisfied())
    if (!unmet.length) return { attr: null as keyof T | null, ratio: 0, lag: 0 }

    // ratio = lag / expected available
    let best: { c: Constraint<T>; ratio: number; lag: number } | null = null
    for (const c of unmet) {
      const lag = this.paceLag(c, used)
      const ef = c.getEmpiricalFrequency() || c.frequency || 1e-6
      const expect = Math.max(1e-6, remaining * ef)
      const ratio = Math.min(CFG.PACE.scarcityCap, lag / expect)
      if (!best || ratio > best.ratio) best = { c, ratio, lag }
    }
    return { attr: best?.c?.attribute ?? null, ratio: best?.ratio ?? 0, lag: best?.lag ?? 0 }
  }

  /* --- bouncer API --- */
  admit({ status, nextPerson }: GameStatusRunning<ScenarioAttributes>): boolean {
    if (status !== 'running') return false
    if (this.remaining() <= 0) return false

    const person = nextPerson.attributes as Person<T>
    const features = this.extractFeatures(person)
    const prices = this.computeShadowPrices()
    const used = this.usedFrac()

    // send attributes to remote debug stream
    dump(
      Object.entries(nextPerson.attributes)
        .filter((t) => t[1])
        .map((t) => t[0])
    )

    const unmet = this.getConstraints().filter((c) => !c.isSatisfied())

    // worst-lag information (pace-based)
    const { attr: worstAttr, ratio: worstRatio } = this.worstLagInfo()
    const helpsWorst = !!(worstAttr && person[worstAttr])

    // choose PACE gate start from table
    let gateStart = CFG.PACE.worstLagGateStart.find((x) => worstRatio >= x.ratio)?.start
    if (gateStart === undefined) gateStart = CFG.PACE.worstLagGateStart.at(-1)!.start

    // whether person helps a lagging unmet attr (pace view)
    const helpsLagging = unmet.some((c) => person[c.attribute] && c.getProgress() < used - CFG.PACE.gateLagSlack)

    // Single PACE gate: late, only let in people who help the worst lag (and are actually lagging)
    if (unmet.length && used > gateStart && !(helpsWorst && helpsLagging)) {
      const reward = this.calculateReward(person, false, prices)
      const masked = features.slice()
      for (let i = 0; i < this.indicatorCount; i++) masked[i] = 0
      this.bandit.updateModel(masked, Math.max(CFG.WARMUP.negClampAfter, reward))
      this.getConstraints().forEach((c) => c.update(person, false))
      this.totalRejected++
      this.recordDecision(person, 'reject', reward, 0, features)
      this.bandit.updateController(false)
      return false
    }

    // Lagger-first safety: never reject someone who closes the most-behind unmet constraint
    if (CFG.LAGGER_FORCE.enable && unmet.length) {
      const lagInfo = unmet.map((c) => ({ c, lag: Math.max(0, used - c.getProgress() - CFG.PRICE.paceSlack) }))
      lagInfo.sort((a, b) => b.lag - a.lag)
      const mostLag = lagInfo[0]
      if (mostLag && mostLag.lag >= CFG.LAGGER_FORCE.minLag && person[mostLag.c.attribute]) {
        return this.applyAdmit(person, features, prices, this.bandit.getLastRawValue?.() ?? 0)
      }
    }

    // Scarcity-severe fast path (pace-based, using worstRatio)
    if (unmet.length && worstAttr) {
      const helpsCount = unmet.reduce((k, c) => k + (person[c.attribute] ? 1 : 0), 0)
      const multiHelp = helpsCount >= 2
      const severe = worstRatio >= 2.0 || (worstRatio >= 1.2 && used >= 0.75)
      if ((severe || multiHelp) && person[worstAttr]) {
        return this.applyAdmit(person, features, prices, this.bandit.getLastRawValue?.() ?? 0)
      }
    }

    // Price-based hinting + pace scaling + tiny anti-hint for ahead-of-pace attrs
    const priceLabel = this.getConstraints().map((c) => (person[c.attribute] ? prices[String(c.attribute)] || 0 : 0))
    if (priceLabel.some((p) => p > 0)) {
      const hint = Array(this.indicatorCount + 2).fill(0) // +2 for capacity & scarcity alignment
      const usedNow = this.usedFrac()
      const scaled = priceLabel.slice()
      this.getConstraints().forEach((c, i) => {
        if (!scaled[i]) return
        const lag = Math.max(0, usedNow - c.getProgress() - CFG.PRICE.paceSlack)
        scaled[i] *= 1 + lag // gentle pull for laggards
      })
      for (let i = 0; i < this.indicatorCount; i++) hint[i] = scaled[i]

      const sum = scaled.reduce((a, b) => a + b, 0)
      const hintReward = Math.min(3, sum)
      if (sum > 1e-6) for (let i = 0; i < this.indicatorCount; i++) hint[i] *= hintReward / sum
      this.bandit.updateModel(hint, CFG.BANDIT.hintEta)

      const abundance = Array(this.indicatorCount + 2).fill(0)
      let any = false
      this.getConstraints().forEach((c, i) => {
        if (!person[c.attribute]) return
        const skew = c.getSupplySkew?.() ?? 1
        const lagging = c.getProgress() < usedNow - CFG.PACE.lagSlack
        if (skew > 1 && !lagging) {
          const d = Math.min(CFG.HINTS.abundance.max, CFG.HINTS.abundance.scale * (skew - 1))
          abundance[i] = -d
          any = true
        }
      })
      if (any) this.bandit.updateModel(abundance, CFG.BANDIT.hintEta)

      // ahead-of-pace anti-hint
      const ahead = Array(this.indicatorCount + 2).fill(0)
      let anyAhead = false
      this.getConstraints().forEach((c, i) => {
        if (!person[c.attribute]) return
        const skew = c.getSupplySkew?.() ?? 1
        const slackAdj = Math.max(0, CFG.PRICE.paceSlack - 0.5 * Math.max(0, skew - 1) * CFG.PRICE.paceSlack)
        const paceLead = Math.max(0, c.getProgress() - used - slackAdj)
        if (c.isSatisfied() || paceLead > 0) {
          ahead[i] = -Math.min(0.5, 0.5 * (c.isSatisfied() ? 1 : paceLead))
          anyAhead = true
        }
      })
      if (anyAhead) this.bandit.updateModel(ahead, CFG.HINTS.antiHintScale * CFG.BANDIT.hintEta)
    }

    // Fill-to-capacity if ALL constraints satisfied late
    const allSatisfied = this.getConstraints().every((c) => c.isSatisfied())
    if (allSatisfied && used >= CFG.FILL.enableAtUsed) {
      if (CFG.FILL.learn) this.bandit.updateModel(features, 0)
      this.bandit.updateController(true)
      this.getConstraints().forEach((c) => c.update(person, true))
      this.totalAdmitted++
      this.bandit.totalAdmitted++
      this.recordDecision(person, 'admit', 0, this.bandit.getLastRawValue?.() ?? 0, features)
      return true
    }

    // Endgame helper: admit if it closes a tiny remaining gap
    if (this.tryForceFinish(person, features, prices)) return true

    // Early epsilon-admit for exploration
    if (used < CFG.EXPLORE.epsUntilUsed) {
      const hasAny = this.getConstraints().some((c) => person[c.attribute])
      if (hasAny && Math.random() < CFG.EXPLORE.epsAdmit) {
        return this.applyAdmit(person, features, prices, 0)
      }
    }

    // Bandit decision with small pace-aware bias
    let bias = 0
    if (used > CFG.PACE.lateCutoverUsed) {
      bias += helpsWorst ? CFG.PACE.helperBonus : -CFG.PACE.nonHelperMalus
    }
    const { action, value } = this.bandit.selectAction(features, bias)
    const shouldAdmit = action === 'admit'

    // Learning with warmup rules (mask indicators on rejections)
    const inWarmup = used < CFG.WARMUP.usedMax || this.bandit.admitRateEma < CFG.WARMUP.minEma
    const reward = this.calculateReward(person, shouldAdmit, prices)
    if (shouldAdmit) {
      this.bandit.updateModel(features, reward)
    } else if (!inWarmup) {
      const r = Math.max(CFG.WARMUP.negClampAfter, reward)
      const masked = features.slice()
      for (let i = 0; i < this.indicatorCount; i++) masked[i] = 0
      this.bandit.updateModel(masked, r)
    }

    // state updates
    this.getConstraints().forEach((c) => c.update(person, shouldAdmit))
    if (shouldAdmit) {
      this.totalAdmitted++
      this.bandit.totalAdmitted++
    } else {
      this.totalRejected++
    }

    this.recordDecision(person, action, reward, value, features)
    this.bandit.updateController(shouldAdmit)
    return shouldAdmit
  }

  /* --- progress / output --- */
  private buildContext(): Context {
    const cs = this.getConstraints()
    return {
      admittedCount: this.totalAdmitted,
      remainingSlots: this.remaining(),
      constraintProgress: cs.map((c) => c.getProgress()),
      constraintShortfalls: cs.map((c) => c.getShortfall()),
    }
  }

  getProgress() {
    const feas = this.getFeasibilityRatios()
    const targetRate = this.bandit.getTargetAdmitRate?.() ?? this.bandit.targetAdmitRate
    const error = this.bandit.admitRateEma - targetRate

    if (this.totalAdmitted % 200 === 0) {
      const p = this.computeShadowPrices()
      const peek = Object.entries(p)
        .sort((a, b) => b[1] - a[1])
        .slice(0, CFG.UI.TOPK_ATTRS)
      this.log(`prices=${peek.map(([k, v]) => `${k}:${v.toFixed(2)}`).join(',')}`)

      const { attr: wlAttr, ratio: wlR, lag: wlLag } = this.worstLagInfo()
      this.log(`worstLag=${String(wlAttr)} r=${wlR.toFixed(2)} lag=${wlLag}`)
    }

    const base = {
      model: CFG.MODEL_VERSION,
      pretrained: this.pretrained,
      progress: pct(this.totalAdmitted / CFG.UI.PROGRESS_DENOM, 2) + '%',
      attributes: this.getConstraints().map((c) => ({
        total: `${c.admitted}/${c.minRequired} (${pct(c.admitted / c.minRequired, 1)}%)`,
        attribute: c.attribute,
        admitted: c.admitted,
        required: c.minRequired,
        satisfied: c.isSatisfied(),
        progress: c.getProgress(),
        shortfall: c.getShortfall(),
        frequency: c.frequency,
        empiricalFrequency: c.getEmpiricalFrequency(),
        scarcity: c.getScarcity(this.remaining()),
      })),
      lastLogs: this.logs.slice(-CFG.DEBUG.keepLastLogs),
      banditStats: this.bandit?.getStats(),
      remainingSlots: this.remaining(),
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      threshold: this.bandit.getLastThreshold(),
      lastRawValue: this.bandit.getLastRawValue(),
      admitRate: +this.bandit.admitRateEma.toFixed(3),
      feasibility: {
        maxRatio: feas.max,
        mostCritical: feas.mostCritical,
      },
      top: feas.ratios.slice(0, CFG.UI.TOPK_ATTRS).map((r) => ({
        attr: r.attr,
        ratio: r.ratio,
        lag: r.lag,
        freq: r.freq,
      })),
      controller: {
        targetRate: +targetRate.toFixed(3),
        error: +error.toFixed(3),
      },
    } as const

    if (!CFG.DEBUG.includeThresholdBlock) return base
    return { ...base, thresholdDebug: this.bandit.getThresholdDebug() }
  }

  private computeOvershootPenalty(): number {
    const rem = this.remaining()
    let cost = 0
    for (const c of this.getConstraints()) {
      const over = Math.max(0, c.admitted - c.minRequired)
      const extra = Math.max(0, over - CFG.SCORE.overshootSlack)
      if (!extra) continue

      // Optionally weight by how “valuable” that attr was at the end
      const w = CFG.SCORE.weightByScarcity ? 1 + c.getScarcity(rem) : 1

      const base = CFG.SCORE.overshootL1 * extra + CFG.SCORE.overshootL2 * extra * extra
      cost += w * base
    }
    return Math.round(cost)
  }

  private estimateExtraRejections(): number {
    const unmet = this.getConstraints().filter((c) => !c.isSatisfied())
    if (!unmet.length) return 0

    // d_i = shortfall / p_i (using empirical freq with a tiny floor)
    const eps = 1e-6
    const di = unmet.map((c) => c.getShortfall() / Math.max(eps, c.getEmpiricalFrequency()))

    const sum = di.reduce((a, b) => a + b, 0)
    const maxv = di.reduce((a, b) => Math.max(a, b), 0)

    // derive gamma from positive correlations among unmet attrs (fallback 0.3)
    let gamma = 0.3
    try {
      const keys = unmet.map((c) => String(c.attribute))
      const corr = (this.state.game.attributeStatistics?.correlations || {}) as Record<string, Record<string, number>>
      const pos: number[] = []
      for (let i = 0; i < keys.length; i++)
        for (let j = i + 1; j < keys.length; j++) {
          const v = corr[keys[i]]?.[keys[j]]
          if (typeof v === 'number' && v > 0) pos.push(v)
        }
      if (pos.length) {
        const avgPos = pos.reduce((a, b) => a + b, 0) / pos.length
        // map avgPos∈[0,1] to gamma∈[0.2,0.7]
        gamma = Math.min(0.7, Math.max(0.2, 0.2 + 0.6 * avgPos))
      }
    } catch {
      /* keep default gamma */
    }

    const drawsNeeded = (1 - gamma) * sum + gamma * maxv
    const extra = Math.max(0, Math.ceil(drawsNeeded - this.remaining()))
    return extra
  }

  getOutput(lastStatus: GameStatusCompleted | GameStatusFailed) {
    const extraPenalty = this.estimateExtraRejections()
    const overshootPenalty = this.computeOvershootPenalty()
    const finalScore = this.totalRejected + extraPenalty + overshootPenalty
    const gameData: GameState<any> = {
      ...this.state,
      status: lastStatus,
      timestamp: new Date().toISOString(),
      output: {
        completed: lastStatus.status,
        reason: lastStatus.status === 'failed' ? lastStatus.reason : undefined,
        finished: lastStatus.status === 'completed',
        finalScore,
        decisions: this.decisions,
        snapshot: this.bandit.toSnapshot(),
        ...this.getProgress(), // MODEL_VERSION is in progress.
      },
    }
    const summary = {
      model: CFG.MODEL_VERSION,
      status: lastStatus.status,
      gameId: this.state.game.gameId,
      finalScore,
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      admitRate: this.bandit.admitRateEma,
      threshold: this.bandit.getLastThreshold(),
      ...(CFG.DEBUG.includeThresholdBlock ? { thresholdDebug: this.bandit.getThresholdDebug() } : {}),
    }
    Disk.saveJsonFile('summary.json', summary).catch(() => {})
    Disk.saveGameState(gameData).catch(() => {})
    return gameData
  }

  /* --- storage --- */
  private async getPreviousGameResults(): Promise<GameResult[]> {
    try {
      const saved = await Disk.getJsonDataFromFiles<GameState<any>>()
      return saved
        .filter((g) => g.output?.decisions && g.output.decisions.length > 50)
        .filter((g) => g.output?.model === CFG.MODEL_VERSION)
        .map((g) => ({
          gameId: g.game.gameId,
          finalScore: g.output?.finalScore || 20000,
          timestamp: new Date(g.timestamp || 0),
          constraints: g.game?.constraints || [],
          decisions: g.output?.decisions || [],
          snapshot: g.output?.snapshot,
        }))
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 3)
    } catch {
      return []
    }
  }
}
