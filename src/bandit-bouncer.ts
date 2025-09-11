/* eslint-disable max-classes-per-file */

import type { BerghainBouncer } from './berghain'
import type { GameState, GameStatusRunning, ScenarioAttributes } from './types'
import { Disk } from './utils/disk'

/* =========================
   ✅ TUNING CONFIG (all knobs)
   ========================= */
const CFG = {
  // Capacity / schedule
  TARGET_RATE_BASE: { early: 0.24, mid: 0.16, late: 0.1 },
  TARGET_RATE_MIN: 0.08, // min admit rate
  TARGET_RISK_PULL: { slope: 0.025, max: 0.04 },

  // Linear bandit
  BANDIT: {
    featureDim: 10,
    eta: 0.12,
    emaBeta: 0.035,
    iBeta: 0.002,
    priorWeights: [1.2, 1.0, 1.8, 1.0, -0.8, -0.8, -1.4, -0.8, -1.6, 1.2],
    weightClamp: [-5, 5] as const,
    progressFloor: -0.15, // indices 4..7 ≤ this
    capacityFloor: -0.1, // index 8 ≤ this
    scarcityFloor: 0, // index 9 ≥ this
    initRecentValues: { n: 60, start: 1.6, step: 0.01 },
    recentCap: 500,
    earlyNoiseBoostDecisions: 200, // more exploration early
  },

  // Thresholding (robust quantile + controller)
  THRESH: {
    sigmaFloor: 0.4, // slightly higher to avoid too-tight thresholds early
    warmupDecisions: 300,
    warmupErrCap: 0.25, // max err used in floor bump during warmup
    floorBumpBase: 0.2,
    floorBumpSlope: 1.25,
    capacityBiasScale: { early: 0.5, late: 0.9 }, // * used * sigma
    urgencyMax: 2.5,
    ctrlGains: { kP: 1.6, kI: 0.6, boostEdge: 0.2, boostFactor: 1.35 },
  },

  // Gates (policy, no learning updates)
  GATES: {
    pace: { minUsed: 0.35, gap: 0.16, penalty: -1.5 }, // later & gentler
    infeasibleCutoff: {
      earlyOffUntil: 0.35, // no infeasible gating before 35% used
      mid: 2.2, // 0.35–0.66
      late1: 1.6, // 0.66–0.85
      late2: 1.2, // >0.85
    },
    reserve: { minUsed: 0.75, buffer: 0.95, penalty: -2 }, // late-only
    top1MustHave: { minUsed: 0.7, minRatio: 1.6, dominance: 1.3, penalty: -2 },
    lateFeas: { minUsed: 0.82, maxRatio: 1.2, helpAtRiskMin: 2, penalty: -2 },
  },

  // Reward calculation
  REWARD: {
    baseCreative: 3.4,
    baseLocal: 2.0,
    baseDefault: 0.8,
    feasBoost: { slope: 0.65, base: 0.75, max: 2.2 },
    scarcityClamp: [0.5, 2.0] as const,
    mcBoost: { slope: 0.5, max: 2.2 }, // only for most-critical attr
    pace: { margin: 0.04, kUnder: 2.2, kOver: 1.6, clamp: [0.2, 2.0] as const },
    lateGuardrail: { minUsed: 0.5, penaltyIfUsefulLE1: 1.4 },
    rejectPenalty: {
      early: -0.15, // friendlier early
      late: -0.5,
      step: 0.12, // a bit softer
      extraLate: 0.08,
      min: -1.0, // cap early sting
    },
    clamp: [-2, 6] as const,
  },

  FINISH: {
    enableAtUsed: 0.9, // start finishing once ≥90% of capacity is used
    maxShortfall: 3, // if a constraint is missing ≤3 people
  },
  // Fill-to-capacity helper when all constraints are satisfied
  FILL: {
    enableAtUsed: 0.9, // once ≥90% capacity used and constraints met -> fill seats
    learn: false, // don't train the bandit on pure fill admits (avoid overshoot penalties)
  },

  // Learning warmup
  WARMUP: {
    usedMax: 0.15,
    minEma: 0.02,
    negClampAfter: -0.3, // clamp negative reward updates after warmup
  },

  // Optional epsilon-admit very early (helps break stalemates)
  EXPLORE: { epsAdmit: 0.08, epsUntilUsed: 0.05 },

  // Debug / logging
  DEBUG: {
    keepLastLogs: 2, // less noise
    includeThresholdBlock: false,
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
}

interface Statistics<T> {
  correlations: Record<keyof T, Record<keyof T, number>>
  relativeFrequencies: Record<keyof T, number>
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

  getEmpiricalFrequency(): number {
    const priorTrue = 2,
      priorTotal = 8
    const num = this.seenTrue + priorTrue
    const den = this.seenTotal + priorTotal
    return Math.max(1e-6, num / Math.max(1, den))
  }

  getProgress(): number {
    return Math.min(1.0, this.admitted / this.minRequired)
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

  static fromSnapshot(s: BanditSnapshot) {
    const lb = new LinearBandit(s.featureDim, s.maxCapacity)
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
  private decisionCount = 0
  private lastThreshold = 9
  private lastRawValue = 9

  public totalAdmitted = 0
  public maxCapacity: number
  public maxFeasRatio = 0

  private recentValues: number[] = []
  private recentCap = CFG.BANDIT.recentCap

  public targetAdmitRate = 0.21
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

  constructor(featureDimension: number, maxCapacity: number, previousData: DecisionRecord[] = []) {
    this.featureDim = featureDimension
    this.maxCapacity = maxCapacity
    this.reset()
    this.initializeFromHistory(previousData)
  }

  /** allow warm start without exposing a private method */
  public warmStartFromHistory(decisions: DecisionRecord[]) {
    this.initializeFromHistory(decisions)
  }

  private reset() {
    const lambda = 0.1
    this.A = Array.from({ length: this.featureDim }, (_, i) =>
      Array.from({ length: this.featureDim }, (_, j) => (i === j ? lambda : 0))
    )
    this.b = CFG.BANDIT.priorWeights.map((w) => lambda * w)
    this.weights = [...CFG.BANDIT.priorWeights]
    this.recentValues = LinearBandit.randomInitialValues()
    // Initialize controller EMA to current target to avoid initial overshoot
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
    }
  }

  public getTargetAdmitRate(): number {
    const used = this.totalAdmitted / this.maxCapacity
    const base =
      used < 0.33 ? CFG.TARGET_RATE_BASE.early : used < 0.66 ? CFG.TARGET_RATE_BASE.mid : CFG.TARGET_RATE_BASE.late
    const riskPull = Math.min(
      CFG.TARGET_RISK_PULL.max,
      CFG.TARGET_RISK_PULL.slope * Math.max(0, (this.maxFeasRatio ?? 1) - 1)
    )
    return Math.max(CFG.TARGET_RATE_MIN, base - riskPull)
  }

  getScoreSummary() {
    const s = [...this.recentValues].sort((a, b) => a - b)
    const pick = (p: number) => (s.length ? s[Math.floor(p * (s.length - 1))] : 0)
    return { n: s.length, p10: pick(0.1), p50: pick(0.5), p90: pick(0.9) }
  }

  updateController(admitted: boolean) {
    this.admitRateEma = this.admitRateEma * (1 - this.emaBeta) + (admitted ? 1 : 0) * this.emaBeta
  }

  private initializeFromHistory(decisions: DecisionRecord[]) {
    const valid = decisions.filter((d) => d.features?.length === this.featureDim).slice(-100)
    valid.forEach((d, idx) => {
      const age = valid.length - idx
      const w = Math.pow(0.9, age)
      this.updateModel(d.features, d.reward * w)
    })
    if (valid.length) this.updateWeights()
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

  // Inverse normal CDF approximation (A&S 26.2.23): FIXED tail formulas
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
      return -(
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
      )
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

  private calculateAdaptiveThreshold(constraintUrgency = 0): number {
    const arr = this.recentValues.length ? this.recentValues : [0]
    const med = this.median(arr)
    const mad = this.mad(arr, med)
    const sigma = Math.max(1.4826 * mad, CFG.THRESH.sigmaFloor)

    const target = this.getTargetAdmitRate()
    const err = this.admitRateEma - target
    const tail = 1 - target
    const z = this.zForTail(tail)
    const quantileEstimate = med + z * sigma

    const used = this.totalAdmitted / this.maxCapacity
    const capK = used < 0.5 ? CFG.THRESH.capacityBiasScale.early : CFG.THRESH.capacityBiasScale.late
    const capacityBias = capK * used * sigma

    const urgencyScale = Math.pow(Math.max(0, used - 0.3) / 0.6, 1.2)
    const urgencyAdj = Math.min(CFG.THRESH.urgencyMax, Math.max(0, constraintUrgency)) * urgencyScale

    // PI controller
    const { kP, kI, boostEdge, boostFactor } = CFG.THRESH.ctrlGains
    this.rateErrI = (1 - this.iBeta) * this.rateErrI + this.iBeta * err
    const boost = err > boostEdge ? boostFactor : 1
    const rateAdjustment = kP * boost * err + kI * boost * this.rateErrI

    let thr = quantileEstimate + capacityBias + rateAdjustment - urgencyAdj

    // floor vs err (more rejections if over target)
    if (err > 0) {
      const warm = this.decisionCount < CFG.THRESH.warmupDecisions
      const cap = warm ? CFG.THRESH.warmupErrCap : 0.6
      const bump = CFG.THRESH.floorBumpBase + CFG.THRESH.floorBumpSlope * Math.min(cap, Math.max(0, err))
      thr = Math.max(thr, med + bump * sigma)
    }

    const lo = med - 3 * sigma,
      hi = med + 3 * sigma
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
      urgencyAdj,
      lo,
      hi,
      final,
      target,
      err,
      tail,
    }
    return final
  }

  selectAction(features: number[], constraintUrgency = 0) {
    this.decisionCount++
    this.updateWeights()
    const rawValue = this.predictValue(features)
    this.pushRaw(rawValue)
    const threshold = this.calculateAdaptiveThreshold(constraintUrgency)
    this.lastThreshold = threshold
    this.lastRawValue = rawValue

    // slightly larger noise early to encourage exploration
    const base = 0.2
    const decay = Math.min(1, this.decisionCount / 400)
    const boost = this.decisionCount < CFG.BANDIT.earlyNoiseBoostDecisions ? 1.5 : 1.0
    const noise = (Math.random() - 0.5) * (base * boost * (1 - 0.5 * decay))

    const action = rawValue + noise > threshold ? 'admit' : 'reject'
    return { action, value: rawValue } as const
  }

  private predictValue(features: number[]): number {
    let v = 0
    for (let i = 0; i < Math.min(features.length, this.weights.length); i++) v += this.weights[i] * features[i]
    return v
  }

  private updateWeights() {
    for (let i = 0; i < this.featureDim; i++) if (this.A[i][i] > 1e-6) this.weights[i] = this.b[i] / this.A[i][i]
    const clamp = (x: number, [lo, hi]: readonly [number, number]) => Math.max(lo, Math.min(hi, x))

    // global clamp
    for (let i = 0; i < this.weights.length; i++) this.weights[i] = clamp(this.weights[i], CFG.BANDIT.weightClamp)

    // keep progress negative, cap/capacity/scarcity signs
    for (let i = 4; i <= 7; i++) this.weights[i] = Math.min(CFG.BANDIT.progressFloor, this.weights[i])
    this.weights[8] = Math.min(CFG.BANDIT.capacityFloor, this.weights[8])
    this.weights[9] = Math.max(CFG.BANDIT.scarcityFloor, this.weights[9])

    // optional: during warmup, prevent indicators [0..3] from going negative
    const used = this.totalAdmitted / Math.max(1, this.maxCapacity)
    const warmup = used < CFG.WARMUP.usedMax || this.admitRateEma < CFG.WARMUP.minEma
    if (warmup) {
      for (let i = 0; i < 4; i++) this.weights[i] = Math.max(0, this.weights[i])
    }
  }

  updateModel(features: number[], reward: number) {
    const r = Math.max(-50, Math.min(50, reward))
    for (let i = 0; i < features.length; i++) {
      this.A[i][i] += this.eta * (features[i] * features[i])
      this.b[i] += this.eta * (r * features[i])
    }
  }

  getStats() {
    const absAvg = this.weights.reduce((a, b) => a + Math.abs(b), 0) / this.weights.length
    return {
      weights: this.weights.slice(0, 6),
      decisionCount: this.decisionCount,
      capacityUsed: +(this.totalAdmitted / this.maxCapacity).toFixed(3),
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
    const s = 8
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
    const ratios = this.getConstraints()
      .filter((c) => !c.isSatisfied())
      .map((c) => {
        const f = c.getEmpiricalFrequency() || c.frequency
        return {
          attr: String(c.attribute),
          ratio: c.getShortfall() / Math.max(1e-6, this.remaining() * f),
          shortfall: c.getShortfall(),
          freq: f,
        }
      })
    const max = ratios.reduce((m, r) => Math.max(m, r.ratio), 0)
    const most = ratios.slice().sort((a, b) => b.ratio - a.ratio)[0]?.attr
    return { max, mostCritical: most, ratios }
  }

  private computeConstraintUrgency(): number {
    const scores: number[] = []
    for (const c of this.getConstraints()) {
      if (!c.isSatisfied()) {
        const f = c.getEmpiricalFrequency() || c.frequency
        const ratio = c.getShortfall() / Math.max(1e-6, this.remaining() * f)
        scores.push(ratio)
      }
    }
    if (!scores.length) return 0
    const maxRatio = Math.max(...scores)
    return Math.min(CFG.THRESH.urgencyMax, Math.log1p(Math.E * maxRatio))
  }

  private mostBehindConstraint(): { attr: keyof T; gap: number } | null {
    const used = this.usedFrac()
    let best: { attr: keyof T; gap: number } | null = null
    for (const c of this.getConstraints()) {
      const gap = used - c.getProgress()
      if (!best || gap > best.gap) best = { attr: c.attribute, gap }
    }
    return best
  }

  private computeReserves() {
    const list = this.getConstraints()
      .filter((c) => !c.isSatisfied())
      .map((c) => {
        const f = c.getEmpiricalFrequency() || c.frequency
        return {
          attr: c.attribute,
          drawsNeeded: c.getShortfall() / Math.max(1e-6, f),
        }
      })
    const total = list.reduce((a, r) => a + r.drawsNeeded, 0)
    return { list, total }
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
  initializeConstraints() {
    for (const gc of this.state.game.constraints) {
      const attribute = gc.attribute as keyof T
      this.constraints.set(attribute, new Constraint(attribute, gc.minCount, this.getFrequency(attribute), this.config))
    }
  }

  async initializeLearningData() {
    const prev = await this.getPreviousGameResults()
    const all = prev.flatMap((g) => g.decisions || [])
    const snap = prev.at(0)?.snapshot
    const dim = CFG.BANDIT.featureDim
    this.bandit = snap ? LinearBandit.fromSnapshot(snap) : new LinearBandit(dim, this.config.MAX_CAPACITY)
    if (!snap) this.bandit.warmStartFromHistory(all)
  }

  get statistics(): Statistics<T> {
    return this.state.game.attributeStatistics as any
  }
  getFrequency(attribute: keyof T): number {
    return this.statistics.relativeFrequencies[attribute] as number
  }

  /* --- features --- */
  private extractFeatures(person: Person<T>): number[] {
    const cs = this.getConstraints()
    const feats: number[] = []
    // indicators (only if unmet)
    cs.forEach((c) => feats.push(person[c.attribute] && !c.isSatisfied() ? 1 : 0))
    // progress (clamped negative weight)
    cs.forEach((c) => feats.push(Math.max(0, 1 - c.getProgress())))
    // capacity
    feats.push(Math.pow(this.usedFrac(), 0.8))
    // creative scarcity feature
    const creative = cs.find((c) => String(c.attribute) === 'creative')
    feats.push(creative ? 0.5 * Math.min(3, creative.getScarcity(this.remaining())) : 0)
    return feats
  }

  /* --- policy gate (no learning updates) --- */
  private policyGate(person: Person<T>, features: number[]): { blocked: boolean } {
    const used = this.usedFrac()
    const pace = this.mostBehindConstraint()
    if (pace && used > CFG.GATES.pace.minUsed && pace.gap > CFG.GATES.pace.gap) {
      if (!person[pace.attr]) {
        this.recordDecision(person, 'reject', CFG.GATES.pace.penalty, 0, features)
        this.getConstraints().forEach((c) => c.update(person, false))
        this.totalRejected++
        this.bandit.updateController(false)
        return { blocked: true }
      }
    }

    const feas = this.getFeasibilityRatios()

    // reserve lock (late only)
    const reserves = this.computeReserves()
    if (used > CFG.GATES.reserve.minUsed && reserves.total > this.remaining() * CFG.GATES.reserve.buffer) {
      const hitsReserve = reserves.list.some((r) => r.drawsNeeded > 0.5 && person[r.attr as keyof T])
      if (!hitsReserve) {
        this.recordDecision(person, 'reject', CFG.GATES.reserve.penalty, 0, features)
        this.getConstraints().forEach((c) => c.update(person, false))
        this.totalRejected++
        this.bandit.updateController(false)
        return { blocked: true }
      }
    }

    // top1 must-have
    const sorted = [...feas.ratios].sort((a, b) => b.ratio - a.ratio)
    const top1 = sorted[0],
      top2 = sorted[1]
    if (
      used > CFG.GATES.top1MustHave.minUsed &&
      top1 &&
      top1.ratio >= CFG.GATES.top1MustHave.minRatio &&
      (!top2 || top1.ratio >= CFG.GATES.top1MustHave.dominance * top2.ratio)
    ) {
      if (!person[top1.attr as keyof T]) {
        this.recordDecision(person, 'reject', CFG.GATES.top1MustHave.penalty, 0, features)
        this.getConstraints().forEach((c) => c.update(person, false))
        this.totalRejected++
        this.bandit.updateController(false)
        return { blocked: true }
      }
    }

    // late feasibility
    if (used > CFG.GATES.lateFeas.minUsed && feas.max > CFG.GATES.lateFeas.maxRatio && feas.mostCritical) {
      const hasCritical = person[feas.mostCritical as keyof T] ? 1 : 0
      const helpsAtRisk = feas.ratios.filter((r) => r.ratio > 1.0 && person[r.attr as keyof T]).length
      if (!hasCritical && helpsAtRisk < CFG.GATES.lateFeas.helpAtRiskMin) {
        this.recordDecision(person, 'reject', CFG.GATES.lateFeas.penalty, 0, features)
        this.getConstraints().forEach((c) => c.update(person, false))
        this.totalRejected++
        this.bandit.updateController(false)
        return { blocked: true }
      }
    }

    // infeasible cutoff by phase (off before earlyOffUntil)
    let cutoff = Infinity
    if (used >= CFG.GATES.infeasibleCutoff.earlyOffUntil && used < 0.66) cutoff = CFG.GATES.infeasibleCutoff.mid
    else if (used >= 0.66 && used < 0.85) cutoff = CFG.GATES.infeasibleCutoff.late1
    else if (used >= 0.85) cutoff = CFG.GATES.infeasibleCutoff.late2

    if (feas.max > cutoff) {
      this.bandit.maxFeasRatio = feas.max
      const helpsAny = feas.ratios.some((r) => r.ratio > cutoff && person[r.attr as keyof T])
      if (!helpsAny) {
        this.recordDecision(person, 'reject', -2, 0, features)
        this.getConstraints().forEach((c) => c.update(person, false))
        this.totalRejected++
        this.bandit.updateController(false)
        return { blocked: true }
      }
    }

    // let bandit decide
    this.bandit.maxFeasRatio = feas.max
    return { blocked: false }
  }

  /* --- reward --- */
  private calculateReward(person: Person<T>, admitted: boolean): number {
    const used = this.usedFrac()

    const feasNow = this.getFeasibilityRatios()
    const mostCriticalAttr = feasNow.mostCritical
    const max = feasNow.max

    const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))
    const R = CFG.REWARD

    if (!admitted) {
      const base = used < 0.2 ? R.rejectPenalty.early : R.rejectPenalty.late
      const extra = -(R.rejectPenalty.step + (used > 0.6 ? R.rejectPenalty.extraLate : 0)) * Math.min(5, max)
      return Math.max(R.rejectPenalty.min, base + extra)
    }

    let reward = 0
    let useful = 0

    for (const c of this.getConstraints()) {
      if (!person[c.attribute]) continue
      const prog = c.getProgress()
      const paceGap = used - prog
      const absGap = Math.abs(paceGap)
      let paceAdj = 1
      if (absGap > R.pace.margin) {
        paceAdj =
          paceGap > 0 ? 1 + R.pace.kUnder * (paceGap - R.pace.margin) : 1 - R.pace.kOver * (absGap - R.pace.margin)
      }
      paceAdj = clamp(paceAdj, R.pace.clamp[0], R.pace.clamp[1])

      if (!c.isSatisfied()) {
        useful++
        const base =
          String(c.attribute) === 'creative'
            ? R.baseCreative
            : String(c.attribute) === 'berlin_local'
            ? R.baseLocal
            : R.baseDefault

        const f = c.getEmpiricalFrequency() || c.frequency
        const feasRatio = c.getShortfall() / Math.max(1e-6, this.remaining() * f)
        const feasBoost = Math.min(R.feasBoost.max, R.feasBoost.base + R.feasBoost.slope * feasRatio)
        const scarcityBoost = clamp(c.getScarcity(this.remaining()), R.scarcityClamp[0], R.scarcityClamp[1])
        const mcBoost =
          String(c.attribute) === String(mostCriticalAttr)
            ? Math.min(R.mcBoost.max, 1.0 + R.mcBoost.slope * Math.max(0, feasRatio - 1))
            : 1.0

        reward += base * feasBoost * scarcityBoost * paceAdj * mcBoost
      } else {
        const overshoot = (c.admitted - c.minRequired) / Math.max(1, c.minRequired)
        reward -= Math.min(2.2, 1.2 + Math.max(0, overshoot))
      }
    }

    if (max > 1.0 && useful === 0) return -2
    if (useful > 1) reward *= Math.min(1.4, 1.0 + 0.15 * Math.log2(1 + useful))
    if (used > CFG.REWARD.lateGuardrail.minUsed && useful <= 1) reward -= CFG.REWARD.lateGuardrail.penaltyIfUsefulLE1
    if (useful === 0) reward = Math.min(reward, -0.5)

    return clamp(reward, CFG.REWARD.clamp[0], CFG.REWARD.clamp[1])
  }

  /* --- bouncer API --- */
  admit({ status, nextPerson }: GameStatusRunning<ScenarioAttributes>): boolean {
    if (status !== 'running') return false
    if (this.remaining() <= 0) return false

    const person = nextPerson.attributes as Person<T>
    const features = this.extractFeatures(person)
    const used = this.usedFrac()

    // --- fill-to-capacity: if ALL constraints are satisfied late, just admit ---
    const allSatisfied = this.getConstraints().every((c) => c.isSatisfied())
    if (allSatisfied && used >= CFG.FILL.enableAtUsed) {
      // Optional: skip model training to avoid overshoot penalties on satisfied attrs
      if (CFG.FILL.learn) {
        const reward = 0 // neutral learning if you want
        this.bandit.updateModel(features, reward)
      }
      // Update state
      this.getConstraints().forEach((c) => c.update(person, true))
      this.totalAdmitted++
      this.bandit.totalAdmitted++
      // Minimal log/decision record
      this.recordDecision(
        person,
        'admit',
        0, // neutral reward for record
        this.bandit.getLastRawValue?.() ?? 0,
        features
      )
      this.bandit.updateController(true)
      return true
    }

    // handle end-game logic
    if (used >= CFG.FINISH.enableAtUsed) {
      const outstanding = this.getConstraints().filter((c) => !c.isSatisfied())
      if (outstanding.length) {
        // find the smallest remaining shortfall among unmet constraints
        let minShort = Infinity
        let minAttr: keyof T | null = null
        for (const c of outstanding) {
          const s = c.getShortfall()
          if (s > 0 && s < minShort) {
            minShort = s
            minAttr = c.attribute
          }
        }

        // If we're down to a tiny shortfall and this person helps it -> admit
        if (minAttr && minShort <= CFG.FINISH.maxShortfall && person[minAttr]) {
          const reward = this.calculateReward(person, true)
          this.bandit.updateModel(features, reward)
          this.getConstraints().forEach((c) => c.update(person, true))
          this.totalAdmitted++
          this.bandit.totalAdmitted++
          this.recordDecision(person, 'admit', reward, this.bandit.getLastRawValue?.() ?? 0, features)
          this.bandit.updateController(true)
          return true
        }
      }
    }

    // Optional epsilon-admit very early to break stalemates
    if (used < CFG.EXPLORE.epsUntilUsed) {
      const helpsAny = this.getConstraints().some((c) => person[c.attribute] && !c.isSatisfied())
      if (helpsAny && Math.random() < CFG.EXPLORE.epsAdmit) {
        const reward = this.calculateReward(person, true)
        this.bandit.updateModel(features, reward)
        this.getConstraints().forEach((c) => c.update(person, true))
        this.totalAdmitted++
        this.bandit.totalAdmitted++
        this.recordDecision(person, 'admit', reward, 0, features)
        this.bandit.updateController(true)
        return true
      }
    }

    // policy gate (does not train the bandit)
    const gate = this.policyGate(person, features)
    if (gate.blocked) return false

    // bandit decision
    const urgency = this.computeConstraintUrgency()
    const { action, value } = this.bandit.selectAction(features, urgency)
    const shouldAdmit = action === 'admit'

    // learning with warmup rules
    const inWarmup = used < CFG.WARMUP.usedMax || this.bandit.admitRateEma < CFG.WARMUP.minEma
    const reward = this.calculateReward(person, shouldAdmit)
    if (shouldAdmit) {
      this.bandit.updateModel(features, reward)
    } else if (!inWarmup) {
      this.bandit.updateModel(features, Math.max(CFG.WARMUP.negClampAfter, reward))
    } // else: no learning from early rejections

    // state
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

    const base = {
      attributes: this.getConstraints().map((c) => ({
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
        top: feas.ratios.slice(0, 3),
      },
      controller: {
        targetRate: +targetRate.toFixed(3),
        error: +error.toFixed(3),
      },
    } as const

    if (!CFG.DEBUG.includeThresholdBlock) return base
    return { ...base, thresholdDebug: this.bandit.getThresholdDebug() }
  }

  getOutput() {
    const finalScore = this.totalRejected
    const gameData: GameState<any> = {
      ...this.state,
      timestamp: new Date().toISOString(),
      output: {
        finished: true,
        finalScore,
        decisions: this.decisions,
        snapshot: this.bandit.toSnapshot(),
        ...this.getProgress(),
      },
    }
    const summary = {
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
