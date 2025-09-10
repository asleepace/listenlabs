import type { BerghainBouncer } from './berghain'
import type { GameState, GameStatusRunning, ScenarioAttributes } from './types'
import { Disk } from './utils/disk'

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

class Constraint<T> {
  public admitted = 0
  public rejected = 0

  constructor(
    public attribute: keyof T,
    public minRequired: number,
    public frequency: number,
    public config: Config
  ) {}

  update(person: Person<T>, admitted: boolean): void {
    if (!person[this.attribute]) return
    if (admitted) this.admitted++
    else this.rejected++
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
    const needed = this.getShortfall()
    const expectedAvailable = remainingSlots * this.frequency
    return needed / Math.max(0.1, expectedAvailable)
  }
}

class LinearBandit {
  // initialize values with random data
  // in LinearBandit
  // LinearBandit
  static randomInitialValues(n = 60) {
    // was 3.5 + 0.02*i
    return Array.from({ length: n }, (_, i) => 1.6 + 0.01 * i) // ~1.6..2.2
  }

  static fromSnapshot(s: BanditSnapshot) {
    const lb = new LinearBandit(s.featureDim, s.maxCapacity) // OR use constructorBare if you split it
    // overwrite core params
    lb.A = s.A
    lb.b = s.b
    lb.weights = s.weights
    // IMPORTANT: neutralize recentValues to avoid stale, inflated quantiles
    lb.recentValues = LinearBandit.randomInitialValues()
    // align controller to target to avoid big early shift
    lb.admitRateEma = lb.targetAdmitRate
    return lb
  }

  private priorWeights = [
    1.2,
    1.0,
    1.8,
    1.0, // indicators
    -0.8,
    -0.8,
    -1.4,
    -0.8, // progress (negative)
    -1.6,
    1.2, // capacity, scarcity
  ]

  // for debugging:
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

  private weights!: number[]
  private A!: number[][]
  private b!: number[]
  private featureDim: number
  private decisionCount: number = 0
  private lastThreshold = 9
  private lastRawValue = 9

  public totalAdmitted = 0
  public maxCapacity: number // Remove hardcoded value
  public maxFeasRatio = 0

  private recentValues: number[] = []
  private recentCap = 500

  public targetAdmitRate = 0.21
  private eta = 0.12 // was 0.2

  public admitRateEma = this.targetAdmitRate
  private emaBeta = 0.035 // a bit more responsive

  private rateErrI = 0
  private iBeta = 0.002 // very slow

  constructor(
    featureDimension: number,
    maxCapacity: number,
    previousData: DecisionRecord[] = []
  ) {
    this.featureDim = featureDimension
    this.maxCapacity = maxCapacity // Pass actual capacity
    this.reset()
    this.initializeFromHistory(previousData)
  }

  public warmStartFromHistory(decisions: DecisionRecord[]) {
    this.initializeFromHistory(decisions)
  }

  private reset() {
    const lambda = 0.1
    this.A = this.createIdentityMatrix(this.featureDim, lambda)
    // Seed b so that w = A^{-1} b starts at your priors
    this.b = this.priorWeights.map((w) => lambda * w)
    this.weights = [...this.priorWeights]

    this.recentValues = LinearBandit.randomInitialValues()
    console.log('Bandit reset with initial weights:', this.weights.slice(0, 6))
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
    // earlier base: 0.26/0.18/0.10; you had 0.20/0.14/0.09
    const base = used < 0.33 ? 0.24 : used < 0.66 ? 0.16 : 0.1
    const maxRatio = this.maxFeasRatio ?? 1.0
    const riskPull = Math.min(0.05, 0.03 * Math.max(0, maxRatio - 1)) // was up to 0.08
    return Math.max(0.07, base - riskPull) // min 7%
  }

  // inside LinearBandit
  getScoreSummary() {
    const s = [...this.recentValues].sort((a, b) => a - b)
    const pick = (p: number) =>
      s.length ? s[Math.floor(p * (s.length - 1))] : 0
    return { n: s.length, p10: pick(0.1), p50: pick(0.5), p90: pick(0.9) }
  }

  updateController(admitted: boolean) {
    this.admitRateEma =
      this.admitRateEma * (1 - this.emaBeta) + (admitted ? 1 : 0) * this.emaBeta
  }

  private createIdentityMatrix(size: number, lambda: number): number[][] {
    return Array(size)
      .fill(0)
      .map((_, i) =>
        Array(size)
          .fill(0)
          .map((_, j) => (i === j ? lambda : 0))
      )
  }

  private initializeFromHistory(decisions: DecisionRecord[]) {
    const validDecisions = decisions
      .filter((d) => d.features && d.features.length === this.featureDim)
      .slice(-100)

    console.log(
      `Learning from ${validDecisions.length} valid historical decisions`
    )

    validDecisions.forEach((decision, idx) => {
      const age = validDecisions.length - idx // 1=newest
      const w = Math.pow(0.9, age) // newer decisions count more
      this.updateModel(decision.features, decision.reward * w)
    })

    if (validDecisions.length > 0) {
      this.updateWeights()
    }
  }

  // Helper: robust stats
  private median(xs: number[]): number {
    if (xs.length === 0) return 0
    const a = [...xs].sort((x, y) => x - y)
    const m = Math.floor(a.length / 2)
    return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m])
  }
  private mad(xs: number[], med: number): number {
    if (xs.length === 0) return 1e-6
    const dev = xs.map((v) => Math.abs(v - med)).sort((x, y) => x - y)
    const m = Math.floor(dev.length / 2)
    const mad = dev.length % 2 ? dev[m] : 0.5 * (dev[m - 1] + dev[m])
    return Math.max(mad, 1e-6)
  }
  // Abramowitz & Stegun 26.2.23-ish rational approx for Φ^{-1}(p)
  private invNorm(p: number): number {
    // clamp away from 0/1
    const pp = Math.max(1e-9, Math.min(1 - 1e-9, p))
    const a1 = -39.69683028665376,
      a2 = 220.9460984245205,
      a3 = -275.9285104469687
    const a4 = 138.357751867269,
      a5 = -30.66479806614716,
      a6 = 2.506628277459239
    const b1 = -54.47609879822406,
      b2 = 161.5858368580409,
      b3 = -155.6989798598866
    const b4 = 66.80131188771972,
      b5 = -13.28068155288572
    const c1 = -0.007784894002430293,
      c2 = -0.3223964580411365
    const c3 = -2.400758277161838,
      c4 = -2.549732539343734
    const c5 = 4.374664141464968,
      c6 = 2.938163982698783
    const d1 = 0.007784695709041462,
      d2 = 0.3224671290700398
    const d3 = 2.445134137142996,
      d4 = 3.754408661907416

    let q, r
    if (pp < 0.02425) {
      // lower tail
      q = Math.sqrt(-2 * Math.log(pp))
      return (
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
      )
    } else if (pp > 1 - 0.02425) {
      // upper tail
      q = Math.sqrt(-2 * Math.log(1 - pp))
      return (
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
      )
    } else {
      // central
      q = pp - 0.5
      r = q * q
      return (
        ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
      )
    }
  }

  private zForTail(tail: number): number {
    // tail = fraction ABOVE the threshold. If targetAdmitRate = 0.21,
    // we want the 79th percentile => z ≈ +0.806
    return this.invNorm(tail)
  }

  /**
   * Scenario-agnostic adaptive threshold:
   * - Robustly estimates the desired quantile from recent scores (median + 1.4826*MAD*z)
   * - Adds small capacity bias
   * - Adds P-control admit-rate correction
   * - Subtracts constraint urgency
   * - Clips to robust data-dependent band (median ± 3*MAD*)
   */
  private calculateAdaptiveThreshold(constraintUrgency = 0): number {
    const arr = this.recentValues.length ? this.recentValues : [0]
    const med = this.median(arr)
    const mad = this.mad(arr, med)

    const warmup = this.decisionCount < 300
    const sigma = Math.max(1.4826 * mad, 0.35)

    const target = this.getTargetAdmitRate()
    const err = this.admitRateEma - target

    const tail = 1 - target
    const z = this.zForTail(tail)

    const quantileEstimate = med + z * sigma
    const used = this.totalAdmitted / this.maxCapacity
    const biasK = used < 0.5 ? 0.7 : 1.0 // gentler <50% capacity
    const capacityBias = biasK * used * sigma // scale by sigma

    const urgencyScale = Math.pow(Math.max(0, used - 0.3) / 0.6, 1.2)
    const urgencyAdj =
      Math.min(2.5, Math.max(0, constraintUrgency)) * urgencyScale

    this.rateErrI = (1 - this.iBeta) * this.rateErrI + this.iBeta * err
    const kPbase = 1.6,
      kIbase = 0.6
    const boost = err > 0.2 ? 1.35 : 1.0
    const kP = kPbase * boost
    const kI = kIbase * boost
    const rateAdjustment = kP * err + kI * this.rateErrI

    let thr = quantileEstimate + capacityBias + rateAdjustment - urgencyAdj

    if (err > 0) {
      const cap = warmup ? 0.3 : 0.75
      const floorBump = 0.25 + 1.25 * Math.min(cap, Math.max(0, err))
      thr = Math.max(thr, med + floorBump * sigma)
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

  getThresholdDebug() {
    return this.lastThrDbg
  }

  private getNoise() {
    const baseNoise = 0.2
    const decay = Math.min(1, this.decisionCount / 400)
    const noise = (Math.random() - 0.5) * (baseNoise * (1 - 0.5 * decay))
    // 0.2 → ~0.1 by 400 decisions
    return noise
  }

  selectAction(features: number[], constraintUrgency = 0) {
    this.decisionCount++
    this.updateWeights()
    const rawValue = this.predictValue(features)
    this.pushRaw(rawValue)

    const threshold = this.calculateAdaptiveThreshold(constraintUrgency)

    this.lastThreshold = threshold
    this.lastRawValue = rawValue
    const noise = this.getNoise()
    const action = rawValue + noise > threshold ? 'admit' : 'reject'
    return { action, value: rawValue } as const
  }

  private predictValue(features: number[]): number {
    let value = 0
    for (let i = 0; i < Math.min(features.length, this.weights.length); i++) {
      value += this.weights[i] * features[i]
    }
    return value
  }

  private updateWeights() {
    for (let i = 0; i < this.featureDim; i++) {
      if (this.A[i][i] > 1e-6) {
        this.weights[i] = this.b[i] / this.A[i][i]
      }
    }

    // Indices by your extractFeatures:
    // 0..3: has-attribute (free sign)
    // 4..7: progress (should be <= 0)
    // 8: capacity utilization (should be <= 0)
    // 9: creative scarcity (should be >= 0)
    const clamp = (v: number, lo: number, hi: number) =>
      Math.max(lo, Math.min(hi, v))

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = clamp(this.weights[i], -5, 5)
    }
    for (let i = 4; i <= 7; i++) this.weights[i] = Math.min(0, this.weights[i])
    this.weights[8] = Math.min(0, this.weights[8])
    this.weights[9] = Math.max(0, this.weights[9])
  }

  getLastRawValue() {
    return this.lastRawValue
  }

  getLastThreshold() {
    return this.lastThreshold
  }

  updateModel(features: number[], reward: number) {
    const r = Math.max(-50, Math.min(50, reward))
    for (let i = 0; i < features.length; i++) {
      this.A[i][i] += this.eta * (features[i] * features[i])
      this.b[i] += this.eta * (r * features[i])
    }
  }

  getStats() {
    return {
      weights: this.weights.slice(0, 6),
      decisionCount: this.decisionCount,
      capacityUsed: (this.totalAdmitted / this.maxCapacity).toFixed(3),
      threshold: this.lastThreshold.toFixed(1),
      avgWeight: (
        this.weights.reduce((a, b) => a + Math.abs(b), 0) / this.weights.length
      ).toFixed(2),
    }
  }

  private pushRaw(v: number) {
    // Soft clip using tanh so extreme values compress but still move stats
    const s = 8 // scale: scores map roughly into (-s, s)
    const soft = s * Math.tanh(v / s)
    this.recentValues.push(soft)
    if (this.recentValues.length > this.recentCap) this.recentValues.shift()
  }
}

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

  private computeConstraintUrgency(): number {
    let scores: number[] = []
    for (const c of this.getConstraints()) {
      if (!c.isSatisfied()) {
        const need = c.getShortfall()
        const expAvail = Math.max(1e-6, this.remainingSlots * c.frequency)
        const ratio = need / expAvail // >1 means unlikely to finish
        scores.push(ratio)
      }
    }
    if (scores.length === 0) return 0
    // Use a softplus-ish squashing so it grows >1 but doesn’t explode
    const maxRatio = Math.max(...scores)
    const urgency = Math.log1p(Math.E * maxRatio) // ~1 at ratio≈1, grows gently
    return Math.min(5.0, urgency)
  }

  private mostBehindConstraint(): { attr: keyof T; gap: number } | null {
    const used = this.totalAdmitted / this.config.MAX_CAPACITY
    let best: { attr: keyof T; gap: number } | null = null
    for (const c of this.getConstraints()) {
      const gap = used - c.getProgress() // >0 => behind pace
      if (!best || gap > best.gap) best = { attr: c.attribute, gap }
    }
    return best
  }

  // add this helper inside BanditBouncer<T>
  private getFeasibilityRatios() {
    const ratios = this.getConstraints()
      .filter((c) => !c.isSatisfied())
      .map((c) => ({
        attr: String(c.attribute),
        ratio:
          c.getShortfall() / Math.max(1e-6, this.remainingSlots * c.frequency),
        shortfall: c.getShortfall(),
        freq: c.frequency,
      }))
    const max = ratios.reduce((m, r) => Math.max(m, r.ratio), 0)
    const most = ratios.sort((a, b) => b.ratio - a.ratio)[0]?.attr
    return { max, mostCritical: most, ratios }
  }

  // Optional: expose this in getProgress() (small addition shown later)

  initializeConstraints() {
    for (const gameConstraint of this.state.game.constraints) {
      const attribute = gameConstraint.attribute as keyof T
      const constraint = new Constraint(
        attribute,
        gameConstraint.minCount,
        this.getFrequency(attribute),
        this.config
      )
      this.constraints.set(attribute, constraint)
    }
  }

  async initializeLearningData() {
    console.log('Initializing Simple Linear Bandit...')

    const previousGames = await this.getPreviousGameResults()
    const allDecisions = previousGames.flatMap((game) => game.decisions || [])
    const lastSnapshot = previousGames.at(0)?.snapshot

    const featureDim = 10
    // Pass actual capacity instead of hardcoded 10_000
    this.bandit = lastSnapshot
      ? LinearBandit.fromSnapshot(lastSnapshot)
      : new LinearBandit(featureDim, this.config.MAX_CAPACITY)

    if (!lastSnapshot) {
      this.bandit.warmStartFromHistory(allDecisions) // optional: decay weights inside as you do now
    }
  }

  get statistics(): Statistics<T> {
    return this.state.game.attributeStatistics as any
  }

  get remainingSlots(): number {
    return this.config.MAX_CAPACITY - this.totalAdmitted
  }

  admit({
    status,
    nextPerson,
  }: GameStatusRunning<ScenarioAttributes>): boolean {
    if (status !== 'running') return false
    if (this.remainingSlots <= 0) return false

    const person = nextPerson.attributes as Person<T>
    const features = this.extractFeatures(person)
    const used = this.totalAdmitted / this.config.MAX_CAPACITY

    // ---- Pace gate (later + bigger gap, and no bandit training on gate) ----
    const pace = this.mostBehindConstraint()
    if (pace && used > 0.2 && pace.gap > 0.15) {
      // was 0.12/0.12
      if (!person[pace.attr]) {
        const reward = -1.5
        this.decisions.push({
          context: this.buildContext(),
          action: 'reject',
          person,
          reward,
          features,
          banditValue: 0,
          heuristicValue: 0,
        })
        this.constraints.forEach((c) => c.update(person, false))
        this.totalRejected++
        this.logDecision(person, 'reject', reward, 0, features)
        this.bandit.updateController(false)
        // NOTE: no updateModel() on policy-gate rejections
        return false
      }
    }

    // ---- Feasibility gate (phase-dependent cutoff; no bandit training on gate) ----
    const feas = this.getFeasibilityRatios()
    const infeasibleCutoff = used < 0.33 ? 1.6 : used < 0.66 ? 1.3 : 1.0

    if (feas.max > infeasibleCutoff) {
      this.bandit.maxFeasRatio = feas.max // inform controller immediately
      const helpsAnyAtRisk = feas.ratios.some(
        (r) => r.ratio > infeasibleCutoff && person[r.attr as keyof T]
      )
      if (!helpsAnyAtRisk) {
        const reward = -2
        this.decisions.push({
          context: this.buildContext(),
          action: 'reject',
          person,
          reward,
          features,
          banditValue: 0, // didn't score; placeholder
          heuristicValue: 0,
        })
        this.constraints.forEach((c) => c.update(person, false))
        this.totalRejected++
        this.logDecision(person, 'reject', reward, 0, features)
        this.bandit.updateController(false)
        // NOTE: no updateModel() on policy-gate rejections
        return false
      }
    }

    // Single write (avoid duplicate assignments)
    this.bandit.maxFeasRatio = feas.max

    // ---- Bandit decision ----
    let urgency = this.computeConstraintUrgency()

    if (feas.mostCritical && person[feas.mostCritical as keyof T]) {
      urgency += 1.0 // small, targeted nudge
    }

    const { action, value } = this.bandit.selectAction(features, urgency)
    const shouldAdmit = action === 'admit'

    // Reward + learning
    const reward = this.calculateReward(person, shouldAdmit)
    this.bandit.updateModel(features, reward)

    // Record decision
    const context = this.buildContext()
    this.decisions.push({
      context,
      action,
      person,
      reward,
      features,
      banditValue: value,
      heuristicValue: 0,
    })

    // Update state
    this.constraints.forEach((c) => c.update(person, shouldAdmit))
    if (shouldAdmit) {
      this.totalAdmitted++
      this.bandit.totalAdmitted++
    } else {
      this.totalRejected++
    }

    this.logDecision(person, action, reward, value, features)
    this.bandit.updateController(shouldAdmit)

    return shouldAdmit
  }

  private extractFeatures(person: Person<T>): number[] {
    const constraints = this.getConstraints()
    const features: number[] = []

    // Indicators 0..3: 1 only if person has it AND the constraint isn't already satisfied
    constraints.forEach((c) => {
      features.push(person[c.attribute] && !c.isSatisfied() ? 1 : 0)
    })

    // Progress 4..7: keep as-is (weights for these are clamped ≤ 0, so higher progress reduces value)
    constraints.forEach((c) => {
      features.push(c.getProgress())
    })

    // Capacity 8
    features.push(Math.pow(this.totalAdmitted / this.config.MAX_CAPACITY, 0.8))

    // Scarcity 9: keep creative scarcity focus
    const creativeConstraint = constraints.find(
      (c) => String(c.attribute) === 'creative'
    )
    features.push(
      creativeConstraint
        ? 0.5 * Math.min(3, creativeConstraint.getScarcity(this.remainingSlots))
        : 0
    )

    return features
  }

  private calculateReward(person: Person<T>, admitted: boolean): number {
    const used = this.totalAdmitted / this.config.MAX_CAPACITY
    const { max } = this.getFeasibilityRatios()

    // --- Rejection penalty (slightly harsher when urgent/late) ---
    if (!admitted) {
      const base = used < 0.2 ? -0.25 : -0.5
      const urgency = Math.min(5, max) // reuse feasibility as urgency proxy
      const extra = -(0.15 + 0.1 * (used > 0.6 ? 1 : 0)) * urgency
      return Math.max(-2, base + extra)
    }

    // --- Pace controller knobs ---
    const paceMargin = 0.04 // deadband (~4% progress error)
    const kUnder = 2.2 // boost when under-pace
    const kOver = 1.6 // penalize when over-pace
    const clamp = (x: number, lo: number, hi: number) =>
      Math.max(lo, Math.min(hi, x))

    let reward = 0
    let usefulCount = 0

    this.constraints.forEach((constraint) => {
      if (!person[constraint.attribute]) return

      const prog = constraint.getProgress() // admitted/required ∈ [0,1]
      const paceGap = used - prog // >0 => behind pace
      let paceAdj: number
      if (Math.abs(paceGap) <= paceMargin) {
        paceAdj = 1
      } else if (paceGap > 0) {
        // under-pace → boost
        paceAdj = 1 + kUnder * (paceGap - paceMargin)
      } else {
        // over-pace → penalize
        paceAdj = 1 - kOver * (-paceGap - paceMargin)
      }
      paceAdj = clamp(paceAdj, 0.2, 2.0)

      if (!constraint.isSatisfied()) {
        usefulCount++

        // base payoff (your priors)
        // base payoff in calculateReward()
        const base =
          String(constraint.attribute) === 'creative'
            ? 3.4
            : String(constraint.attribute) === 'berlin_local'
            ? 2.0
            : 0.8

        // feasibility / scarcity (your existing signals)
        const need = constraint.getShortfall()
        const expAvail = Math.max(
          1e-6,
          this.remainingSlots * constraint.frequency
        )
        const feasRatio = need / expAvail // >1 => at-risk
        const feasBoost = Math.min(2.0, 0.75 + 0.5 * feasRatio)

        const scarcity = constraint.getScarcity(this.remainingSlots)
        const scarcityBoost = clamp(scarcity, 0.5, 2.0)

        // 👉 apply pace adjustment multiplicatively
        reward += base * feasBoost * scarcityBoost * paceAdj
      } else {
        // stronger overshoot penalty
        const overshoot =
          (constraint.admitted - constraint.minRequired) /
          Math.max(1, constraint.minRequired)

        const overshootPenalty = Math.min(
          2.2,
          1.2 + 1.0 * Math.max(0, overshoot)
        )
        reward -= overshootPenalty
      }
    })

    // If any constraint is at-risk and this admit helps none → hard slap
    if (max > 1.0 && usefulCount === 0) return -2

    // Small concave combo bonus for helping multiple unmet constraints
    if (usefulCount > 1) {
      const combo = Math.min(1.4, 1.0 + 0.15 * Math.log2(1 + usefulCount))
      reward *= combo
    }

    // Late-capacity guardrail
    if (used > 0.5 && usefulCount <= 1) reward -= 1.4 // was 1.2 and at 0.6

    // If truly useless, small negative
    if (usefulCount === 0) reward = Math.min(reward, -0.5)

    // Clamp to keep the bandit stable
    return Math.max(-2, Math.min(6, reward))
  }

  private logDecision(
    person: Person<T>,
    action: string,
    reward: number,
    value: number,
    features: number[]
  ) {
    const constraintStatus = this.getConstraints()
      .map(
        (c) => `${String(c.attribute)}:${(c.getProgress() * 100).toFixed(0)}%`
      )
      .join(' ')

    const usefulAttrs = this.getConstraints()
      .filter((c) => person[c.attribute] && !c.isSatisfied())
      .map((c) => String(c.attribute))

    this.logs.push(
      `[${this.remainingSlots.toString().padStart(3)}] ${JSON.stringify(
        person
      )} -> ${action}`
    )
    this.logs.push(`reward: ${reward.toFixed(1)}, bandit: ${value.toFixed(1)}`)
    this.logs.push(
      `useful: [${usefulAttrs.join(',')}], status: [${constraintStatus}]`
    )

    this.logs.push(
      `features: [${features.map((f) => f.toFixed(2)).join(', ')}]`
    )
  }

  private buildContext(): Context {
    const constraints = this.getConstraints()

    return {
      admittedCount: this.totalAdmitted,
      remainingSlots: this.remainingSlots,
      constraintProgress: constraints.map((c) => c.getProgress()),
      constraintShortfalls: constraints.map((c) => c.getShortfall()),
    }
  }

  getProgress() {
    const admitRate = this.bandit.admitRateEma?.toFixed?.(3)
    const thrDbg = this.bandit.getThresholdDebug?.() || {}
    const feas = this.getFeasibilityRatios()

    // NEW: expose dynamic target + error if available
    const targetRate =
      this.bandit.getTargetAdmitRate?.() ?? this.bandit.targetAdmitRate
    const rateError =
      this.bandit.admitRateEma !== undefined
        ? this.bandit.admitRateEma - targetRate
        : null
    return {
      attributes: this.getConstraints().map((c) => ({
        attribute: c.attribute,
        admitted: c.admitted,
        required: c.minRequired,
        satisfied: c.isSatisfied(),
        progress: c.getProgress(),
        shortfall: c.getShortfall(),
        frequency: c.frequency,
        scarcity: c.getScarcity(this.remainingSlots),
      })),
      lastLogs: this.logs.slice(-3),
      banditStats: this.bandit?.getStats(),
      remainingSlots: this.remainingSlots,
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      threshold: this.bandit.getLastThreshold(),
      lastRawValue: this.bandit.getLastRawValue(),
      admitRate,
      scoreDist: this.bandit.getScoreSummary(),
      thresholdDebug: thrDbg,
      // 👇 NEW FIELDS
      feasibility: {
        maxRatio: feas.max,
        mostCritical: feas.mostCritical,
      },
      feasibilityRatios: feas.ratios.slice(0, 4),
      controller: {
        targetRate: targetRate?.toFixed?.(3),
        error: rateError?.toFixed?.(3),
      },
    }
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

    // save summary of game
    const summary = {
      gameId: this.state.game.gameId,
      finalScore,
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      admitRate: this.bandit.admitRateEma,
      threshold: this.bandit.getLastThreshold(),
      thresholdDebug: this.bandit.getThresholdDebug(),
    }
    Disk.saveJsonFile('summary.json', summary).catch(() => {})

    Disk.saveGameState(gameData).catch((e) => {
      console.warn('[bandit-bouncer] failed to save game file:', e)
    })

    return gameData
  }

  // Helper methods
  getConstraints(): Constraint<T>[] {
    return Array.from(this.constraints.values())
  }

  getFrequency(attribute: keyof T): number {
    return this.statistics.relativeFrequencies[attribute] as number
  }

  private async getPreviousGameResults(): Promise<GameResult[]> {
    try {
      const savedGameData = await Disk.getJsonDataFromFiles<GameState<any>>()
      return savedGameData
        .filter(
          (game) => game.output?.decisions && game.output.decisions.length > 50
        )
        .map((game) => ({
          gameId: game.game.gameId,
          finalScore: game.output?.finalScore || 20000,
          timestamp: new Date(game.timestamp || 0),
          constraints: game.game?.constraints || [],
          decisions: game.output?.decisions || [],
          snapshot: game.output?.snapshot,
        }))
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 3)
    } catch (e) {
      console.warn('Failed to load previous games:', e)
      return []
    }
  }
}
