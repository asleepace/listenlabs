import type { BerghainBouncer } from './berghain'
import type {
  GameConstraints,
  GameState,
  GameStatusRunning,
  ScenarioAttributes,
} from './types'
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
  static fromSnapshot(s: BanditSnapshot) {
    const lb = new LinearBandit(s.featureDim, s.maxCapacity) // OR use constructorBare if you split it
    // overwrite core params
    lb.A = s.A
    lb.b = s.b
    lb.weights = s.weights
    // IMPORTANT: neutralize recentValues to avoid stale, inflated quantiles
    lb.recentValues = Array(Math.min(60, lb.recentCap)).fill(2.0)
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

  // for deugging:
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

  private recentValues: number[] = []
  private recentCap = 500

  private targetAdmitRate = 0.21
  private eta = 0.2 // tune 0.05–0.5

  private admitRateEma = this.targetAdmitRate
  private emaBeta = 0.02 // slow + stable

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

    this.recentValues = Array(60).fill(2.0)
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
  // Map target admit rate -> z for a Normal (robust approx)
  private zForTail(p: number): number {
    // p is the tail prob: e.g. 0.30 -> z≈0.524 (70th percentile)
    const table: Record<string, number> = {
      '0.50': 0.0,
      '0.40': 0.253,
      '0.35': 0.385,
      '0.30': 0.524,
      '0.25': 0.674,
      '0.20': 0.842,
      '0.15': 1.036,
      '0.10': 1.282,
      '0.07': 1.476,
      '0.05': 1.645,
      '0.03': 1.881,
      '0.02': 2.054,
    }
    // snap to nearest key
    const keys = Object.keys(table)
      .map(parseFloat)
      .sort((a, b) => a - b)
    const nearest = keys.reduce(
      (best, k) => (Math.abs(k - p) < Math.abs(best - p) ? k : best),
      keys[0]
    )
    return table[nearest.toFixed(2)] ?? 0.524
  }

  /**
   * Scenario-agnostic adaptive threshold:
   * - Robustly estimates the desired quantile from recent scores (median + 1.4826*MAD*z)
   * - Adds small capacity bias
   * - Adds P-control admit-rate correction
   * - Subtracts constraint urgency
   * - Clips to robust data-dependent band (median ± 3*MAD*)
   */
  private calculateAdaptiveThreshold(constraintUrgency: number = 0): number {
    const arr = this.recentValues.length ? this.recentValues : [0]
    const med = this.median(arr)
    const mad = this.mad(arr, med)
    const sigma = 1.4826 * mad

    const tail = 1 - this.targetAdmitRate
    const z = this.zForTail(tail)

    const quantileEstimate = med + z * sigma
    const used = this.totalAdmitted / this.maxCapacity
    const capacityBias = 0.6 * used
    const k = 0.8
    const rateAdjustment = k * (this.admitRateEma - this.targetAdmitRate)
    const urgencyAdj = Math.min(5.0, Math.max(0, constraintUrgency))

    let thr = quantileEstimate + capacityBias + rateAdjustment - urgencyAdj

    const lo = med - 3 * sigma
    const hi = med + 3 * sigma
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
    }
    return final
  }

  getThresholdDebug() {
    return this.lastThrDbg
  }

  private calculateUrgencyBonus(): number {
    // This would need access to constraints - for now return 0
    // In practice, you'd pass constraint info or calculate here
    return 0
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
    const clipped = Math.max(-4, Math.min(4, v))
    this.recentValues.push(clipped)
    if (this.recentValues.length > this.recentCap) this.recentValues.shift()
  }

  private percentile(p: number): number {
    if (this.recentValues.length < 50) return this.calculateAdaptiveThreshold()
    const arr = [...this.recentValues].sort((a, b) => a - b)
    const n = arr.length
    const lo = Math.floor(0.05 * n),
      hi = Math.ceil(0.95 * n)
    const trimmed = arr.slice(lo, Math.max(lo + 1, hi)) // ensure non-empty
    const idx = Math.min(
      trimmed.length - 1,
      Math.max(0, Math.floor(p * (trimmed.length - 1)))
    )
    return trimmed[idx]
  }
}

export class BanditBouncer<T> implements BerghainBouncer {
  public constraints = new Map<keyof T, Constraint<T>>()
  public totalAdmitted = 0
  public totalRejected = 0
  private bandit!: LinearBandit
  private decisions: DecisionRecord[] = []

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

    const urgency = this.computeConstraintUrgency()
    const { action, value } = this.bandit.selectAction(features, urgency)
    const shouldAdmit = action === 'admit'

    // Simple, bounded reward calculation
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
      heuristicValue: 0, // not used in this version
    })

    // Update state
    this.constraints.forEach((constraint) => {
      constraint.update(person, shouldAdmit)
    })

    if (shouldAdmit) {
      this.totalAdmitted++
      this.bandit.totalAdmitted++
    } else {
      this.totalRejected++
    }

    // Enhanced logging
    this.logDecision(person, action, reward, value, features)
    this.bandit.updateController(shouldAdmit)

    return shouldAdmit
  }

  private extractFeatures(person: Person<T>): number[] {
    const constraints = this.getConstraints()
    const features: number[] = []

    // Features 0-3: Has each constraint attribute
    constraints.forEach((constraint) => {
      features.push(person[constraint.attribute] ? 1 : 0)
    })

    // Features 4-7: Progress on each constraint
    constraints.forEach((constraint) => {
      features.push(constraint.getProgress())
    })

    // Feature 8: Capacity utilization
    // features.push(this.totalAdmitted / this.config.MAX_CAPACITY)
    features.push(Math.pow(this.totalAdmitted / this.config.MAX_CAPACITY, 0.8))

    // Feature 9: Creative scarcity (special focus on bottleneck)
    const creativeConstraint = constraints.find(
      (c) => String(c.attribute) === 'creative'
    )
    if (creativeConstraint) {
      features.push(
        creativeConstraint
          ? 0.5 *
              Math.min(3, creativeConstraint.getScarcity(this.remainingSlots))
          : 0
      )
    } else {
      features.push(0)
    }

    return features
  }

  private calculateReward(person: Person<T>, admitted: boolean): number {
    const used = this.totalAdmitted / this.config.MAX_CAPACITY
    if (!admitted) return used < 0.2 ? -0.25 : -0.5

    let reward = 0
    let usefulCount = 0

    this.constraints.forEach((constraint) => {
      if (person[constraint.attribute]) {
        if (!constraint.isSatisfied()) {
          // 👍 positive reward for helping satisfy this constraint
          usefulCount++
          let baseReward =
            String(constraint.attribute) === 'creative'
              ? 4.0 // boosted
              : String(constraint.attribute) === 'berlin_local'
              ? 2.0 // boosted
              : 0.8
          const scarcityMultiplier = constraint.getScarcity(this.remainingSlots)
          reward +=
            baseReward * Math.max(0.5, Math.min(2.0, scarcityMultiplier))
        } else {
          // 👇 penalty if constraint already satisfied
          reward -= 1.0
        }
      }
    })

    // late capacity penalty if admits aren’t very useful
    if (used > 0.6 && usefulCount <= 1) reward -= 1.2

    // if truly useless, small negative
    reward = usefulCount === 0 ? -0.5 : reward

    return Math.max(-2, Math.min(6, reward))
  }

  private logDecision(
    person: Person<T>,
    action: string,
    reward: number,
    value: number,
    features: number[]
  ) {
    const shouldLog =
      this.remainingSlots < 100 ||
      Math.abs(reward) > 15 ||
      this.totalAdmitted + this.totalRejected < 50 ||
      (this.totalAdmitted + this.totalRejected) % 100 === 0

    if (shouldLog) {
      const constraintStatus = this.getConstraints()
        .map(
          (c) => `${String(c.attribute)}:${(c.getProgress() * 100).toFixed(0)}%`
        )
        .join(' ')

      const usefulAttrs = this.getConstraints()
        .filter((c) => person[c.attribute] && !c.isSatisfied())
        .map((c) => String(c.attribute))

      console.log(
        `[${this.remainingSlots.toString().padStart(3)}] ${JSON.stringify(
          person
        )} -> ${action}`
      )
      console.log(
        `    Reward: ${reward.toFixed(1)}, Bandit Value: ${value.toFixed(1)}`
      )
      console.log(
        `    Useful: [${usefulAttrs.join(',')}], Status: [${constraintStatus}]`
      )

      if (this.totalAdmitted + this.totalRejected < 20) {
        console.log(
          `    Features: [${features.map((f) => f.toFixed(2)).join(', ')}]`
        )
        console.log(`    Bandit stats:`, this.bandit.getStats())
      }
    }
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
    const admitRate = (this.bandit as any).admitRateEma?.toFixed?.(3)
    const thrDbg = (this.bandit as any).getThresholdDebug?.() || {}
    const rv = (this as any).bandit ? (this as any).bandit : null
    const recent =
      rv && (rv as any).recentValues ? (rv as any).recentValues : []
    const sorted = [...recent].sort((a: number, b: number) => a - b)
    const pick = (p: number) =>
      sorted.length ? sorted[Math.floor(p * (sorted.length - 1))] : 0

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
      remainingSlots: this.remainingSlots,
      banditStats: this.bandit?.getStats(),
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      threshold: this.bandit.getLastThreshold(),
      lastRawValue: this.bandit.getLastRawValue(),
      admitRate,
      // NEW: score distribution + threshold components
      scoreDist: {
        n: sorted.length,
        p10: pick(0.1),
        p50: pick(0.5),
        p90: pick(0.9),
      },
      thresholdDebug: thrDbg, // med, mad, z, capacityBias, rateAdjustment, urgencyAdj, lo/hi, final
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
      admitRate: (this.bandit as any).admitRateEma,
      threshold: this.bandit.getLastThreshold(),
      thresholdDebug: (this.bandit as any).getThresholdDebug?.(),
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
