import type { BergainBouncer } from './berghain'
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
  private weights!: number[]
  private A!: number[][]
  private b!: number[]
  private featureDim: number
  private decisionCount: number = 0
  private lastThreshold = 9
  private lastRawValue = 9

  public totalAdmitted = 0
  public maxCapacity: number // Remove hardcoded value
  private admitRateEma = 0.5
  private emaBeta = 0.02 // slow + stable

  private recentValues: number[] = []
  private recentCap = 500
  private targetAdmitRate = 0.1 // tune: 0.25–0.45 are typical

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

  private reset() {
    this.A = this.createIdentityMatrix(this.featureDim, 0.1)
    this.b = Array(this.featureDim).fill(0)

    // Initialize with reasonable weights instead of zeros
    this.weights = [
      1.2, // techno_lover
      1.0, // well_connected
      1.8, // creative
      1.0, // berlin_local
      -0.8, // techno_lover progress
      -0.8, // well_connected progress
      -1.4, // creative progress
      -0.8, // berlin_local progress
      -1.6, // capacity utilization
      1.2, // creative scarcity
    ]

    this.recentValues = Array(60).fill(0) // > 50 so percentile is used
    console.log('Bandit reset with initial weights:', this.weights.slice(0, 6))
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

    validDecisions.forEach((decision) => {
      let normalizedReward = decision.reward
      if (Math.abs(decision.reward) > 100) {
        normalizedReward = decision.reward * 0.1
        normalizedReward = Math.max(-50, Math.min(50, normalizedReward))
      }

      this.updateModel(decision.features, normalizedReward)
    })

    if (validDecisions.length > 0) {
      this.updateWeights()
    }
  }

  private calculateAdaptiveThreshold(): number {
    const capacityUsed = this.totalAdmitted / this.maxCapacity
    const baseThreshold = 0 // better matches your reward scale

    let threshold = baseThreshold
    if (capacityUsed < 0.3) {
      threshold = baseThreshold + capacityUsed * 2.0
    } else if (capacityUsed < 0.7) {
      threshold = baseThreshold + 0.6 + (capacityUsed - 0.3) * 3.0
    } else {
      threshold = baseThreshold + 1.8 + (capacityUsed - 0.7) * 5.0
    }

    const urgencyBonus = this.calculateUrgencyBonus()
    threshold -= urgencyBonus

    // Keep a sensible floor/ceiling so we never saturate
    return Math.max(0.0, Math.min(4.0, threshold)) // floor at 0
  }

  private calculateUrgencyBonus(): number {
    // This would need access to constraints - for now return 0
    // In practice, you'd pass constraint info or calculate here
    return 0
  }

  selectAction(
    features: number[],
    constraintUrgency: number = 0
  ): {
    action: 'admit' | 'reject'
    value: number
  } {
    this.decisionCount++

    this.updateWeights()

    const rawValue = this.predictValue(features)
    this.pushRaw(rawValue)

    // quantile: admit rate ~ targetAdmitRate
    const base = this.percentile(1 - this.targetAdmitRate)
    const urgencyAdj = Math.min(2.0, Math.max(0, constraintUrgency))

    // Nudge threshold up if we're admitting too much, down if too little
    const targetRate = this.targetAdmitRate // use the same target
    const k = 0.8
    const rateAdjustment = k * (this.admitRateEma - targetRate)

    // urgency pulls threshold DOWN a bit when constraints are at risk
    const threshold = base + rateAdjustment - urgencyAdj

    this.lastThreshold = threshold
    this.lastRawValue = rawValue

    const noise = (Math.random() - 0.5) * 0.2

    const action = rawValue + noise > threshold ? 'admit' : 'reject'

    return { action, value: rawValue }
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
      this.A[i][i] += features[i] * features[i] // only diagonal
      this.b[i] += r * features[i]
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
    this.recentValues.push(v)
    if (this.recentValues.length > this.recentCap) this.recentValues.shift()
  }

  private percentile(p: number): number {
    if (this.recentValues.length < 50) return this.calculateAdaptiveThreshold()
    const arr = [...this.recentValues].sort((a, b) => a - b)
    const idx = Math.min(
      arr.length - 1,
      Math.max(0, Math.floor(p * (arr.length - 1)))
    )
    return arr[idx]
  }
}

export class BanditBouncer<T> implements BergainBouncer {
  public constraints = new Map<keyof T, Constraint<T>>()
  public totalAdmitted = 0
  public totalRejected = 0
  private bandit!: LinearBandit
  private decisions: DecisionRecord[] = []

  constructor(public state: GameState, public config: Config) {
    this.initializeConstraints()
  }

  private computeConstraintUrgency(): number {
    // Aggregate scarcity of unmet constraints; softly capped
    let urgency = 0
    for (const c of this.getConstraints()) {
      if (!c.isSatisfied()) {
        urgency += Math.min(3, c.getScarcity(this.remainingSlots))
      }
    }
    // Normalize a bit so it behaves like a small threshold shift
    return Math.min(2.0, urgency * 0.25)
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

    const featureDim = 10
    // Pass actual capacity instead of hardcoded 10_000
    this.bandit = new LinearBandit(
      featureDim,
      this.config.MAX_CAPACITY,
      allDecisions
    )
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
    if (!admitted) return -0.5 // <-- NEGATIVE, not positive

    let reward = 0
    let usefulCount = 0
    const capacityUsed = this.totalAdmitted / this.config.MAX_CAPACITY

    this.constraints.forEach((constraint) => {
      if (person[constraint.attribute] && !constraint.isSatisfied()) {
        usefulCount++
        let baseReward =
          String(constraint.attribute) === 'creative'
            ? 2.5
            : String(constraint.attribute) === 'berlin_local'
            ? 1.2
            : 0.8
        const scarcityMultiplier = constraint.getScarcity(this.remainingSlots)
        reward += baseReward * Math.max(0.5, Math.min(2.0, scarcityMultiplier))
      }
    })

    if (capacityUsed > 0.6 && usefulCount <= 1) reward -= 1.2
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
    return {
      attributes: this.getConstraints().map((constraint) => ({
        attribute: constraint.attribute,
        admitted: constraint.admitted,
        required: constraint.minRequired,
        satisfied: constraint.isSatisfied(),
        progress: constraint.getProgress(),
        shortfall: constraint.getShortfall(),
        frequency: constraint.frequency,
        scarcity: constraint.getScarcity(this.remainingSlots),
      })),
      remainingSlots: this.remainingSlots,
      banditStats: this.bandit?.getStats(),
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      threshold: this.bandit.getLastThreshold(),
      lastRawValue: this.bandit.getLastRawValue(),
      admitRate,
    }
  }

  getOutput() {
    const finalScore = this.totalRejected

    const gameData: GameState<any> = {
      ...this.state,
      timestamp: new Date().toISOString(),
      output: {
        ...this.getProgress(),
        finished: true,
        finalScore,
        decisions: this.decisions,
      },
    }

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
        }))
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 3)
    } catch (e) {
      console.warn('Failed to load previous games:', e)
      return []
    }
  }
}
