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

/**
 * Simple Linear Bandit with proper reward scaling
 */
class LinearBandit {
  private weights!: number[]
  private A!: number[][]
  private b!: number[]
  private featureDim: number
  private decisionCount: number = 0

  constructor(featureDimension: number, previousData: DecisionRecord[] = []) {
    this.featureDim = featureDimension
    this.reset()
    this.initializeFromHistory(previousData)
  }

  private reset() {
    this.A = this.createIdentityMatrix(this.featureDim, 0.1)
    this.b = Array(this.featureDim).fill(0)

    // Initialize with reasonable weights instead of zeros
    this.weights = [
      7, // techno_lover indicator
      7, // well_connected indicator
      10, // creative indicator (more valuable)
      8, // berlin_local indicator
      -5, // techno_lover progress (less valuable when satisfied)
      -5, // well_connected progress
      -10, // creative progress (penalty for being satisfied)
      -8, // berlin_local progress
      -3, // capacity utilization (penalty for being full)
      5, // creative scarcity (bonus for high scarcity)
    ]

    console.log('Bandit reset with initial weights:', this.weights.slice(0, 6))
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
    // Filter valid decisions but be more lenient with reward scale
    // (historical data might have different reward scales)
    const validDecisions = decisions
      .filter((d) => d.features && d.features.length === this.featureDim)
      .slice(-100) // Just take recent decisions, ignore reward scale

    console.log(
      `Learning from ${validDecisions.length} valid historical decisions`
    )

    validDecisions.forEach((decision) => {
      // Normalize historical rewards to current scale
      let normalizedReward = decision.reward
      if (Math.abs(decision.reward) > 100) {
        // Scale down large historical rewards
        normalizedReward = decision.reward * 0.2 // rough scaling factor
        normalizedReward = Math.max(-50, Math.min(50, normalizedReward))
      }

      this.updateModel(decision.features, normalizedReward)
    })

    if (validDecisions.length > 0) {
      this.updateWeights()
    }
  }

  selectAction(features: number[]): {
    action: 'admit' | 'reject'
    value: number
  } {
    this.decisionCount++

    // Skip the complex UCB - just predict raw value
    this.updateWeights()
    const rawValue = this.predictValue(features)

    // Simple threshold on raw prediction
    const threshold = 15 + this.decisionCount / 100 // gradually increase threshold
    const action = rawValue > threshold ? 'admit' : 'reject'

    return { action, value: rawValue }
  }

  private predictValue(features: number[]): number {
    let value = 0
    for (let i = 0; i < Math.min(features.length, this.weights.length); i++) {
      value += this.weights[i] * features[i]
    }
    return value
  }

  private calculateConfidence(features: number[]): number {
    let confidence = 0
    for (let i = 0; i < features.length; i++) {
      if (this.A[i][i] > 1e-6) {
        confidence += (features[i] * features[i]) / this.A[i][i]
      }
    }
    return Math.sqrt(Math.max(0, confidence))
  }

  private updateWeights() {
    for (let i = 0; i < this.featureDim; i++) {
      if (this.A[i][i] > 1e-6) {
        this.weights[i] = this.b[i] / this.A[i][i]
        this.weights[i] = Math.max(-25, Math.min(25, this.weights[i])) // increased bounds
      }
    }
  }

  updateModel(features: number[], reward: number) {
    // Clip extreme rewards
    const clippedReward = Math.max(-50, Math.min(50, reward))

    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        this.A[i][j] += features[i] * features[j]
      }
      this.b[i] += clippedReward * features[i]
    }
  }

  getStats() {
    return {
      weights: this.weights.slice(0, 6),
      decisionCount: this.decisionCount,
      avgWeight: (
        this.weights.reduce((a, b) => a + Math.abs(b), 0) / this.weights.length
      ).toFixed(2),
    }
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
    this.bandit = new LinearBandit(featureDim, allDecisions)

    console.log(
      `Bandit initialized: ${allDecisions.length} decisions, ${featureDim}D features`
    )
    console.log('Initial stats:', this.bandit.getStats())
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
    const { action, value } = this.bandit.selectAction(features)
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
    } else {
      this.totalRejected++
    }

    // Enhanced logging
    this.logDecision(person, action, reward, value, features)

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
    features.push(this.totalAdmitted / this.config.MAX_CAPACITY)

    // Feature 9: Creative scarcity (special focus on bottleneck)
    const creativeConstraint = constraints.find(
      (c) => String(c.attribute) === 'creative'
    )
    if (creativeConstraint) {
      features.push(
        Math.min(3, creativeConstraint.getScarcity(this.remainingSlots))
      )
    } else {
      features.push(0)
    }

    return features
  }

  private calculateReward(person: Person<T>, admitted: boolean): number {
    if (!admitted) {
      return 1 // Small positive for rejection
    }

    // Simple reward: sum of rarity values for useful attributes
    let reward = 0
    let usefulCount = 0

    this.constraints.forEach((constraint) => {
      if (person[constraint.attribute] && !constraint.isSatisfied()) {
        usefulCount++

        // Fixed values to prevent explosion
        if (String(constraint.attribute) === 'creative') {
          reward += 20 // creative is very valuable
        } else if (String(constraint.attribute) === 'berlin_local') {
          reward += 8 // berlin_local is moderately valuable
        } else {
          reward += 3 // others are less valuable
        }
      }
    })

    // Small combo bonus
    if (usefulCount > 1) {
      reward += usefulCount * 2
    }

    // Penalty for no useful attributes
    if (usefulCount === 0) {
      reward = -5
    }

    return reward
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
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      remainingSlots: this.remainingSlots,
      banditStats: this.bandit?.getStats(),
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
