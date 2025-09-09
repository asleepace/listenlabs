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
  constraintsSatisfied: boolean[]
}

interface Context {
  // Current state features
  admittedCount: number
  remainingSlots: number
  progressRatios: number[] // progress toward each constraint [0,1]
  urgencyScores: number[] // how urgent each constraint is
  personAttributes: boolean[] // which attributes this person has
  scarcityScore: number // how scarce good candidates are becoming
}

interface Config {
  TOTAL_PEOPLE: number
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
    return this.admitted / this.minRequired
  }

  getUrgency(remainingSlots: number): number {
    const needed = Math.max(0, this.minRequired - this.admitted)
    return remainingSlots > 0 ? needed / remainingSlots : 1.0
  }

  isSatisfied(): boolean {
    return this.admitted >= this.minRequired
  }
}

/**
 * Linear Contextual Bandit using UCB (Upper Confidence Bound)
 * Each action (admit/reject) has a linear model predicting reward
 */
class ContextualBandit {
  private admitWeights: number[]
  private rejectWeights: number[]
  private admitCovMatrix: number[][]
  private rejectCovMatrix: number[][]
  private alpha: number = 1.0 // exploration parameter
  private contextDim: number

  constructor(contextDimension: number, previousData: DecisionRecord[] = []) {
    this.contextDim = contextDimension

    // Initialize with small random weights
    this.admitWeights = Array(contextDimension)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.01)
    this.rejectWeights = Array(contextDimension)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.01)

    // Initialize covariance matrices (A^-1 in LinUCB)
    this.admitCovMatrix = this.createIdentityMatrix(contextDimension)
    this.rejectCovMatrix = this.createIdentityMatrix(contextDimension)

    // Learn from previous data if available
    this.initializeFromHistory(previousData)
  }

  private createIdentityMatrix(size: number): number[][] {
    return Array(size)
      .fill(0)
      .map((_, i) =>
        Array(size)
          .fill(0)
          .map((_, j) => (i === j ? 1 : 0))
      )
  }

  private initializeFromHistory(decisions: DecisionRecord[]) {
    // Replay historical decisions to warm-start the model
    decisions.forEach((decision) => {
      const context = this.contextToVector(decision.context)
      this.updateModel(context, decision.action, decision.reward)
    })
  }

  public contextToVector(context: Context): number[] {
    // Convert context to feature vector
    return [
      context.admittedCount / 1000, // normalized admitted count
      context.remainingSlots / 1000, // normalized remaining slots
      ...context.progressRatios, // constraint progress [0,1]
      ...context.urgencyScores, // constraint urgencies
      ...context.personAttributes.map((x) => (x ? 1 : 0)), // person features
      context.scarcityScore, // scarcity metric
      // Add interaction features
      Math.min(...context.urgencyScores), // most urgent constraint
      Math.max(...context.progressRatios), // best constraint progress
    ]
  }

  selectAction(context: Context): 'admit' | 'reject' {
    const contextVec = this.contextToVector(context)

    // Calculate UCB scores for both actions
    const admitScore = this.calculateUCB(
      contextVec,
      this.admitWeights,
      this.admitCovMatrix
    )
    const rejectScore = this.calculateUCB(
      contextVec,
      this.rejectWeights,
      this.rejectCovMatrix
    )

    return admitScore > rejectScore ? 'admit' : 'reject'
  }

  private calculateUCB(
    context: number[],
    weights: number[],
    covMatrix: number[][]
  ): number {
    // Calculate predicted reward: θ^T * x
    const predicted = this.dotProduct(weights, context)

    // Calculate confidence radius: α * sqrt(x^T * A^-1 * x)
    const confidence =
      this.alpha * Math.sqrt(this.quadraticForm(context, covMatrix))

    return predicted + confidence
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i]!, 0)
  }

  private quadraticForm(x: number[], matrix: number[][]): number {
    // Calculate x^T * A^-1 * x
    let result = 0
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x.length; j++) {
        result += x[i]! * matrix[i]![j]! * x[j]!
      }
    }
    return Math.max(0, result) // ensure non-negative
  }

  updateModel(context: number[], action: 'admit' | 'reject', reward: number) {
    const weights = action === 'admit' ? this.admitWeights : this.rejectWeights
    const covMatrix =
      action === 'admit' ? this.admitCovMatrix : this.rejectCovMatrix

    // Update A = A + x * x^T (covariance matrix)
    for (let i = 0; i < context.length; i++) {
      for (let j = 0; j < context.length; j++) {
        covMatrix[i]![j] += context[i]! * context[j]!
      }
    }

    // Update b = b + r * x (reward-weighted context sum)
    for (let i = 0; i < weights.length; i++) {
      weights[i] += reward * context[i]!
    }

    // In practice, you'd solve A * θ = b for θ (weights)
    // For simplicity, we're doing incremental updates
    // In production, use proper matrix inversion or online learning
  }

  getStats() {
    return {
      admitWeights: [...this.admitWeights],
      rejectWeights: [...this.rejectWeights],
      alpha: this.alpha,
    }
  }
}

export class BanditBouncer<T> implements BergainBouncer {
  public constraints = new Map<keyof T, Constraint<T>>()
  public totalAdmitted = 0
  public totalRejected = 0
  private bandit!: ContextualBandit
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
    console.log('Initializing Contextual Bandit...')

    // Load previous game data
    const previousGames = await this.getPreviousGameResults()
    const allDecisions = previousGames.flatMap((game) => game.decisions || [])

    // Calculate context dimension
    const numConstraints = this.constraints.size
    const contextDim = 2 + numConstraints * 2 + numConstraints + 1 + 2 // see contextToVector

    this.bandit = new ContextualBandit(contextDim, allDecisions)

    console.log(
      `Bandit initialized with ${allDecisions.length} historical decisions`
    )
    console.log('Bandit stats:', this.bandit.getStats())
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

    // Build context
    const context = this.buildContext(person)

    // Get bandit decision
    const action = this.bandit.selectAction(context)
    const shouldAdmit = action === 'admit'

    // Calculate immediate reward
    const reward = this.calculateReward(person, shouldAdmit, context)

    // Update bandit model
    const contextVec = this.bandit['contextToVector'](context) // access private method
    this.bandit.updateModel(contextVec, action, reward)

    // Record decision
    this.decisions.push({
      context,
      action,
      person,
      reward,
      constraintsSatisfied: this.getConstraints().map((c) => c.isSatisfied()),
    })

    // Update constraint states
    this.constraints.forEach((constraint) => {
      constraint.update(person, shouldAdmit)
    })

    if (shouldAdmit) {
      this.totalAdmitted++
    } else {
      this.totalRejected++
    }

    console.log(
      `Person: ${JSON.stringify(
        person
      )}, Action: ${action}, Reward: ${reward.toFixed(3)}`
    )

    return shouldAdmit
  }

  private buildContext(person: Person<T>): Context {
    const constraints = this.getConstraints()

    return {
      admittedCount: this.totalAdmitted,
      remainingSlots: this.remainingSlots,
      progressRatios: constraints.map((c) => c.getProgress()),
      urgencyScores: constraints.map((c) => c.getUrgency(this.remainingSlots)),
      personAttributes: constraints.map((c) => !!person[c.attribute]),
      scarcityScore: this.calculateScarcityScore(),
    }
  }

  private calculateReward(
    person: Person<T>,
    admitted: boolean,
    context: Context
  ): number {
    let reward = 0

    if (admitted) {
      // Positive reward for progress toward unsatisfied constraints
      this.constraints.forEach((constraint, attr) => {
        if (person[attr] && !constraint.isSatisfied()) {
          const urgency = constraint.getUrgency(this.remainingSlots)
          reward += urgency * 2 // base reward for needed attribute
        }
      })

      // Penalty for admitting when nearly full and constraints satisfied
      const allSatisfied = this.getConstraints().every((c) => c.isSatisfied())
      if (allSatisfied && this.remainingSlots < 100) {
        reward -= 0.5 // small penalty for unnecessary admissions
      }
    } else {
      // Small positive reward for rejecting when no useful attributes
      const hasUsefulAttribute = Array.from(this.constraints.entries()).some(
        ([attr, constraint]) => person[attr] && !constraint.isSatisfied()
      )

      if (!hasUsefulAttribute) {
        reward += 0.1 // small reward for good rejection
      } else {
        // Penalty for rejecting useful people
        reward -= Math.max(...context.urgencyScores) // penalty proportional to max urgency
      }
    }

    return reward
  }

  private calculateScarcityScore(): number {
    const constraints = this.getConstraints()
    const totalExpectedPeople = constraints
      .filter((c) => !c.isSatisfied())
      .reduce((sum, c) => {
        const needed = c.minRequired - c.admitted
        return sum + needed / c.frequency
      }, 0)

    return totalExpectedPeople / Math.max(1, this.remainingSlots)
  }

  getProgress() {
    return {
      attributes: this.getConstraints().map((constraint) => ({
        attribute: constraint.attribute,
        admitted: constraint.admitted,
        required: constraint.minRequired,
        satisfied: constraint.isSatisfied(),
        progress: constraint.getProgress(),
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
      return savedGameData.map((game) => ({
        gameId: game.game.gameId,
        finalScore: game.output?.finalScore || 20000,
        timestamp: new Date(game.timestamp || 0),
        constraints: game.game?.constraints || [],
        decisions: game.output?.decisions || [],
      }))
    } catch (e) {
      console.warn('Failed to load previous games:', e)
      return []
    }
  }
}
