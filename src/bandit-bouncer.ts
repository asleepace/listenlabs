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
  admittedCount: number
  remainingSlots: number
  progressRatios: number[]
  urgencyScores: number[]
  personAttributes: boolean[]
  scarcityScore: number
  capacityUtilization: number
  worstConstraintProgress: number
  hasRareAttribute: boolean
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

  getUrgency(remainingSlots: number): number {
    if (this.isSatisfied()) return 0
    const needed = this.minRequired - this.admitted
    return remainingSlots > 0 ? needed / remainingSlots : 1000
  }

  isSatisfied(): boolean {
    return this.admitted >= this.minRequired
  }

  isRare(): boolean {
    return this.frequency <= 0.1
  }
}

/**
 * Linear Contextual Bandit using epsilon-greedy with UCB exploration
 */
class ContextualBandit {
  private admitWeights: number[]
  private rejectWeights: number[]
  private admitA: number[][]
  private rejectA: number[][]
  private admitB: number[]
  private rejectB: number[]
  private epsilon: number = 0.1 // exploration rate
  private contextDim: number
  private decisionCount: number = 0

  constructor(contextDimension: number, previousData: DecisionRecord[] = []) {
    this.contextDim = contextDimension

    // Initialize A matrices with small regularization
    this.admitA = this.createRegularizedIdentity(contextDimension, 0.1)
    this.rejectA = this.createRegularizedIdentity(contextDimension, 0.1)

    // Initialize b vectors
    this.admitB = Array(contextDimension).fill(0)
    this.rejectB = Array(contextDimension).fill(0)

    // Initialize weights with small random values
    this.admitWeights = Array(contextDimension)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.01)
    this.rejectWeights = Array(contextDimension)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.01)

    // Learn from previous data
    this.initializeFromHistory(previousData)
  }

  private createRegularizedIdentity(size: number, lambda: number): number[][] {
    return Array(size)
      .fill(0)
      .map((_, i) =>
        Array(size)
          .fill(0)
          .map((_, j) => (i === j ? lambda : 0))
      )
  }

  private initializeFromHistory(decisions: DecisionRecord[]) {
    console.log(
      `Initializing bandit with ${decisions.length} historical decisions`
    )
    decisions.forEach((decision) => {
      const context = this.contextToVector(decision.context)
      this.updateModel(context, decision.action, decision.reward)
    })

    if (decisions.length > 0) {
      this.updateWeights()
    }
  }

  contextToVector(context: Context): number[] {
    return [
      // Basic state (4 features)
      context.admittedCount / 1000,
      context.remainingSlots / 1000,
      context.capacityUtilization,
      Math.min(context.scarcityScore / 1000, 3),

      // Constraint progress (dynamic size)
      ...context.progressRatios,

      // Constraint urgencies (dynamic size, capped)
      ...context.urgencyScores.map((u) => Math.min(u / 100, 10)),

      // Person attributes (dynamic size)
      ...context.personAttributes.map((x) => (x ? 1 : 0)),

      // Derived features (4 features)
      context.worstConstraintProgress,
      context.hasRareAttribute ? 1 : 0,
      context.remainingSlots < 50 ? 1 : 0, // emergency mode
      context.progressRatios.filter((p) => p < 0.5).length /
        Math.max(1, context.progressRatios.length),
    ]
  }

  selectAction(context: Context): 'admit' | 'reject' {
    this.decisionCount++

    // Decay epsilon over time
    const currentEpsilon = this.epsilon * Math.exp(-this.decisionCount / 1000)

    // Epsilon-greedy exploration
    if (Math.random() < currentEpsilon) {
      return Math.random() < 0.5 ? 'admit' : 'reject'
    }

    // Exploit: choose action with highest predicted reward
    const contextVec = this.contextToVector(context)
    this.updateWeights() // Ensure weights are current

    const admitScore = this.predictReward(contextVec, this.admitWeights)
    const rejectScore = this.predictReward(contextVec, this.rejectWeights)

    return admitScore > rejectScore ? 'admit' : 'reject'
  }

  private predictReward(context: number[], weights: number[]): number {
    return this.dotProduct(context, weights)
  }

  private updateWeights() {
    this.admitWeights = this.solveLinearSystem(this.admitA, this.admitB)
    this.rejectWeights = this.solveLinearSystem(this.rejectA, this.rejectB)
  }

  private solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = b.length
    const weights = Array(n).fill(0)

    // Use diagonal approximation with regularization
    for (let i = 0; i < n; i++) {
      const diag = A[i][i]
      if (Math.abs(diag) > 1e-8) {
        weights[i] = b[i] / diag

        // Clip weights to prevent explosion
        weights[i] = Math.max(-10, Math.min(10, weights[i]))
      }
    }

    return weights
  }

  updateModel(context: number[], action: 'admit' | 'reject', reward: number) {
    const A = action === 'admit' ? this.admitA : this.rejectA
    const b = action === 'admit' ? this.admitB : this.rejectB

    // Update A = A + x * x^T
    for (let i = 0; i < context.length; i++) {
      for (let j = 0; j < context.length; j++) {
        A[i][j] += context[i] * context[j]
      }
    }

    // Update b = b + r * x
    for (let i = 0; i < context.length; i++) {
      b[i] += reward * context[i]
    }
  }

  private dotProduct(a: number[], b: number[]): number {
    let sum = 0
    const len = Math.min(a.length, b.length)
    for (let i = 0; i < len; i++) {
      sum += a[i] * b[i]
    }
    return sum
  }

  getStats() {
    return {
      admitWeights: [...this.admitWeights],
      rejectWeights: [...this.rejectWeights],
      epsilon: this.epsilon,
      contextDim: this.contextDim,
      decisionCount: this.decisionCount,
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

    const previousGames = await this.getPreviousGameResults()
    const allDecisions = previousGames.flatMap((game) => game.decisions || [])

    // Calculate exact context dimension
    const numConstraints = this.constraints.size
    const contextDim =
      4 + // basic state features
      numConstraints + // progress ratios
      numConstraints + // urgency scores
      numConstraints + // person attributes
      4 // derived features

    this.bandit = new ContextualBandit(contextDim, allDecisions)

    console.log(
      `Bandit initialized: ${allDecisions.length} decisions, ${contextDim}D context`
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
    const context = this.buildContext(person)
    const action = this.bandit.selectAction(context)
    const shouldAdmit = action === 'admit'

    // Calculate reward after seeing the outcome
    const reward = this.calculateReward(person, shouldAdmit, context)

    // Update bandit
    const contextVec = this.bandit.contextToVector(context)
    this.bandit.updateModel(contextVec, action, reward)

    // Record decision
    this.decisions.push({
      context,
      action,
      person,
      reward,
      constraintsSatisfied: this.getConstraints().map((c) => c.isSatisfied()),
    })

    // Update constraints
    this.constraints.forEach((constraint) => {
      constraint.update(person, shouldAdmit)
    })

    if (shouldAdmit) {
      this.totalAdmitted++
    } else {
      this.totalRejected++
    }

    // Log important decisions
    if (this.remainingSlots < 100 || reward < -10) {
      console.log(
        `[${this.remainingSlots} slots] ${JSON.stringify(
          person
        )} -> ${action} (reward: ${reward.toFixed(2)})`
      )
    }

    return shouldAdmit
  }

  private buildContext(person: Person<T>): Context {
    const constraints = this.getConstraints()
    const progressRatios = constraints.map((c) => c.getProgress())
    const urgencyScores = constraints.map((c) =>
      c.getUrgency(this.remainingSlots)
    )
    const personAttributes = constraints.map((c) => !!person[c.attribute])

    return {
      admittedCount: this.totalAdmitted,
      remainingSlots: this.remainingSlots,
      progressRatios,
      urgencyScores,
      personAttributes,
      scarcityScore: this.calculateScarcityScore(),
      capacityUtilization: this.totalAdmitted / this.config.MAX_CAPACITY,
      worstConstraintProgress: Math.min(...progressRatios),
      hasRareAttribute: constraints.some(
        (c) => person[c.attribute] && c.isRare()
      ),
    }
  }

  private calculateReward(
    person: Person<T>,
    admitted: boolean,
    context: Context
  ): number {
    let reward = 0

    if (admitted) {
      // Calculate value of this person for unsatisfied constraints ONLY
      let personValue = 0
      let hasUsefulAttribute = false
      let hasOverSatisfiedAttribute = false

      this.constraints.forEach((constraint, attr) => {
        if (person[attr]) {
          if (!constraint.isSatisfied()) {
            hasUsefulAttribute = true
            const progress = constraint.getProgress()
            const urgency = constraint.getUrgency(this.remainingSlots)

            // Exponentially higher rewards for less satisfied constraints
            const progressValue = Math.pow(1 - progress, 2) * 25 // 0-25 points, exponential
            const urgencyValue = Math.min(urgency * 3, 30) // 0-30 points

            personValue += progressValue + urgencyValue

            // Massive bonus for rare attributes when desperately needed
            if (constraint.isRare() && progress < 0.5) {
              personValue += 50
            }
          } else {
            // Heavy penalty for over-satisfying constraints
            hasOverSatisfiedAttribute = true
            personValue -= 15 // significant penalty
          }
        }
      })

      // Special bonus: focus heavily on the worst constraint
      const worstProgress = Math.min(...context.progressRatios)
      if (worstProgress < 0.3) {
        const worstConstraintIndex =
          context.progressRatios.indexOf(worstProgress)
        const worstConstraint = this.getConstraints()[worstConstraintIndex]
        if (person[worstConstraint.attribute]) {
          personValue += 40 // huge bonus for helping worst constraint
        }
      }

      reward = personValue

      // Massive penalties for bad decisions
      if (!hasUsefulAttribute) {
        const wastefulness = Math.max(1, (1000 - this.remainingSlots) / 50)
        reward -= wastefulness * 20

        if (this.remainingSlots < 50) {
          reward -= 150
        }
        if (this.remainingSlots < 20) {
          reward -= 300
        }
      }

      // Extra penalty for admitting people who only help over-satisfied constraints
      if (hasOverSatisfiedAttribute && !hasUsefulAttribute) {
        reward -= 25 // penalty for wasting slot on over-satisfied constraints
      }

      // Penalty for impossible situations
      if (this.remainingSlots < 100) {
        const unsatisfiedConstraints = this.getConstraints().filter(
          (c) => !c.isSatisfied()
        )
        const totalNeeded = unsatisfiedConstraints.reduce((sum, c) => {
          const needed = c.minRequired - c.admitted
          return sum + needed / c.frequency
        }, 0)

        if (totalNeeded > this.remainingSlots * 3 && !hasUsefulAttribute) {
          reward -= 50
        }
      }
    } else {
      // Rejecting someone
      const hasUsefulAttribute = Array.from(this.constraints.entries()).some(
        ([attr, constraint]) => person[attr] && !constraint.isSatisfied()
      )

      if (!hasUsefulAttribute) {
        reward += 5 // good rejection

        // Bonus for rejecting over-satisfied constraint holders
        const hasOverSatisfiedOnly = Array.from(
          this.constraints.entries()
        ).some(([attr, constraint]) => person[attr] && constraint.isSatisfied())
        if (hasOverSatisfiedOnly) {
          reward += 10 // extra bonus for smart rejection
        }
      } else {
        // Penalty for rejecting useful people
        const maxUrgency = Math.max(...context.urgencyScores, 0)
        reward -= Math.min(maxUrgency * 0.3, 10) // reduced penalty since we want to be selective

        // But smaller penalty if we're running out of chances
        if (this.remainingSlots < 200) {
          reward -= 3
        }
      }
    }

    return reward
  }

  private calculateScarcityScore(): number {
    const constraints = this.getConstraints()
    const totalExpected = constraints
      .filter((c) => !c.isSatisfied())
      .reduce((sum, c) => {
        const needed = c.minRequired - c.admitted
        return sum + needed / c.frequency
      }, 0)

    return totalExpected / Math.max(1, this.remainingSlots)
  }

  getProgress() {
    return {
      attributes: this.getConstraints().map((constraint) => ({
        attribute: constraint.attribute,
        admitted: constraint.admitted,
        required: constraint.minRequired,
        satisfied: constraint.isSatisfied(),
        progress: constraint.getProgress(),
        urgency: constraint.getUrgency(this.remainingSlots),
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
        finalConstraintStatus: this.getConstraints().map((c) => ({
          attribute: c.attribute,
          satisfied: c.isSatisfied(),
          progress: c.getProgress(),
          shortfall: Math.max(0, c.minRequired - c.admitted),
        })),
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
        .filter((game) => game.output?.decisions) // only games with decision data
        .map((game) => ({
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
