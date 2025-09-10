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
  criticalityLevel: number
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

  isNearlySatisfied(): boolean {
    return this.getProgress() >= 0.95
  }
}

/**
  # Bandit Bouncer

  1. Mathematical Foundation

  Proper context dimension calculation (4 + 3×numConstraints + 5)
  Improved linear algebra with adaptive regularization
  Better weight clipping based on feature importance
  Tanh normalization for urgency scores

  2. Sophisticated Exploration Strategy

  Adaptive epsilon based on performance, criticality, and game state
  Zero exploration in critical endgame (≤10 slots or impossible constraints)
  Exploration bias toward rejection (safer default)
  Performance-based epsilon adjustment

  3. Enhanced Override System

  Multiple override conditions (dire endgame, over-satisfied constraints)
  Overrides teach the bandit with positive rewards
  Clear logging of override reasons

  4. Strategic Reward Engineering

  Massive bonuses (100+ points) for worst constraint and rare attributes
  Exponential penalties for bad decisions (up to 800 points in final slots)
  Clear distinctions between useful/useless attributes
  Smart rejection rewards for over-satisfied constraints

  5. Robust Learning

  Learns preferentially from recent decisions
  Tracks recent performance for adaptive behavior
  Better historical data integration
  Memory management for long games

  6. Comprehensive Context

  Criticality level calculation combining capacity and impossibility
  Better scarcity scoring
  Enhanced person attribute detection
  Rich progress tracking

  This implementation should dramatically outperform the previous versions by:

  Never admitting useless people in endgame
  Heavily prioritizing the worst constraint (creative in your case)
  Learning quickly from massive reward differences
  Adapting exploration based on game criticality
  Making strategic decisions about constraint priorities

  The bandit will be conservative when it matters most and aggressive about pursuing the bottleneck constraints that determine success.
*/
class ContextualBandit {
  private admitWeights: number[]
  private rejectWeights: number[]
  private admitA: number[][]
  private rejectA: number[][]
  private admitB: number[]
  private rejectB: number[]
  private baseEpsilon: number = 0.15
  private contextDim: number
  private decisionCount: number = 0
  private recentRewards: number[] = []

  constructor(contextDimension: number, previousData: DecisionRecord[] = []) {
    this.contextDim = contextDimension

    // Initialize with proper regularization
    const lambda = 0.1
    this.admitA = this.createRegularizedIdentity(contextDimension, lambda)
    this.rejectA = this.createRegularizedIdentity(contextDimension, lambda)

    this.admitB = Array(contextDimension).fill(0)
    this.rejectB = Array(contextDimension).fill(0)

    // Initialize weights conservatively
    this.admitWeights = Array(contextDimension).fill(0)
    this.rejectWeights = Array(contextDimension).fill(0)

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

    // Learn from recent decisions with higher weight
    const recentDecisions = decisions.slice(-500) // Last 500 decisions

    recentDecisions.forEach((decision, index) => {
      const context = this.contextToVector(decision.context)
      const weight = 1 + (index / recentDecisions.length) * 0.5 // Recent decisions weighted more
      this.updateModel(context, decision.action, decision.reward * weight)
    })

    if (decisions.length > 0) {
      this.updateWeights()
      console.log(`Learned from ${recentDecisions.length} recent decisions`)
    }
  }

  contextToVector(context: Context): number[] {
    // Fixed feature order - exactly 4 + 3*numConstraints + 5 features
    const features = [
      // Basic state (4 features)
      context.admittedCount / 1000,
      context.remainingSlots / 1000,
      context.capacityUtilization,
      Math.min(context.scarcityScore / 1000, 5),

      // Constraint progress (numConstraints features)
      ...context.progressRatios,

      // Constraint urgencies (numConstraints features, normalized)
      ...context.urgencyScores.map((u) => Math.tanh(u / 50)), // tanh normalization

      // Person attributes (numConstraints features)
      ...context.personAttributes.map((x) => (x ? 1 : 0)),

      // Derived features (5 features)
      context.worstConstraintProgress,
      context.hasRareAttribute ? 1 : 0,
      context.criticalityLevel,
      context.progressRatios.filter((p) => p < 0.5).length /
        Math.max(1, context.progressRatios.length),
      context.progressRatios.filter((p) => p >= 0.95).length /
        Math.max(1, context.progressRatios.length),
    ]

    return features
  }

  selectAction(
    context: Context,
    bouncer: BanditBouncer<any>
  ): 'admit' | 'reject' {
    this.decisionCount++

    // Adaptive epsilon based on performance and criticality
    let currentEpsilon = this.calculateAdaptiveEpsilon(context)

    // ZERO exploration in critical endgame situations
    const constraints = bouncer.getConstraints()
    const hasImpossibleConstraints = this.hasImpossibleConstraints(
      context,
      constraints
    )

    if (context.remainingSlots <= 10 || hasImpossibleConstraints) {
      currentEpsilon = 0 // Pure exploitation
    }

    // Epsilon-greedy with adaptive exploration
    if (Math.random() < currentEpsilon) {
      return Math.random() < 0.3 ? 'admit' : 'reject' // Bias toward rejection during exploration
    }

    // Exploit: choose action with highest predicted reward
    const contextVec = this.contextToVector(context)
    this.updateWeights()

    const admitScore = this.predictReward(contextVec, this.admitWeights)
    const rejectScore = this.predictReward(contextVec, this.rejectWeights)

    return admitScore > rejectScore ? 'admit' : 'reject'
  }

  private calculateAdaptiveEpsilon(context: Context): number {
    // Start with base epsilon
    let epsilon = this.baseEpsilon

    // Decay over time
    epsilon *= Math.exp(-this.decisionCount / 1000)

    // Reduce exploration based on criticality
    epsilon *= 1 - context.criticalityLevel * 0.8

    // Reduce if recent performance is good
    if (this.recentRewards.length > 10) {
      const avgReward =
        this.recentRewards.reduce((a, b) => a + b) / this.recentRewards.length
      if (avgReward > 5) {
        epsilon *= 0.5 // Cut exploration when doing well
      }
    }

    return Math.max(0, Math.min(epsilon, 0.3))
  }

  private hasImpossibleConstraints(
    context: Context,
    constraints: Constraint<any>[]
  ): boolean {
    return constraints.some((constraint) => {
      if (constraint.isSatisfied()) return false
      const needed = constraint.minRequired - constraint.admitted
      const expectedNeeded = needed / constraint.frequency
      return expectedNeeded > context.remainingSlots * 2.5
    })
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

    // Improved diagonal approximation with regularization
    for (let i = 0; i < n; i++) {
      const diag = A[i][i]
      if (Math.abs(diag) > 1e-6) {
        weights[i] = b[i] / diag

        // Adaptive weight clipping based on feature importance
        const maxWeight = i < 4 ? 20 : 10 // Higher limits for basic features
        weights[i] = Math.max(-maxWeight, Math.min(maxWeight, weights[i]))
      }
    }

    return weights
  }

  updateModel(context: number[], action: 'admit' | 'reject', reward: number) {
    const A = action === 'admit' ? this.admitA : this.rejectA
    const b = action === 'admit' ? this.admitB : this.rejectB

    // Update matrices with regularization
    const regularization = 0.01
    for (let i = 0; i < context.length; i++) {
      for (let j = 0; j < context.length; j++) {
        A[i][j] += context[i] * context[j] + (i === j ? regularization : 0)
      }
      b[i] += reward * context[i]
    }

    // Track recent performance
    this.recentRewards.push(reward)
    if (this.recentRewards.length > 50) {
      this.recentRewards.shift() // Keep only recent rewards
    }
  }

  private dotProduct(a: number[], b: number[]): number {
    let sum = 0
    const len = Math.min(a.length, b.length)
    for (let i = 0; i < len; i++) {
      sum += (a[i] || 0) * (b[i] || 0)
    }
    return sum
  }

  getStats() {
    const avgRecentReward =
      this.recentRewards.length > 0
        ? this.recentRewards.reduce((a, b) => a + b) / this.recentRewards.length
        : 0

    return {
      admitWeights: this.admitWeights.slice(0, 10), // Only show first 10 for brevity
      rejectWeights: this.rejectWeights.slice(0, 10),
      baseEpsilon: this.baseEpsilon,
      contextDim: this.contextDim,
      decisionCount: this.decisionCount,
      avgRecentReward: avgRecentReward.toFixed(2),
      recentRewardCount: this.recentRewards.length,
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

    // Calculate exact context dimension: 4 + 3*numConstraints + 5
    const numConstraints = this.constraints.size
    const contextDim = 4 + numConstraints * 3 + 5

    this.bandit = new ContextualBandit(contextDim, allDecisions)

    console.log(
      `Bandit initialized: ${allDecisions.length} total decisions, ${contextDim}D context`
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

    // CRITICAL OVERRIDE: Force reject useless people in dire situations
    const hasUsefulAttribute = this.hasUsefulAttribute(person)
    const isDire = this.isDireEndgame(context)

    if (!hasUsefulAttribute && isDire) {
      return this.forceReject(
        person,
        context,
        'No useful attributes in dire endgame'
      )
    }

    // Override: Force reject people who only help over-satisfied constraints
    const onlyHelpsOverSatisfied =
      this.onlyHelpsOverSatisfiedConstraints(person)
    if (onlyHelpsOverSatisfied && context.remainingSlots < 50) {
      return this.forceReject(
        person,
        context,
        'Only helps over-satisfied constraints'
      )
    }

    // Normal bandit decision
    const action = this.bandit.selectAction(context, this)
    const shouldAdmit = action === 'admit'

    // Calculate reward and update
    const reward = this.calculateReward(person, shouldAdmit, context)
    const contextVec = this.bandit.contextToVector(context)
    this.bandit.updateModel(contextVec, action, reward)

    // Record and update
    this.recordDecision(context, action, person, reward, shouldAdmit)

    return shouldAdmit
  }

  private buildContext(person: Person<T>): Context {
    const constraints = this.getConstraints()
    const progressRatios = constraints.map((c) => c.getProgress())
    const urgencyScores = constraints.map((c) =>
      c.getUrgency(this.remainingSlots)
    )
    const personAttributes = constraints.map((c) => !!person[c.attribute])

    const worstProgress = Math.min(...progressRatios)
    const criticalityLevel = this.calculateCriticalityLevel()

    return {
      admittedCount: this.totalAdmitted,
      remainingSlots: this.remainingSlots,
      progressRatios,
      urgencyScores,
      personAttributes,
      scarcityScore: this.calculateScarcityScore(),
      capacityUtilization: this.totalAdmitted / this.config.MAX_CAPACITY,
      worstConstraintProgress: worstProgress,
      hasRareAttribute: this.hasRareAttribute(person),
      criticalityLevel,
    }
  }

  private calculateCriticalityLevel(): number {
    // 0 = early game, 1 = critical endgame
    const capacityFactor = Math.min(1, (1000 - this.remainingSlots) / 900)

    const unsatisfiedConstraints = this.getConstraints().filter(
      (c) => !c.isSatisfied()
    )
    const impossibilityFactor = unsatisfiedConstraints.reduce(
      (max, constraint) => {
        const needed = constraint.minRequired - constraint.admitted
        const expectedNeeded = needed / constraint.frequency
        const impossibility = Math.min(
          1,
          expectedNeeded / Math.max(1, this.remainingSlots)
        )
        return Math.max(max, impossibility)
      },
      0
    )

    return Math.max(capacityFactor, impossibilityFactor)
  }

  private hasUsefulAttribute(person: Person<T>): boolean {
    return Array.from(this.constraints.entries()).some(
      ([attr, constraint]) => person[attr] && !constraint.isSatisfied()
    )
  }

  private hasRareAttribute(person: Person<T>): boolean {
    return this.getConstraints().some((c) => person[c.attribute] && c.isRare())
  }

  private isDireEndgame(context: Context): boolean {
    return context.remainingSlots <= 20 || context.criticalityLevel > 0.8
  }

  private onlyHelpsOverSatisfiedConstraints(person: Person<T>): boolean {
    const hasAnyAttribute = Array.from(this.constraints.entries()).some(
      ([attr]) => person[attr]
    )

    if (!hasAnyAttribute) return false

    const helpsUnsatisfied = Array.from(this.constraints.entries()).some(
      ([attr, constraint]) =>
        person[attr] &&
        !constraint.isSatisfied() &&
        !constraint.isNearlySatisfied()
    )

    return !helpsUnsatisfied
  }

  private forceReject(
    person: Person<T>,
    context: Context,
    reason: string
  ): boolean {
    const action = 'reject'
    const reward = 15 // Strong positive reward for good override

    const contextVec = this.bandit.contextToVector(context)
    this.bandit.updateModel(contextVec, action, reward)

    this.recordDecision(context, action, person, reward, false)

    console.log(`[OVERRIDE] ${this.remainingSlots} slots: ${reason}`)
    return false
  }

  private calculateReward(
    person: Person<T>,
    admitted: boolean,
    context: Context
  ): number {
    if (admitted) {
      return this.calculateAdmissionReward(person, context)
    } else {
      return this.calculateRejectionReward(person, context)
    }
  }

  private calculateAdmissionReward(
    person: Person<T>,
    context: Context
  ): number {
    let reward = 0
    let hasUsefulAttribute = false
    let hasOverSatisfiedAttribute = false

    // Evaluate each attribute
    this.constraints.forEach((constraint, attr) => {
      if (person[attr]) {
        if (!constraint.isSatisfied()) {
          const progress = constraint.getProgress()

          if (progress < 0.95) {
            // Only reward if truly needed
            hasUsefulAttribute = true
            const urgency = constraint.getUrgency(this.remainingSlots)

            // Exponential rewards for less satisfied constraints
            const progressValue = Math.pow(1 - progress, 2) * 30
            const urgencyValue = Math.min(urgency * 4, 40)

            reward += progressValue + urgencyValue

            // Massive bonus for rare attributes when desperately needed
            if (constraint.isRare() && progress < 0.6) {
              reward += 100
            }
          } else {
            // Penalty for nearly-satisfied constraints
            hasOverSatisfiedAttribute = true
            reward -= 15
          }
        } else {
          // Heavy penalty for over-satisfying
          hasOverSatisfiedAttribute = true
          reward -= 25
        }
      }
    })

    // Special focus on worst constraint
    const worstConstraint = this.getWorstConstraint()
    if (worstConstraint && person[worstConstraint.attribute]) {
      const progress = worstConstraint.getProgress()
      if (progress < 0.8) {
        reward += 100 - progress * 100 // 100 points when 0%, 20 points when 80%
      }
    }

    // Severe penalties for bad admissions
    if (!hasUsefulAttribute) {
      const severityMultiplier = Math.max(1, (1000 - this.remainingSlots) / 50)
      reward -= severityMultiplier * 25

      if (this.remainingSlots < 50) reward -= 200
      if (this.remainingSlots < 20) reward -= 400
      if (this.remainingSlots < 10) reward -= 800
    }

    // Penalty for helping only over-satisfied constraints
    if (hasOverSatisfiedAttribute && !hasUsefulAttribute) {
      reward -= 40
    }

    return reward
  }

  private calculateRejectionReward(
    person: Person<T>,
    context: Context
  ): number {
    const hasUsefulAttribute = this.hasUsefulAttribute(person)
    const onlyHelpsOverSatisfied =
      this.onlyHelpsOverSatisfiedConstraints(person)

    if (!hasUsefulAttribute || onlyHelpsOverSatisfied) {
      let reward = 8 // Good rejection

      if (onlyHelpsOverSatisfied) reward += 12 // Extra bonus for smart rejection
      if (context.remainingSlots < 50) reward += 5 // Bonus for endgame selectivity

      return reward
    } else {
      // Penalty for rejecting useful people, but keep it reasonable
      const worstConstraint = this.getWorstConstraint()
      const isHelpingWorstConstraint =
        worstConstraint && person[worstConstraint.attribute]

      let penalty = 8
      if (isHelpingWorstConstraint) penalty += 10
      if (context.remainingSlots < 100) penalty += 5

      return -penalty
    }
  }

  private getWorstConstraint(): Constraint<T> | null {
    const unsatisfied = this.getConstraints().filter((c) => !c.isSatisfied())
    if (unsatisfied.length === 0) return null

    return unsatisfied.reduce((worst, constraint) =>
      constraint.getProgress() < worst.getProgress() ? constraint : worst
    )
  }

  private recordDecision(
    context: Context,
    action: 'admit' | 'reject',
    person: Person<T>,
    reward: number,
    admitted: boolean
  ) {
    this.decisions.push({
      context,
      action,
      person,
      reward,
      constraintsSatisfied: this.getConstraints().map((c) => c.isSatisfied()),
    })

    // Update constraints and counts
    this.constraints.forEach((constraint) => {
      constraint.update(person, admitted)
    })

    if (admitted) {
      this.totalAdmitted++
    } else {
      this.totalRejected++
    }

    // Enhanced logging
    if (this.remainingSlots < 100 || Math.abs(reward) > 15) {
      const constraintStatus = this.getConstraints()
        .map(
          (c) => `${String(c.attribute)}:${(c.getProgress() * 100).toFixed(0)}%`
        )
        .join(' ')
      console.log(
        `[${this.remainingSlots} slots] ${JSON.stringify(
          person
        )} -> ${action} (${reward.toFixed(1)}) [${constraintStatus}]`
      )
    }
  }

  private calculateScarcityScore(): number {
    const constraints = this.getConstraints().filter((c) => !c.isSatisfied())
    const totalExpected = constraints.reduce((sum, c) => {
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
        isRare: constraint.isRare(),
        isNearlySatisfied: constraint.isNearlySatisfied(),
      })),
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      remainingSlots: this.remainingSlots,
      banditStats: this.bandit?.getStats(),
      criticalityLevel: this.calculateCriticalityLevel(),
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
          isRare: c.isRare(),
          frequency: c.frequency,
        })),
      },
    }

    Disk.saveGameState(gameData).catch((e) => {
      console.warn('[bandit-bouncer] failed to save game file:', e)
    })

    return gameData
  }

  // Public helper methods
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
          (game) => game.output?.decisions && game.output.decisions.length > 0
        )
        .map((game) => ({
          gameId: game.game.gameId,
          finalScore: game.output?.finalScore || 20000,
          timestamp: new Date(game.timestamp || 0),
          constraints: game.game?.constraints || [],
          decisions: game.output?.decisions || [],
        }))
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()) // Most recent first
    } catch (e) {
      console.warn('Failed to load previous games:', e)
      return []
    }
  }
}
