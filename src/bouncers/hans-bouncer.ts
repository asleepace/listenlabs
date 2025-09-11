import type { BergainBouncer } from '../berghain'
import type {
  GameConstraints,
  GameState,
  GameStatusRunning,
  ScenarioAttributes,
} from '../types'
import { Disk } from '../utils/disk'

interface GameResult {
  gameId: string
  finalScore: number
  thresholdParams: any
  timestamp: Date
  constraints: any[]
}

async function getPreviousGameResults(): Promise<GameResult[]> {
  const savedGameData = await Disk.getJsonDataFromFiles<
    GameState<{
      finished: boolean
      finalScore: number
      thresholdParams: ThresholdParams<any>
      timestamp: string
      constraints: GameConstraints
    }>
  >()

  console.log({ savedGameData })

  return savedGameData
    .map((previous): GameResult | undefined => {
      if (!previous || !previous.output || !previous.output?.finished)
        return undefined
      if (previous.status.status !== 'completed') return undefined
      return {
        gameId: previous.game.gameId,
        finalScore: previous.output.finalScore || 20000,
        thresholdParams: previous.output.thresholdParams || {},
        timestamp: new Date(previous.timestamp || 0),
        constraints: previous.game?.constraints || [],
      }
    })
    .filter((data): data is GameResult => data !== undefined)
}

/**
 * Game configuration data.
 */
export interface Config {
  TOTAL_PEOPLE: number
  TARGET_RANGE: number
  MAX_CAPACITY: number
}

type Correlations<T> = {
  [K in keyof T]: {
    [J in keyof T]: number
  }
}

type Person<T> = Record<keyof T, boolean>

interface Statistics<T> {
  correlations: Correlations<T>
  relativeFrequencies: Record<keyof T, number>
}

type ThresholdParams<T> = {
  [K in keyof T]: number
}

interface ConstraintStats {
  needed: number
  expectedPeople: number
  urgency: number
}

/**
 * Thompson Sampling for learning optimal threshold parameters
 */
class ThompsonSampler<T> {
  private alphas: Record<keyof T, number> = {} as any
  private betas: Record<keyof T, number> = {} as any

  private previousGames: GameResult[] = []

  constructor(attributes: (keyof T)[], previousGames: GameResult[] = []) {
    // Initialize with prior knowledge
    attributes.forEach((attr) => {
      this.alphas[attr] = 1
      this.betas[attr] = 1
    })

    // Update with previous game results
    this.initializeFromHistory(previousGames)
  }

  private initializeFromHistory(previousGames: GameResult[]) {
    if (previousGames.length === 0) {
      console.log('No previous games found, starting with uniform priors')
      return
    }
    // set previous games on the class
    this.previousGames = previousGames

    // Find best score to use as success threshold
    const scores = previousGames.map((g) => g.finalScore)
    const bestScore = Math.min(...scores)
    const successThreshold = Math.min(bestScore * 1.2, 5000) // 20% worse than best, max 5000

    console.log(
      `[thompson-sampling] initializing Thompson Sampling from ${previousGames.length} games`
    )
    console.log(
      `[thompson-sampling] best previous score: ${bestScore}, success threshold: ${successThreshold}`
    )

    let successCount = 0
    let totalCount = 0

    previousGames.forEach((game) => {
      if (!game.thresholdParams) return

      const wasSuccess = game.finalScore <= successThreshold
      if (wasSuccess) successCount++
      totalCount++

      Object.entries(game.thresholdParams).forEach(([attr, value]) => {
        const key = attr as keyof T
        if (this.alphas[key] !== undefined) {
          const paramValue = value as number

          if (wasSuccess) {
            // Reward successful parameter values
            this.alphas[key] += paramValue * 0.35 // Weight successful params
          } else {
            // Penalize unsuccessful parameter values
            this.betas[key] += (2 - paramValue) * 0.25
          }
        }
      })
    })

    console.log(
      `[thompson-sampling] initialized: ${successCount}/${totalCount} successful games`
    )
    console.log('[thompson-sampling] updated priors:', {
      alphas: Object.fromEntries(Object.entries(this.alphas)),
      betas: Object.fromEntries(Object.entries(this.betas)),
    })
  }

  sample(): ThresholdParams<T> {
    const params = {} as ThresholdParams<T>
    Object.keys(this.alphas).forEach((attr) => {
      const key = attr as keyof T
      params[key] = this.sampleBeta(this.alphas[key], this.betas[key])
    })
    return params
  }

  get previousScores(): number[] {
    return this.previousGames.map((game) => game.finalScore)
  }

  update(params: ThresholdParams<T>, finalScore: number): void {
    // Use relative performance instead of absolute threshold
    const scores = this.previousScores // track all historical scores
    const percentile =
      scores.filter((s) => s > finalScore).length / scores.length
    const success = percentile > 0.7 // top 30% of performances

    Object.entries(params).forEach(([attr, value]) => {
      const key = attr as keyof T
      const val = value as number
      if (success) {
        this.alphas[key] += val
      } else {
        this.betas[key] += 1 - val // beta gets opposite weight
      }
    })
  }

  private sampleBeta(alpha: number, beta: number): number {
    // Simple approximation - in production use proper beta sampling
    const mean = alpha / (alpha + beta)
    const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    const noise = (Math.random() - 0.5) * Math.sqrt(variance) * 2
    return Math.max(0.1, Math.min(3.0, mean + noise)) // Increased max from 2.0 to 3.0
  }

  getStats() {
    return {
      alphas: Object.fromEntries(Object.entries(this.alphas)),
      betas: Object.fromEntries(Object.entries(this.betas)),
    }
  }
}

/**
 * Each constraint tracks its own internal state.
 */
class Constraint<T> {
  public admitted = 0
  public rejected = 0

  constructor(
    public attribute: keyof T,
    public minRequired: number,
    public frequency: number,
    public correlations: Correlations<T>,
    public config: Config
  ) {}

  public getScore(nextPerson: Person<T>): number {
    if (!nextPerson[this.attribute]) return 0.5 // Neutral score for non-matching
    return 2.0 // High value for matching attributes
  }

  public update(nextPerson: Person<T>, admitted: boolean): void {
    if (!nextPerson[this.attribute]) return
    if (admitted) this.admitted++
    else this.rejected++
  }

  public getStats(remainingSlots: number): ConstraintStats {
    const needed = Math.max(0, this.minRequired - this.admitted)
    const expectedPeople = needed / this.frequency
    const urgency = remainingSlots > 0 ? needed / remainingSlots : 1.0

    return { needed, expectedPeople, urgency }
  }

  public isSatisfied(): boolean {
    return this.admitted >= this.minRequired
  }

  public isRare(): boolean {
    return this.frequency <= 0.1
  }

  public getOverageMultiplier(): number {
    if (!this.isSatisfied()) return 1.0 // No penalty if not satisfied

    const overage = this.admitted - this.minRequired
    const overageRatio = overage / this.minRequired

    // Exponential penalty for overages
    // 10% overage = 0.9x multiplier
    // 50% overage = 0.5x multiplier
    // 100% overage = 0.25x multiplier
    return Math.max(0.1, 1 / (1 + overageRatio * 2))
  }

  public getAttributeValue(
    stats: ConstraintStats,
    learnedMultiplier: number
  ): number {
    const urgencyMultiplier = Math.max(0.5, 1 + stats.urgency)

    if (this.isSatisfied()) {
      // For satisfied constraints, apply overage penalty
      const baseValue = 0.5 // Base value for satisfied constraint
      return baseValue * this.getOverageMultiplier()
    } else {
      // For unsatisfied constraints, normal calculation
      return (1 + stats.urgency * learnedMultiplier) * urgencyMultiplier
    }
  }
}

/**
 * The bouncer tracks global state along with managing attributes.
 */
export class HansBouncer<T> implements BergainBouncer {
  public constraints = new Map<keyof T, Constraint<T>>()
  public totalAdmitted = 0
  public totalRejected = 0
  private thresholdParams!: ThresholdParams<T>
  private sampler!: ThompsonSampler<T>

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
        this.getCorrelations(attribute),
        this.config
      )
      this.set(constraint)
    }
  }

  async initializeLearningData() {
    console.log('#'.repeat(65))
    // Load previous game data for Thompson Sampling
    const previousGames = await getPreviousGameResults()
    console.log({ previousGames })
    this.sampler = new ThompsonSampler(
      Array.from(this.constraints.keys()),
      previousGames
    )
    this.thresholdParams = this.sampler.sample()
    console.log(
      'Thompson Sampling initialized with parameters:',
      this.thresholdParams
    )
    console.log('Sampler stats:', this.sampler.getStats())
  }

  get statistics(): Statistics<T> {
    return this.state.game.attributeStatistics as any
  }

  get totalPeople(): number {
    return this.totalAdmitted + this.totalRejected
  }

  get remainingSlots(): number {
    return this.config.MAX_CAPACITY - this.totalAdmitted
  }

  get remainingRejections(): number {
    return this.config.TOTAL_PEOPLE - this.totalRejected
  }

  // Interface methods

  admit({
    status,
    nextPerson,
  }: GameStatusRunning<ScenarioAttributes>): boolean {
    if (status !== 'running') return false
    if (this.remainingSlots <= 0) return false

    // Extract attributes from the nested structure
    const person = nextPerson.attributes as Person<T>
    const shouldAdmit = this.makeDecision(person)

    // Update constraint states
    this.constraints.forEach((constraint) => {
      constraint.update(person, shouldAdmit)
    })

    if (shouldAdmit) {
      this.totalAdmitted++
    } else {
      this.totalRejected++
    }

    return shouldAdmit
  }

  getProgress() {
    return {
      attributes: this.getConstraints().map((constraint) => ({
        attribute: constraint.attribute,
        admitted: constraint.admitted,
        required: constraint.minRequired,
        satisfied: constraint.isSatisfied(),
        needed: constraint.getStats(this.remainingSlots).needed,
        expectedPeople: constraint.getStats(this.remainingSlots).expectedPeople,
        urgency: constraint.getStats(this.remainingSlots).urgency,
      })),
      totalAdmitted: this.totalAdmitted,
      totalRejected: this.totalRejected,
      remainingSlots: this.remainingSlots,
    }
  }

  getOutput() {
    const finalScore = this.totalRejected
    this.sampler.update(this.thresholdParams, finalScore)

    type GameProgress = ReturnType<typeof this.getProgress>

    const gameData: GameState<
      GameProgress & {
        finished: boolean
        finalScore: number
        thresholdParams: ThresholdParams<T>
      }
    > = {
      ...this.state,
      timestamp: new Date().toISOString(),
      output: {
        ...this.getProgress(),
        finished: true,
        finalScore,
        thresholdParams: this.thresholdParams,
        // constraints: this.getConstraints(),
      },
    }

    Disk.saveGameState(gameData).catch((e) => {
      console.warn('[has-bouncer] failed to save game file:', e)
    })

    return gameData
  }

  // Core decision logic
  private makeDecision(person: Person<T>): boolean {
    // Calculate person's value
    const personValue = this.calculatePersonValue(person)
    const requiredThreshold = this.getRequiredThreshold()

    // Debug logging
    const hasAnyAttribute = Object.values(person).some((v) => v)
    console.log(`Person: ${JSON.stringify(person)}`)
    console.log(`Has any attribute: ${hasAnyAttribute}`)
    console.log(`Value: ${personValue}, Threshold: ${requiredThreshold}`)
    console.log(`Decision: ${personValue >= requiredThreshold}`)
    console.log('---')

    return personValue >= requiredThreshold
  }

  private calculatePersonValue(person: Person<T>): number {
    let value = 0.01 // Base value for everyone (never 0)
    let attributeCount = 0
    let hasRareAttribute = false

    // Debug: log what we're checking
    console.log(`Calculating value for person:`, person)
    console.log(`Constraints:`, Array.from(this.constraints.keys()))

    this.constraints.forEach((constraint) => {
      const hasAttribute = person[constraint.attribute]
      console.log(`  Checking ${String(constraint.attribute)}: ${hasAttribute}`)

      if (!hasAttribute) return // skip for this attribute

      const stats = constraint.getStats(this.remainingSlots)

      // if the person has an attribute that appears < 10% of the time,
      // set this flag to true.
      if (constraint.isRare()) {
        hasRareAttribute = true
      }

      // Use learned parameter for urgency scaling
      const learnedMultiplier = this.thresholdParams[constraint.attribute]

      // add score to overall value
      value += constraint.getAttributeValue(stats, learnedMultiplier)

      // incrament attribute count
      attributeCount++
    })

    // heavily reward rare attributes
    if (hasRareAttribute) {
      value += 10
    }

    const totalExpectedPeople = this.getConstraints()
      .filter((c) => !c.isSatisfied())
      .reduce(
        (sum, c) => sum + c.getStats(this.remainingSlots).expectedPeople,
        0
      )

    const scarcityRatio = totalExpectedPeople / Math.max(1, this.remainingSlots)
    // const globalScarcityMultiplier = Math.min(5, Math.max(1, scarcityRatio))

    // more aggresive:
    // Early game (ratio 10): multiplier = 3.16
    // Mid game (ratio 20): multiplier = 3 (capped)
    // Late game (ratio 50+): multiplier = 3 (capped)
    const globalScarcityMultiplier = Math.min(
      3,
      Math.max(1, Math.sqrt(scarcityRatio))
    )

    const finalValue =
      value * Math.max(1, attributeCount) * globalScarcityMultiplier

    console.log(
      `Final value: ${finalValue} (base: ${value}, scarcity: ${globalScarcityMultiplier})`
    )
    return finalValue
  }

  /**
   *  Scales the threshold by the bottlekecks found.
   */
  private getRequiredThreshold(): number {
    const progressRatio = this.totalAdmitted / this.config.MAX_CAPACITY
    const constraints = this.getConstraints()

    // Find the bottleneck constraint - the one requiring the most people
    const bottleneckConstraint = constraints
      .filter((c) => !c.isSatisfied())
      .reduce(
        (worst, constraint) => {
          const stats = constraint.getStats(this.remainingSlots)
          return stats.expectedPeople > worst.expectedPeople ? stats : worst
        },
        { expectedPeople: 0, urgency: 0 }
      )

    // Calculate how impossible the bottleneck is
    const impossibilityRatio =
      bottleneckConstraint.expectedPeople / Math.max(1, this.remainingSlots)

    // Base threshold scales with impossibility
    let baseThreshold = 0.1 + Math.min(5, impossibilityRatio * 0.5) // 0.1 to 5.6 range

    // If we're ahead of pace on constraints, ease up
    const constraintProgress = constraints.map((c) => {
      if (c.isSatisfied()) return 1.0
      const stats = c.getStats(this.remainingSlots)
      const expectedAtThisPoint = c.minRequired * progressRatio
      return c.admitted / Math.max(1, expectedAtThisPoint)
    })

    const avgProgress =
      constraintProgress.reduce((sum, p) => sum + p, 0) /
      constraintProgress.length

    // Ease threshold if we're ahead of schedule
    if (avgProgress > 1.2) baseThreshold *= 0.7 // 30% easier if 20% ahead
    else if (avgProgress > 1.1) baseThreshold *= 0.85 // 15% easier if 10% ahead
    else if (avgProgress < 0.8) baseThreshold *= 1.5 // 50% harder if 20% behind

    // Emergency mode for desperate constraints
    const maxUrgency = Math.max(
      ...constraints.map((c) => c.getStats(this.remainingSlots).urgency)
    )
    if (maxUrgency > 2.0) baseThreshold *= 0.2 // Panic mode
    else if (maxUrgency > 1.5) baseThreshold *= 0.5 // Urgent mode

    return Math.max(0.05, baseThreshold)
  }

  // Helper methods

  set(constraint: Constraint<T>): void {
    this.constraints.set(constraint.attribute, constraint)
  }

  get(attribute: keyof T): Constraint<T> {
    return this.constraints.get(attribute)!
  }

  getConstraints() {
    return Array.from(this.constraints.values())
  }

  getFrequency(attribute: keyof T): number {
    return this.statistics.relativeFrequencies[attribute]! as any
  }

  getCorrelations(attribute: keyof T): Correlations<T> {
    return this.statistics.correlations[attribute]! as any
  }
}
