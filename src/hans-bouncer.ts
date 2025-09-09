import type { BergainBouncer } from './berghain'
import type { GameState, GameStatusRunning, ScenarioAttributes } from './types'

namespace LearningData {
  export async function saveData(state: GameState) {
    const file = Bun.file(state.file)
    const json = JSON.stringify(state, null, 2)
    await file.write(json)
  }

  export async function getSavedData(filePath: string): Promise<GameState> {
    const file = Bun.file(filePath)
    const data: GameState = await file.json()
    if (!data) throw new Error(`Failed to load game file: ${filePath}`)
    return data
  }

  export interface GameResult {
    gameId: string
    finalScore: number
    thresholdParams: any
    timestamp: Date
    constraints: any[]
  }

  export async function getPreviousGameData(): Promise<GameResult[]> {
    try {
      const dataDir = './data/'
      const dirExists = await Bun.file(dataDir).exists()

      if (!dirExists) {
        console.log('No data directory found, starting fresh')
        return []
      }

      // Get all JSON files in the data directory
      const files = await Array.fromAsync(
        new Bun.Glob('*.json').scan({ cwd: dataDir })
      )

      const previousGames: GameResult[] = []

      for (const filename of files) {
        try {
          const filePath = `${dataDir}${filename}`
          const gameData = await getSavedData(filePath)

          // Extract game results if the game is finished
          if (gameData.output?.finished) {
            previousGames.push({
              gameId: gameData.game?.gameId || filename,
              finalScore: gameData.output.finalScore || 20000,
              thresholdParams: gameData.output.thresholdParams || {},
              timestamp: new Date(gameData.output || 0),
              constraints: gameData.game?.constraints || [],
            })
          }
        } catch (error) {
          console.warn(`Failed to load game data from ${filename}:`, error)
        }
      }

      // Sort by timestamp (newest first)
      previousGames.sort(
        (a, b) => b.timestamp.getTime() - a.timestamp.getTime()
      )

      console.log(
        `Loaded ${previousGames.length} previous games for Thompson Sampling`
      )
      return previousGames
    } catch (error) {
      console.warn('Error loading previous game data:', error)
      return []
    }
  }

  export async function saveGameResult(state: GameState, bouncerOutput: any) {
    // Enhanced save that includes bouncer results for Thompson Sampling
    const enhancedState = {
      ...state,
      bouncer: bouncerOutput,
      timestamp: new Date().toISOString(),
    }

    await saveData(enhancedState)
  }
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

  constructor(
    attributes: (keyof T)[],
    previousGames: LearningData.GameResult[] = []
  ) {
    // Initialize with prior knowledge
    attributes.forEach((attr) => {
      this.alphas[attr] = 1
      this.betas[attr] = 1
    })

    // Update with previous game results
    this.initializeFromHistory(previousGames)
  }

  private initializeFromHistory(previousGames: LearningData.GameResult[]) {
    if (previousGames.length === 0) {
      console.log('No previous games found, starting with uniform priors')
      return
    }

    // Find best score to use as success threshold
    const scores = previousGames.map((g) => g.finalScore)
    const bestScore = Math.min(...scores)
    const successThreshold = Math.min(bestScore * 1.2, 5000) // 20% worse than best, max 5000

    console.log(
      `Initializing Thompson Sampling from ${previousGames.length} games`
    )
    console.log(
      `Best previous score: ${bestScore}, success threshold: ${successThreshold}`
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
            this.alphas[key] += paramValue * 0.5 // Weight successful params
          } else {
            // Penalize unsuccessful parameter values
            this.betas[key] += (2 - paramValue) * 0.3
          }
        }
      })
    })

    console.log(
      `Thompson Sampling initialized: ${successCount}/${totalCount} successful games`
    )
    console.log('Updated priors:', {
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

  update(params: ThresholdParams<T>, finalScore: number): void {
    const success = finalScore < 5000 // Adjust based on your target

    Object.entries(params).forEach(([attr, value]) => {
      const key = attr as keyof T
      if (success) {
        this.alphas[key] += value as number
      } else {
        this.betas[key] += 2 - (value as number)
      }
    })
  }

  private sampleBeta(alpha: number, beta: number): number {
    // Simple approximation - in production use proper beta sampling
    const mean = alpha / (alpha + beta)
    const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    const noise = (Math.random() - 0.5) * Math.sqrt(variance) * 2
    return Math.max(0.1, Math.min(2.0, mean + noise))
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

  async initializeLearningData() {
    // Load previous game data for Thompson Sampling
    const previousGames = await LearningData.getPreviousGameData()
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

    const gameOutput = {
      ...this.getProgress(),
      finished: true,
      finalScore,
      thresholdParams: this.thresholdParams,
    }

    LearningData.saveGameResult(this.state, gameOutput).catch((e) => {
      console.warn('[hans-bouncer] failed to saved game data:', e)
    })

    return gameOutput
  }

  // Core decision logic
  private makeDecision(person: Person<T>): boolean {
    // Emergency mode: accept anyone if we're running out of rejections
    if (this.remainingRejections < 100) return true

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
    let value = 0.1 // Base value for everyone (never 0)
    let attributeCount = 0

    // Debug: log what we're checking
    console.log(`Calculating value for person:`, person)
    console.log(`Constraints:`, Array.from(this.constraints.keys()))

    this.constraints.forEach((constraint) => {
      const hasAttribute = person[constraint.attribute]
      console.log(`  Checking ${String(constraint.attribute)}: ${hasAttribute}`)

      if (hasAttribute) {
        const stats = constraint.getStats(this.remainingSlots)
        console.log(
          `    Stats: urgency=${
            stats.urgency
          }, satisfied=${constraint.isSatisfied()}`
        )

        // Much higher value for unsatisfied constraints
        if (!constraint.isSatisfied()) {
          // Scale value by urgency - more urgent = much higher value
          const attributeValue = 1 + stats.urgency * 3
          console.log(`    Adding value: ${attributeValue}`)
          value += attributeValue
        } else {
          console.log(`    Adding satisfied value: 0.5`)
          value += 0.5 // Lower value for already satisfied constraints
        }
        attributeCount++
      }
    })

    // Significant bonus for people with multiple attributes
    if (attributeCount >= 2) {
      console.log(`  Multi-attribute bonus: +1`)
      value += 1
    }
    if (attributeCount >= 3) {
      console.log(`  Triple-attribute bonus: +1.5`)
      value += 1.5
    }

    // Extra bonus for creative people (since they're so rare)
    if (person['creative' as keyof T]) {
      console.log(`  Creative bonus: +2`)
      value += 2 // Big bonus for creative attribute
    }

    // If person has no useful attributes, still give small chance early in game
    if (attributeCount === 0 && this.totalAdmitted < 50) {
      console.log(`  Early game bonus for no attributes: 0.3`)
      value = 0.3 // Small chance early on
    }

    console.log(`  Final calculated value: ${value}`)
    return value
  }

  private getRequiredThreshold(): number {
    const progressRatio = this.totalAdmitted / this.config.MAX_CAPACITY
    const constraints = this.getConstraints()

    // Get most urgent constraint
    const mostUrgent = constraints.reduce((max, constraint) => {
      const stats = constraint.getStats(this.remainingSlots)
      return stats.urgency > max ? stats.urgency : max
    }, 0)

    // Start VERY low threshold early in the game
    let baseThreshold = 0.25 // Even more permissive

    // If any constraint is very urgent (>0.7), be extremely permissive
    if (mostUrgent > 0.7) {
      baseThreshold = 0.15 // Super permissive
    } else if (mostUrgent > 0.5) {
      baseThreshold = 0.2 // Very permissive
    }

    // As venue fills up, become more selective
    const capacityFactor = Math.min(1.0, progressRatio * 1.5)

    return baseThreshold + capacityFactor * 0.2
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
}
