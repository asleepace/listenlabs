/**
 * File Structure Created:
 *
 * The learning system will create:
 *
 *  ./learning-data/
 *  └── portfolio-bouncer.json  // Persistent learning data
 *
 * Learning Evolution:
 *
 * Game 1: Pure portfolio theory (treats creative as high-risk/high-return asset)
 * Games 2-10: Starts blending learned patterns with theory
 * Games 10+: Heavily weighted toward learned optimal strategies
 *
 * Key Benefits for Your Creative Problem:
 *
 * Automatic Creative Detection: Portfolio theory naturally identifies creative as the highest-risk, highest-expected-return asset
 * Rate Learning: Will learn the optimal admission rate through trial and error
 * Pattern Recognition: Will discover that certain attribute combinations (like creative + others) are most valuable
 * Adaptive Thresholds: Risk tolerance adjusts based on what actually works
 *
 * Expected Learning Progression:
 *
 * Early Games: May struggle with rate control while learning
 * Middle Games: Should discover creative needs massive over-weighting
 * Later Games: Should consistently achieve better scores as it learns optimal patterns
 *
 * The system will automatically create the learning-data directory and start tracking performance. Each game's results will inform the next game's strategy, specifically targeting the issues you've been facing with creative quotas and admission rate control.
 * Ready to test this learning portfolio approach?
 */

import type { BergainBouncer } from '../core/berghain'
import type { GameState, ScenarioAttributes } from '../types'

interface LearningData {
  historicalPerformance: {
    scenario: string
    finalScore: number
    admissionRate: number
    quotaCompletion: Record<string, number>
    portfolioWeights: Record<string, number>
    avgSharpeRatio: number
  }[]
  learnedWeights: Record<string, number>
  riskProfile: {
    conservativeThreshold: number
    aggressiveThreshold: number
    optimalRiskTolerance: number
  }
  correlationInsights: Record<string, Record<string, number>>
}

interface AttributePortfolio {
  attribute: keyof ScenarioAttributes
  target: number
  current: number
  weight: number // Portfolio weight (0-1)
  risk: number // Volatility/difficulty score
  expectedReturn: number // How much progress this attribute adds
}

export class PortfolioBouncer implements BergainBouncer {
  private gameData: GameState['game']
  private portfolio: Map<keyof ScenarioAttributes, AttributePortfolio>
  private riskTolerance: number = 0.3
  private rebalanceThreshold: number = 0.1
  private learningData: LearningData
  private currentGameMetrics: {
    sharpeRatios: number[]
    admissionDecisions: boolean[]
    portfolioDeviations: number[]
  }

  constructor(gameState: GameState) {
    this.gameData = gameState.game
    this.portfolio = new Map()

    // Initialize learning data
    this.learningData = {
      historicalPerformance: [],
      learnedWeights: {},
      riskProfile: {
        conservativeThreshold: 0.2,
        aggressiveThreshold: 0.8,
        optimalRiskTolerance: 0.3,
      },
      correlationInsights: {},
    }

    // Initialize current game metrics
    this.currentGameMetrics = {
      sharpeRatios: [],
      admissionDecisions: [],
      portfolioDeviations: [],
    }

    this.initializePortfolio()
    this.initializeLearning()
  }

  /**
   * Static factory method to properly initialize with async loading
   */
  static async create(gameState: GameState): Promise<PortfolioBouncer> {
    const bouncer = new PortfolioBouncer(gameState)
    await bouncer.initializeLearning()
    return bouncer
  }

  private initializePortfolio() {
    const totalTarget = this.gameData.constraints.reduce(
      (sum, c) => sum + c.minCount,
      0
    )

    this.gameData.constraints.forEach((constraint) => {
      const frequency =
        this.gameData.attributeStatistics.relativeFrequencies[
          constraint.attribute
        ]!
      const weight = constraint.minCount / totalTarget

      // Risk = inverse of frequency * demand pressure
      const risk = (1 / frequency) * (constraint.minCount / 1000)

      // Expected return = how much this attribute helps portfolio completion
      const expectedReturn = weight / frequency

      this.portfolio.set(constraint.attribute, {
        attribute: constraint.attribute,
        target: constraint.minCount,
        current: 0,
        weight,
        risk,
        expectedReturn,
      })
    })
  }

  /**
   * Initialize learning system - async wrapper for constructor
   */
  private async initializeLearning() {
    try {
      const loadedData = await this.loadLearningData()
      this.learningData = loadedData
      this.applyLearnings()
    } catch (error) {
      console.warn('Learning initialization failed, using defaults:', error)
    }
  }

  /**
   * LEARNING SYSTEM - Load historical performance data
   * Uses Bun file system for persistent storage
   */
  private async loadLearningData(): Promise<LearningData> {
    try {
      const learningFile = Bun.file('./learning-data/portfolio-bouncer.json')

      if (await learningFile.exists()) {
        const data = await learningFile.json()
        console.log(
          `Loaded learning data: ${
            data.historicalPerformance?.length || 0
          } games`
        )
        return data
      } else {
        console.log('No existing learning data found, starting fresh')
      }
    } catch (error) {
      console.warn('Failed to load learning data:', error)
    }

    // Default learning data for first run
    return {
      historicalPerformance: [],
      learnedWeights: {},
      riskProfile: {
        conservativeThreshold: 0.2,
        aggressiveThreshold: 0.8,
        optimalRiskTolerance: 0.3,
      },
      correlationInsights: {},
    }
  }

  /**
   * LEARNING SYSTEM - Save performance data for future runs
   * Uses Bun file system for persistent storage
   */
  private async saveLearningData(gameResult: {
    finalScore: number
    admissionRate: number
    quotaCompletion: Record<string, number>
  }) {
    try {
      // Calculate performance metrics for this game
      const avgSharpeRatio =
        this.currentGameMetrics.sharpeRatios.length > 0
          ? this.currentGameMetrics.sharpeRatios.reduce(
              (sum, ratio) => sum + ratio,
              0
            ) / this.currentGameMetrics.sharpeRatios.length
          : 0

      const portfolioWeights: Record<string, number> = {}
      this.portfolio.forEach((asset, attr) => {
        portfolioWeights[attr as string] = asset.weight
      })

      // Add this game's performance to history
      this.learningData.historicalPerformance.push({
        scenario: this.gameData.gameId || 'unknown',
        finalScore: gameResult.finalScore,
        admissionRate: gameResult.admissionRate,
        quotaCompletion: gameResult.quotaCompletion,
        portfolioWeights,
        avgSharpeRatio,
      })

      // Keep only recent history (last 50 games)
      if (this.learningData.historicalPerformance.length > 50) {
        this.learningData.historicalPerformance =
          this.learningData.historicalPerformance.slice(-50)
      }

      // Update learned parameters based on successful games
      this.updateLearnedParameters()

      // Ensure learning data directory exists and save
      await Bun.write(
        './learning-data/portfolio-bouncer.json',
        JSON.stringify(this.learningData, null, 2)
      )

      console.log(
        `Learning data saved: ${this.learningData.historicalPerformance.length} total games`
      )
      console.log(
        `Latest performance: Score ${gameResult.finalScore}, Rate ${(
          gameResult.admissionRate * 100
        ).toFixed(1)}%`
      )
    } catch (error) {
      console.warn('Failed to save learning data:', error)
    }
  }

  /**
   * LEARNING SYSTEM - Update algorithm parameters based on historical performance
   */
  private updateLearnedParameters() {
    const history = this.learningData.historicalPerformance
    if (history.length < 3) return // Need minimum data

    // Find best performing games (top 20%)
    const sortedGames = [...history].sort((a, b) => a.finalScore - b.finalScore)
    const topPerformers = sortedGames.slice(
      0,
      Math.max(1, Math.floor(sortedGames.length * 0.2))
    )

    // Learn optimal weights from best games
    const attributeCounts: Record<string, number[]> = {}
    this.portfolio.forEach((_, attr) => {
      attributeCounts[attr as string] = []
    })

    topPerformers.forEach((game) => {
      Object.entries(game.portfolioWeights).forEach(([attr, weight]) => {
        if (attributeCounts[attr]) {
          attributeCounts[attr]!.push(weight)
        }
      })
    })

    // Calculate average optimal weights
    Object.entries(attributeCounts).forEach(([attr, weights]) => {
      if (weights.length > 0) {
        this.learningData.learnedWeights[attr] =
          weights.reduce((sum, w) => sum + w, 0) / weights.length
      }
    })

    // Learn optimal risk tolerance
    const successfulRiskProfiles = topPerformers.map((game) => {
      // Estimate risk tolerance from admission rate
      return game.admissionRate > 0.25
        ? 0.8
        : game.admissionRate < 0.15
        ? 0.2
        : 0.5
    })

    if (successfulRiskProfiles.length > 0) {
      this.learningData.riskProfile.optimalRiskTolerance =
        successfulRiskProfiles.reduce((sum, rt) => sum + rt, 0) /
        successfulRiskProfiles.length
    }
  }

  /**
   * Apply learned parameters to current game
   */
  private applyLearnings() {
    // Apply learned weights if available
    if (Object.keys(this.learningData.learnedWeights).length > 0) {
      this.portfolio.forEach((asset, attr) => {
        const learnedWeight = this.learningData.learnedWeights[attr as string]
        if (learnedWeight !== undefined) {
          // Blend learned weight with calculated weight
          const blendFactor = Math.min(
            this.learningData.historicalPerformance.length / 10,
            0.7
          )
          asset.weight =
            asset.weight * (1 - blendFactor) + learnedWeight * blendFactor
        }
      })
    }

    // Apply learned risk tolerance
    this.riskTolerance = this.learningData.riskProfile.optimalRiskTolerance
  }

  private calculatePortfolioDeviation(): number {
    let totalDeviation = 0
    let totalWeight = 0

    this.portfolio.forEach((asset) => {
      const currentWeight = asset.current / Math.max(1, asset.target)
      const targetWeight = asset.weight
      const deviation = Math.abs(currentWeight - targetWeight)

      totalDeviation += deviation * asset.weight
      totalWeight += asset.weight
    })

    return totalDeviation / totalWeight
  }

  private calculateSharpeRatio(
    attributes: Partial<ScenarioAttributes>
  ): number {
    let portfolioReturn = 0
    let portfolioRisk = 0
    let attributeCount = 0

    Object.entries(attributes).forEach(([attr, hasAttr]) => {
      if (hasAttr) {
        const asset = this.portfolio.get(attr as keyof ScenarioAttributes)
        if (asset && asset.current < asset.target) {
          portfolioReturn += asset.expectedReturn
          portfolioRisk += asset.risk * asset.risk
          attributeCount++
        }
      }
    })

    if (attributeCount === 0) return 0

    // Diversification benefit - reduce risk for multiple attributes
    const diversificationFactor = Math.sqrt(attributeCount) / attributeCount
    portfolioRisk *= diversificationFactor

    // Risk-free rate (baseline admission value)
    const riskFreeRate = 0.1

    return portfolioRisk > 0
      ? (portfolioReturn - riskFreeRate) / Math.sqrt(portfolioRisk)
      : 0
  }

  private shouldRebalance(): boolean {
    const deviation = this.calculatePortfolioDeviation()
    return deviation > this.rebalanceThreshold
  }

  private getRebalancingBonus(attributes: Partial<ScenarioAttributes>): number {
    if (!this.shouldRebalance()) return 1.0

    let bonus = 1.0

    Object.entries(attributes).forEach(([attr, hasAttr]) => {
      if (hasAttr) {
        const asset = this.portfolio.get(attr as keyof ScenarioAttributes)
        if (asset) {
          const currentWeight = asset.current / Math.max(1, asset.target)
          const targetWeight = asset.weight

          // Bonus for underweight assets, penalty for overweight
          if (currentWeight < targetWeight) {
            const underweightBonus =
              (targetWeight - currentWeight) / targetWeight
            bonus += underweightBonus * 2.0
          } else if (currentWeight > targetWeight * 1.2) {
            bonus *= 0.5 // Penalty for overweight
          }
        }
      }
    })

    return bonus
  }

  private updateRiskTolerance(admittedCount: number, spotsLeft: number) {
    const completionRatio = admittedCount / 1000

    // Start conservative, become more risk-tolerant as time runs out
    if (completionRatio < 0.3) {
      this.riskTolerance = 0.2 // Very conservative early
    } else if (completionRatio < 0.7) {
      this.riskTolerance = 0.4 // Moderate
    } else {
      this.riskTolerance = 0.8 // Aggressive late game
    }

    // Emergency mode for critical shortfalls
    const portfolioDeviation = this.calculatePortfolioDeviation()
    if (portfolioDeviation > 0.3 && spotsLeft < 200) {
      this.riskTolerance = 1.0 // Accept any useful attributes
    }
  }

  admit(status: GameState['status']): boolean {
    if (status.status !== 'running' || !status.nextPerson) return false

    const admittedCount = status.admittedCount
    const spotsLeft = 1000 - admittedCount
    const currentRate = admittedCount / (admittedCount + status.rejectedCount)

    // Update risk tolerance based on game state
    this.updateRiskTolerance(admittedCount, spotsLeft)

    // Calculate portfolio metrics for this candidate
    const sharpeRatio = this.calculateSharpeRatio(status.nextPerson.attributes)
    const rebalancingBonus = this.getRebalancingBonus(
      status.nextPerson.attributes
    )

    // Portfolio quality score
    const portfolioScore = sharpeRatio * rebalancingBonus

    // Risk-adjusted threshold
    const baseThreshold = 0.3
    const riskAdjustedThreshold = baseThreshold * (1 - this.riskTolerance)

    // Rate control - more selective if over-admitting
    const targetRate = 0.2
    const rateMultiplier = currentRate > targetRate ? 0.5 : 1.0

    const finalThreshold = riskAdjustedThreshold * rateMultiplier

    const shouldAdmit = portfolioScore > finalThreshold

    // Track metrics for learning
    this.currentGameMetrics.sharpeRatios.push(sharpeRatio)
    this.currentGameMetrics.admissionDecisions.push(shouldAdmit)
    this.currentGameMetrics.portfolioDeviations.push(
      this.calculatePortfolioDeviation()
    )

    // Update portfolio if admitting
    if (shouldAdmit) {
      Object.entries(status.nextPerson.attributes).forEach(
        ([attr, hasAttr]) => {
          if (hasAttr) {
            const asset = this.portfolio.get(attr as keyof ScenarioAttributes)
            if (asset) {
              asset.current += 1
            }
          }
        }
      )
    }

    return shouldAdmit
  }

  getProgress() {
    const quotas = Array.from(this.portfolio.values())
      .map((asset) => ({
        attribute: asset.attribute,
        needed: Math.max(0, asset.target - asset.current),
        current: asset.current,
        progress: asset.current / asset.target,
        weight: asset.weight,
        risk: asset.risk,
        sharpeRatio: asset.expectedReturn / Math.sqrt(asset.risk),
      }))
      .filter((q) => q.needed > 0)

    return {
      quotas: quotas.sort((a, b) => b.sharpeRatio - a.sharpeRatio),
      portfolioDeviation: this.calculatePortfolioDeviation(),
      riskTolerance: this.riskTolerance,
      needsRebalancing: this.shouldRebalance(),
      learningStats: {
        gamesPlayed: this.learningData.historicalPerformance.length,
        avgHistoricalScore:
          this.learningData.historicalPerformance.length > 0
            ? this.learningData.historicalPerformance.reduce(
                (sum, game) => sum + game.finalScore,
                0
              ) / this.learningData.historicalPerformance.length
            : 0,
        hasLearnedWeights:
          Object.keys(this.learningData.learnedWeights).length > 0,
      },
    }
  }

  /**
   * Automatically called when game finishes, final score is total rejections (lower is better).
   */
  getOutput() {
    const progress = this.getProgress()

    // Calculate final game result for learning
    const gameResult = {
      finalScore: this.currentGameMetrics.admissionDecisions.filter(
        (value) => value === false
      ).length,
      admissionRate:
        this.currentGameMetrics.admissionDecisions.filter(Boolean).length /
        Math.max(1, this.currentGameMetrics.admissionDecisions.length),
      quotaCompletion: Object.fromEntries(
        Array.from(this.portfolio.entries()).map(([attr, asset]) => [
          attr as string,
          asset.current / asset.target,
        ])
      ),
    }

    this.saveLearningData(gameResult).catch((e) => {
      console.warn('failed to save game data:', e)
    })

    return {
      ...progress,
      gameResult,
    }
  }
}
