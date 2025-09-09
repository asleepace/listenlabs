import type { GameState, ScenarioAttributes } from './types'
import type { BergainBouncer } from './berghain'
import { Bouncer } from './bouncer' // Your original working algorithm
import { LearningDataManager } from './example/learning-data-manager'

interface ParameterSet {
  id: string
  name: string
  config: {
    BASE_THRESHOLD: number
    TARGET_RATE: number
    URGENCY_MODIFIER: number
    MULTI_ATTRIBUTE_BONUS: number
    CRITICAL_REQUIRED_THRESHOLD: number
    CRITICAL_IN_LINE_RATIO: number
    CRITICAL_CAPACITY_RATIO: number
    MIN_THRESHOLD: number
    MAX_THRESHOLD: number
  }
}

export class ParameterOptimizer implements BergainBouncer {
  private gameState: GameState
  private learningManager: LearningDataManager
  private currentParameterSet!: ParameterSet
  private currentBouncer!: Bouncer<ScenarioAttributes>
  private performanceTracking: {
    startTime: number
    admissionDecisions: number
    currentRejections: number
  }

  // Parameter space to explore
  private parameterSpace: ParameterSet[] = [
    {
      id: 'conservative',
      name: 'Conservative Baseline',
      config: {
        BASE_THRESHOLD: 0.45,
        TARGET_RATE: 0.19,
        URGENCY_MODIFIER: 2.5,
        MULTI_ATTRIBUTE_BONUS: 1.2,
        CRITICAL_REQUIRED_THRESHOLD: 25,
        CRITICAL_IN_LINE_RATIO: 0.8,
        CRITICAL_CAPACITY_RATIO: 0.9,
        MIN_THRESHOLD: 0.2,
        MAX_THRESHOLD: 0.8,
      },
    },
    {
      id: 'aggressive',
      name: 'Aggressive Rare Targeting',
      config: {
        BASE_THRESHOLD: 0.35,
        TARGET_RATE: 0.21,
        URGENCY_MODIFIER: 4.0,
        MULTI_ATTRIBUTE_BONUS: 1.6,
        CRITICAL_REQUIRED_THRESHOLD: 15,
        CRITICAL_IN_LINE_RATIO: 0.7,
        CRITICAL_CAPACITY_RATIO: 0.85,
        MIN_THRESHOLD: 0.15,
        MAX_THRESHOLD: 0.85,
      },
    },
    {
      id: 'balanced',
      name: 'Balanced Approach',
      config: {
        BASE_THRESHOLD: 0.4,
        TARGET_RATE: 0.2,
        URGENCY_MODIFIER: 3.5,
        MULTI_ATTRIBUTE_BONUS: 1.5,
        CRITICAL_REQUIRED_THRESHOLD: 50,
        CRITICAL_IN_LINE_RATIO: 0.75,
        CRITICAL_CAPACITY_RATIO: 0.87,
        MIN_THRESHOLD: 0.2,
        MAX_THRESHOLD: 0.9,
      },
    },
    {
      id: 'endgame_focused',
      name: 'Endgame Optimization',
      config: {
        BASE_THRESHOLD: 0.4,
        TARGET_RATE: 0.19,
        URGENCY_MODIFIER: 3.0,
        MULTI_ATTRIBUTE_BONUS: 1.8,
        CRITICAL_REQUIRED_THRESHOLD: 10,
        CRITICAL_IN_LINE_RATIO: 0.65,
        CRITICAL_CAPACITY_RATIO: 0.8,
        MIN_THRESHOLD: 0.1,
        MAX_THRESHOLD: 0.9,
      },
    },
  ]

  constructor(gameState: GameState, learningManager: LearningDataManager) {
    this.gameState = gameState
    this.learningManager = learningManager
    this.performanceTracking = {
      startTime: Date.now(),
      admissionDecisions: 0,
      currentRejections: 0,
    }

    this.selectParameterSet()
    this.createBouncer()
  }

  private selectParameterSet() {
    const scenarioId = this.gameState.game.gameId || 'unknown'

    console.log(`[optimizer] Selecting parameters for scenario: ${scenarioId}`)

    // Check if we have scenario-specific insights
    const bestParamId =
      this.learningManager.getBestParameterForScenario(scenarioId)
    const explorationRate = this.learningManager.getExplorationRate()

    if (bestParamId && Math.random() > explorationRate) {
      // Use best known parameters for this scenario
      const bestParams = this.parameterSpace.find((p) => p.id === bestParamId)
      if (bestParams) {
        this.currentParameterSet = bestParams
        console.log(
          `[optimizer] Using learned best parameters: ${
            bestParams.name
          } (exploration rate: ${explorationRate.toFixed(2)})`
        )
        return
      }
    }

    // Multi-armed bandit selection using Thompson sampling
    this.currentParameterSet = this.thompsonSampling()
    console.log(
      `[optimizer] Exploring parameter set: ${this.currentParameterSet.name}`
    )
  }

  private thompsonSampling(): ParameterSet {
    const performanceData = this.learningManager.getParameterPerformance()

    // Calculate performance metrics for each parameter set
    const candidates = this.parameterSpace.map((paramSet) => {
      const performance = performanceData.find(
        (p) => p.parameterSet.id === paramSet.id
      )

      if (!performance || performance.runs.length === 0) {
        // No data - high exploration value
        return {
          paramSet,
          sampledValue: Math.random() * 2, // High variance for exploration
        }
      }

      // Calculate beta distribution parameters for Thompson sampling
      const avgReturn = performance.expectedReturn
      const confidence = Math.min(performance.runs.length / 10, 1.0) // More data = more confidence
      const variance = performance.risk

      // Sample from estimated performance distribution
      const sampledValue =
        avgReturn + (Math.random() - 0.5) * variance * (1 - confidence)

      return { paramSet, sampledValue }
    })

    // Select parameter set with highest sampled value
    candidates.sort((a, b) => b.sampledValue - a.sampledValue)
    return candidates[0]!.paramSet
  }

  private createBouncer() {
    // Create your original Bouncer with selected parameters
    const overrides = this.currentParameterSet.config
    this.currentBouncer = new Bouncer(this.gameState, overrides)
    console.log(
      `[optimizer] Created bouncer with ${this.currentParameterSet.name} parameters`
    )
  }

  admit(status: GameState['status']): boolean {
    this.performanceTracking.admissionDecisions++

    // Delegate to the underlying bouncer with optimized parameters
    const decision = this.currentBouncer.admit(status)

    // Track rejections for performance measurement
    if (!decision) {
      this.performanceTracking.currentRejections++
    }

    return decision
  }

  getProgress() {
    const bouncerProgress = this.currentBouncer.getProgress()
    const summary = this.learningManager.getSummary()

    return {
      ...bouncerProgress,
      parameterOptimization: {
        currentParameterSet: this.currentParameterSet.name,
        currentRejections: this.performanceTracking.currentRejections,
        decisions: this.performanceTracking.admissionDecisions,
        learningSummary: summary,
        portfolioInsights: this.getPortfolioInsights(),
      },
    }
  }

  private getPortfolioInsights() {
    return this.learningManager
      .getParameterPerformance()
      .map((perf) => ({
        name: perf.parameterSet.name,
        avgRejections:
          perf.runs.length > 0
            ? perf.runs.reduce((sum, run) => sum + run.rejections, 0) /
              perf.runs.length
            : null,
        sharpeRatio: perf.sharpeRatio,
        confidence: perf.confidence,
        weight: perf.weight,
        sampleSize: perf.runs.length,
      }))
      .sort((a, b) => (b.sharpeRatio || 0) - (a.sharpeRatio || 0))
  }

  async save() {
    try {
      console.log('[parameter-optimizer] saving!')
      const finalRejections = this.performanceTracking.currentRejections
      const admissionRate =
        (this.performanceTracking.admissionDecisions - finalRejections) /
        this.performanceTracking.admissionDecisions
      const bouncerOutput = this.currentBouncer.getOutput()

      // calculate complations
      const quotaCompletion =
        bouncerOutput.quotas?.length > 0
          ? bouncerOutput.quotas.reduce(
              (sum: number, quota: any) => sum + quota.progress,
              0
            ) / bouncerOutput.quotas.length
          : 1.0
      this.learningManager.addRun(
        this.currentParameterSet.id,
        this.currentParameterSet,
        {
          scenario: this.gameState.game.gameId || 'unknown',
          rejections: finalRejections,
          quotaCompletion,
          admissionRate,
          timestamp: Date.now(),
        }
      )

      await this.learningManager
        .save()
        .catch((e) => console.warn('[error] saving:', e))
      console.log('[optimizer] Successfully saved performance data')
    } catch (error) {
      console.error('[optimizer] Failed to save performance data:', error)
    }
  }

  async getOutput() {
    const bouncerOutput = this.currentBouncer.getOutput()

    // Calculate final performance metrics
    const finalRejections = this.performanceTracking.currentRejections
    const admissionRate =
      (this.performanceTracking.admissionDecisions - finalRejections) /
      this.performanceTracking.admissionDecisions

    // Calculate quota completion rate
    const quotaCompletion =
      bouncerOutput.quotas?.length > 0
        ? bouncerOutput.quotas.reduce(
            (sum: number, quota: any) => sum + quota.progress,
            0
          ) / bouncerOutput.quotas.length
        : 1.0

    console.log('[optimizer] Final performance:', {
      rejections: finalRejections,
      admissionRate: Math.round(admissionRate * 1000) / 1000,
      quotaCompletion: Math.round(quotaCompletion * 1000) / 1000,
      parameterSet: this.currentParameterSet.name,
    })

    // Save performance data via learning manager
    try {
      this.learningManager.addRun(
        this.currentParameterSet.id,
        this.currentParameterSet,
        {
          scenario: this.gameState.game.gameId || 'unknown',
          rejections: finalRejections,
          quotaCompletion,
          admissionRate,
          timestamp: Date.now(),
        }
      )

      await this.learningManager.save()
      console.log('[optimizer] Successfully saved performance data')
    } catch (error) {
      console.error('[optimizer] Failed to save performance data:', error)
    }

    const summary = this.learningManager.getSummary()

    return {
      ...bouncerOutput,
      parameterOptimization: {
        finalRejections,
        parameterSet: this.currentParameterSet.name,
        learningInsights: this.getPortfolioInsights(),
        learningSummary: summary,
        nextRecommendation: this.getNextParameterRecommendation(summary),
      },
    }
  }

  private getNextParameterRecommendation(summary: any): string {
    if (summary.totalRuns < 5) {
      return 'Continue exploring parameter space to gather baseline data'
    }

    const insights = this.getPortfolioInsights()
    const best = insights[0]

    if (!best || best.sampleSize < 3) {
      return 'Insufficient data for specific recommendations - continue exploration'
    }

    const currentPerformance = this.performanceTracking.currentRejections
    const bestAvgRejections = best.avgRejections || Infinity

    if (currentPerformance < bestAvgRejections) {
      return `Current run (${currentPerformance}) outperforming best known (${
        best.name
      }: ${Math.round(
        best.avgRejections || 0
      )}). This parameter set shows promise.`
    } else {
      return `Best known: ${best.name} (avg: ${Math.round(
        best.avgRejections || 0
      )} rejections, ${
        best.sampleSize
      } runs). Consider parameters with high Sharpe ratios for consistent performance.`
    }
  }
}
