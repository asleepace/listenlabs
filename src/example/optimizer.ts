import type { GameState, ScenarioAttributes } from '../types'
import type { BergainBouncer } from '../berghain'
import { Bouncer } from '../bouncer' // Your original working algorithm

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

interface ParameterPerformance {
  parameterSet: ParameterSet
  runs: {
    scenario: string
    rejections: number
    quotaCompletion: number
    admissionRate: number
    timestamp: number
  }[]
  expectedReturn: number // 1 / average_rejections (higher is better)
  risk: number // standard deviation of performance
  sharpeRatio: number // risk-adjusted return
  weight: number // portfolio allocation
  confidence: number // how much data we have
}

interface LearningData {
  parameterHistory: ParameterPerformance[]
  scenarioInsights: Record<
    string,
    {
      bestParameters: string
      avgPerformance: number
      sampleSize: number
    }
  >
  globalOptimization: {
    explorationRate: number
    convergenceThreshold: number
    lastUpdate: number
  }
}

async function sleep(time: number) {
  const sleeper = Promise.withResolvers<void>()
  setTimeout(sleeper.resolve, time)
  return sleeper.promise
}

export class ParameterOptimizer implements BergainBouncer {
  static filePath = './learning-data/parameter-optimizer.json'

  private gameState: GameState
  private learningData: LearningData
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
        BASE_THRESHOLD: 0.41,
        TARGET_RATE: 0.22,
        URGENCY_MODIFIER: 3.5,
        MULTI_ATTRIBUTE_BONUS: 1.4,
        CRITICAL_REQUIRED_THRESHOLD: 20,
        CRITICAL_IN_LINE_RATIO: 0.75,
        CRITICAL_CAPACITY_RATIO: 0.87,
        MIN_THRESHOLD: 0.18,
        MAX_THRESHOLD: 0.82,
      },
    },
    {
      id: 'endgame_focused',
      name: 'Endgame Optimization',
      config: {
        BASE_THRESHOLD: 0.4,
        TARGET_RATE: 0.2,
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

  constructor(gameState: GameState) {
    this.gameState = gameState
    this.performanceTracking = {
      startTime: Date.now(),
      admissionDecisions: 0,
      currentRejections: 0,
    }

    // Initialize learning data synchronously with defaults
    this.learningData = {
      parameterHistory: [],
      scenarioInsights: {},
      globalOptimization: {
        explorationRate: 0.3,
        convergenceThreshold: 0.05,
        lastUpdate: Date.now(),
      },
    }

    this.selectParameterSet()
    this.createBouncer()

    // Load saved data asynchronously in background
    this.loadLearningDataAsync()
  }

  private async loadLearningDataAsync() {
    try {
      const learningFile = Bun.file('./learning-data/parameter-optimizer.json')

      if (await learningFile.exists()) {
        const loadedData = await learningFile.json()
        this.learningData = loadedData
        console.log(
          `Background loaded parameter learning data: ${
            loadedData.parameterHistory?.length || 0
          } parameter sets tested`
        )
      }
    } catch (error) {
      console.warn('Failed to background load parameter learning data:', error)
      // await this.initializeLearning()
    }
  }

  async initializeLearning() {
    try {
      const learningFile = Bun.file('./learning-data/parameter-optimizer.json')

      if (await learningFile.exists()) {
        this.learningData = await learningFile.json()
        console.log(
          `Loaded parameter learning data: ${this.learningData.parameterHistory.length} parameter sets tested`
        )
      } else {
        this.learningData = {
          parameterHistory: [],
          scenarioInsights: {},
          globalOptimization: {
            explorationRate: 0.3,
            convergenceThreshold: 0.05,
            lastUpdate: Date.now(),
          },
        }
        await learningFile.write(JSON.stringify(this.learningData, null, 2))
      }
    } catch (error) {
      console.warn('Failed to load parameter learning data:', error)
      this.learningData = {
        parameterHistory: [],
        scenarioInsights: {},
        globalOptimization: {
          explorationRate: 0.3,
          convergenceThreshold: 0.05,
          lastUpdate: Date.now(),
        },
      }
    }
    // allow user to see this data quickly
    await sleep(5_000)
  }

  selectParameterSet() {
    const scenarioId = this.gameState.game.gameId || 'unknown'

    // Check if we have scenario-specific insights
    const scenarioInsight = this.learningData.scenarioInsights[scenarioId]
    if (scenarioInsight && scenarioInsight.sampleSize >= 3) {
      // Use best known parameters for this scenario with some exploration
      if (
        Math.random() > this.learningData.globalOptimization.explorationRate
      ) {
        const bestParams = this.parameterSpace.find(
          (p) => p.id === scenarioInsight.bestParameters
        )
        if (bestParams) {
          this.currentParameterSet = bestParams
          console.log(
            `Using learned best parameters for scenario: ${bestParams.name}`
          )
          return
        }
      }
    }

    // Multi-armed bandit selection using Thompson sampling
    this.currentParameterSet = this.thompsonSampling()
    console.log(`Selected parameter set: ${this.currentParameterSet.name}`)
  }

  private thompsonSampling(): ParameterSet {
    // Calculate performance metrics for each parameter set
    const candidates = this.parameterSpace.map((paramSet) => {
      const performance = this.learningData.parameterHistory.find(
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

  createBouncer() {
    // Create your original Bouncer with selected parameters
    const overrides = this.currentParameterSet.config
    this.currentBouncer = Bouncer.intialize(overrides)(this.gameState)
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

    return {
      ...bouncerProgress,
      parameterOptimization: {
        currentParameterSet: this.currentParameterSet.name,
        currentRejections: this.performanceTracking.currentRejections,
        decisions: this.performanceTracking.admissionDecisions,
        explorationRate: this.learningData.globalOptimization.explorationRate,
        portfolioInsights: this.getPortfolioInsights(),
      },
    }
  }

  private getPortfolioInsights() {
    return this.learningData.parameterHistory
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

    // Save performance data
    await this.saveParameterPerformance({
      scenario: this.gameState.game.gameId || 'unknown',
      rejections: finalRejections,
      quotaCompletion,
      admissionRate,
      timestamp: Date.now(),
    })

    return {
      ...bouncerOutput,
      parameterOptimization: {
        finalRejections,
        parameterSet: this.currentParameterSet.name,
        learningInsights: this.getPortfolioInsights(),
        nextRecommendation: await this.getNextParameterRecommendation(),
      },
    }
  }

  private async saveParameterPerformance(runData: {
    scenario: string
    rejections: number
    quotaCompletion: number
    admissionRate: number
    timestamp: number
  }) {
    try {
      // Find or create parameter performance record
      let paramPerf = this.learningData.parameterHistory.find(
        (p) => p.parameterSet.id === this.currentParameterSet.id
      )

      if (!paramPerf) {
        paramPerf = {
          parameterSet: this.currentParameterSet,
          runs: [],
          expectedReturn: 0,
          risk: 0,
          sharpeRatio: 0,
          weight: 1 / this.parameterSpace.length,
          confidence: 0,
        }
        this.learningData.parameterHistory.push(paramPerf)
      }

      // Add this run's data
      paramPerf.runs.push(runData)

      // Keep only recent runs (last 20 per parameter set)
      if (paramPerf.runs.length > 20) {
        paramPerf.runs = paramPerf.runs.slice(-20)
      }

      // Update performance metrics
      this.updateParameterMetrics(paramPerf)

      // Update scenario insights
      this.updateScenarioInsights(runData.scenario)

      // Rebalance portfolio weights
      this.rebalanceParameterPortfolio()

      // Save to file
      await Bun.write(
        './learning-data/parameter-optimizer.json',
        JSON.stringify(this.learningData, null, 2)
      )

      console.log(
        `Parameter performance saved: ${runData.rejections} rejections with ${this.currentParameterSet.name}`
      )
    } catch (error) {
      console.warn('Failed to save parameter performance:', error)
    }
  }

  private updateParameterMetrics(paramPerf: ParameterPerformance) {
    const rejections = paramPerf.runs.map((run) => run.rejections)
    const avgRejections =
      rejections.reduce((sum, r) => sum + r, 0) / rejections.length

    // Expected return = 1 / average_rejections (higher is better)
    paramPerf.expectedReturn = 1 / Math.max(avgRejections, 1)

    // Risk = standard deviation of rejections
    const variance =
      rejections.reduce((sum, r) => sum + Math.pow(r - avgRejections, 2), 0) /
      rejections.length
    paramPerf.risk = Math.sqrt(variance)

    // Sharpe ratio = risk-adjusted return
    paramPerf.sharpeRatio =
      paramPerf.risk > 0
        ? paramPerf.expectedReturn / paramPerf.risk
        : paramPerf.expectedReturn

    // Confidence based on sample size
    paramPerf.confidence = Math.min(paramPerf.runs.length / 10, 1.0)
  }

  private updateScenarioInsights(scenarioId: string) {
    const scenarioRuns = this.learningData.parameterHistory.flatMap((p) =>
      p.runs
        .filter((r) => r.scenario === scenarioId)
        .map((r) => ({ ...r, paramId: p.parameterSet.id }))
    )

    if (scenarioRuns.length === 0) return

    // Find best performing parameter set for this scenario
    const paramGrouped = scenarioRuns.reduce((acc, run) => {
      if (!acc[run.paramId]) acc[run.paramId] = []
      acc[run.paramId]!.push(run.rejections)
      return acc
    }, {} as Record<string, number[]>)

    let bestParamId = ''
    let bestAvgRejections = Infinity

    Object.entries(paramGrouped).forEach(([paramId, rejections]) => {
      const avgRejections =
        rejections.reduce((sum, r) => sum + r, 0) / rejections.length
      if (avgRejections < bestAvgRejections) {
        bestAvgRejections = avgRejections
        bestParamId = paramId
      }
    })

    this.learningData.scenarioInsights[scenarioId] = {
      bestParameters: bestParamId,
      avgPerformance: bestAvgRejections,
      sampleSize: scenarioRuns.length,
    }
  }

  private rebalanceParameterPortfolio() {
    // Update portfolio weights based on Sharpe ratios
    const totalSharpe = this.learningData.parameterHistory.reduce(
      (sum, p) => sum + Math.max(p.sharpeRatio, 0),
      0
    )

    if (totalSharpe > 0) {
      this.learningData.parameterHistory.forEach((p) => {
        p.weight = Math.max(p.sharpeRatio, 0) / totalSharpe
      })
    } else {
      // Equal weights if no performance data
      const equalWeight = 1 / this.learningData.parameterHistory.length
      this.learningData.parameterHistory.forEach((p) => {
        p.weight = equalWeight
      })
    }

    // Reduce exploration rate as we gather more data
    const totalRuns = this.learningData.parameterHistory.reduce(
      (sum, p) => sum + p.runs.length,
      0
    )
    this.learningData.globalOptimization.explorationRate = Math.max(
      0.1,
      0.5 - totalRuns / 100
    )
  }

  private async getNextParameterRecommendation(): Promise<string> {
    const insights = this.getPortfolioInsights()
    const best = insights[0]

    if (!best || best.sampleSize < 3) {
      return 'Insufficient data - continue exploring parameter space'
    }

    const currentPerformance = this.performanceTracking.currentRejections
    const bestAvgRejections = best.avgRejections || Infinity

    if (currentPerformance < bestAvgRejections) {
      return `Current run outperforming best known (${best.name}). Consider this parameter set for future runs.`
    } else {
      return `Best known: ${best.name} (avg: ${Math.round(
        best.avgRejections || 0
      )} rejections). Consider switching if current performance plateaus.`
    }
  }
}
