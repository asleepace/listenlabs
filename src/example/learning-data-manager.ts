interface ParameterSet {
  id: string
  name: string
  config: Record<string, number>
}

interface ParameterRun {
  scenario: string
  rejections: number
  quotaCompletion: number
  admissionRate: number
  timestamp: number
}

interface ParameterPerformance {
  parameterSet: ParameterSet
  runs: ParameterRun[]
  expectedReturn: number
  risk: number
  sharpeRatio: number
  weight: number
  confidence: number
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

export class LearningDataManager {
  private data: LearningData
  private filePath: string

  constructor(filePath: string) {
    this.filePath = filePath
    this.data = this.getDefaultData()
  }

  private getDefaultData(): LearningData {
    return {
      parameterHistory: [],
      scenarioInsights: {},
      globalOptimization: {
        explorationRate: 0.3,
        convergenceThreshold: 0.05,
        lastUpdate: Date.now(),
      },
    }
  }

  /**
   * Load learning data from disk (async)
   */
  async load(): Promise<LearningData> {
    try {
      console.log(`[learning] Attempting to load from ${this.filePath}`)
      const file = Bun.file(this.filePath)

      if (await file.exists()) {
        const loadedData = await file.json()
        this.data = { ...this.getDefaultData(), ...loadedData }
        console.log(
          `[learning] Loaded data: ${
            this.data.parameterHistory.length
          } parameter sets, ${this.getTotalRuns()} total runs`
        )
        return this.data
      } else {
        console.log('[learning] No existing data file found, using defaults')
        return this.data
      }
    } catch (error) {
      console.warn('[learning] Failed to load data, using defaults:', error)
      throw error
    }
  }

  /**
   * Save learning data to disk (async)
   */
  async save(): Promise<void> {
    try {
      console.log('[learning] saving data!')
      // Ensure directory exists
      // const dir = this.filePath.substring(0, this.filePath.lastIndexOf('/'))
      // if (dir) {
      //   await Bun.write(`${dir}/.keep`, '') // Create directory if it doesn't exist
      // }
      await Bun.write(this.filePath, JSON.stringify(this.data, null, 2))
      console.log(
        `[learning] Saved data: ${
          this.data.parameterHistory.length
        } parameter sets, ${this.getTotalRuns()} total runs`
      )
    } catch (error) {
      console.error('[learning] Failed to save data:', error)
      throw error
    }
  }

  /**
   * Get current learning data (synchronous)
   */
  getData(): LearningData {
    return this.data
  }

  /**
   * Add a new run result for a parameter set
   */
  addRun(
    parameterSetId: string,
    parameterSet: ParameterSet,
    runData: ParameterRun
  ): void {
    console.log(
      `[learning] Adding run: ${parameterSet.name} -> ${runData.rejections} rejections`
    )

    // Find or create parameter performance record
    let paramPerf = this.data.parameterHistory.find(
      (p) => p.parameterSet.id === parameterSetId
    )

    if (!paramPerf) {
      paramPerf = {
        parameterSet,
        runs: [],
        expectedReturn: 0,
        risk: 0,
        sharpeRatio: 0,
        weight: 0,
        confidence: 0,
      }
      this.data.parameterHistory.push(paramPerf)
      console.log(
        `[learning] Created new parameter record: ${parameterSet.name}`
      )
    }

    // Add run data
    paramPerf.runs.push(runData)

    // Keep only recent runs (last 20 per parameter set)
    if (paramPerf.runs.length > 20) {
      paramPerf.runs = paramPerf.runs.slice(-20)
    }

    // Update metrics
    this.updateParameterMetrics(paramPerf)
    this.updateScenarioInsights(runData.scenario)
    this.rebalancePortfolio()

    console.log(
      `[learning] Updated metrics: expectedReturn=${paramPerf.expectedReturn.toFixed(
        4
      )}, sharpeRatio=${paramPerf.sharpeRatio.toFixed(4)}`
    )
  }

  /**
   * Get best parameter set for a specific scenario
   */
  getBestParameterForScenario(scenarioId: string): string | null {
    const insight = this.data.scenarioInsights[scenarioId]
    return insight && insight.sampleSize >= 2 ? insight.bestParameters : null
  }

  /**
   * Get parameter weights for Thompson sampling
   */
  getParameterWeights(): Record<string, number> {
    const weights: Record<string, number> = {}
    this.data.parameterHistory.forEach((p) => {
      weights[p.parameterSet.id] = p.weight
    })
    return weights
  }

  /**
   * Get exploration rate
   */
  getExplorationRate(): number {
    return this.data.globalOptimization.explorationRate
  }

  /**
   * Get parameter performance for analysis
   */
  getParameterPerformance(): ParameterPerformance[] {
    return this.data.parameterHistory
  }

  /**
   * Get summary statistics
   */
  getSummary(): {
    totalParameterSets: number
    totalRuns: number
    bestOverallParameter: string | null
    avgPerformance: number
    explorationRate: number
  } {
    const totalRuns = this.getTotalRuns()
    const best = this.data.parameterHistory
      .filter((p) => p.runs.length > 0)
      .sort((a, b) => b.sharpeRatio - a.sharpeRatio)[0]

    const avgRejections = this.data.parameterHistory
      .flatMap((p) => p.runs.map((r) => r.rejections))
      .reduce((sum, r, _, arr) => sum + r / arr.length, 0)

    return {
      totalParameterSets: this.data.parameterHistory.length,
      totalRuns,
      bestOverallParameter: best?.parameterSet.name || null,
      avgPerformance: avgRejections,
      explorationRate: this.data.globalOptimization.explorationRate,
    }
  }

  private getTotalRuns(): number {
    return this.data.parameterHistory.reduce((sum, p) => sum + p.runs.length, 0)
  }

  private updateParameterMetrics(paramPerf: ParameterPerformance): void {
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

  private updateScenarioInsights(scenarioId: string): void {
    const scenarioRuns = this.data.parameterHistory.flatMap((p) =>
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
      if (rejections.length >= 1) {
        // Require at least 1 run
        const avgRejections =
          rejections.reduce((sum, r) => sum + r, 0) / rejections.length
        if (avgRejections < bestAvgRejections) {
          bestAvgRejections = avgRejections
          bestParamId = paramId
        }
      }
    })

    if (bestParamId) {
      this.data.scenarioInsights[scenarioId] = {
        bestParameters: bestParamId,
        avgPerformance: bestAvgRejections,
        sampleSize: scenarioRuns.length,
      }
    }
  }

  private rebalancePortfolio(): void {
    if (this.data.parameterHistory.length === 0) return

    // Update portfolio weights based on Sharpe ratios
    const totalSharpe = this.data.parameterHistory.reduce(
      (sum, p) => sum + Math.max(p.sharpeRatio, 0),
      0
    )

    if (totalSharpe > 0) {
      this.data.parameterHistory.forEach((p) => {
        p.weight = Math.max(p.sharpeRatio, 0) / totalSharpe
      })
    } else {
      // Equal weights if no performance data
      const equalWeight = 1 / this.data.parameterHistory.length
      this.data.parameterHistory.forEach((p) => {
        p.weight = equalWeight
      })
    }

    // Reduce exploration rate as we gather more data
    const totalRuns = this.getTotalRuns()
    this.data.globalOptimization.explorationRate = Math.max(
      0.1,
      0.5 - totalRuns / 100
    )
    this.data.globalOptimization.lastUpdate = Date.now()
  }
}
