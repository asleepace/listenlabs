import { Stats } from './statistics'

/**
 * For your current 80% admission rate problem, I'd recommend the combined approach initially,
 * then potentially switch to the exponential or high-sensitivity sigmoid once you're closer to target rates.
 * The modular design also lets you easily A/B test different deflation strategies or even blend them based on game phase (aggressive early, smooth later).
 * Which deflation function are you planning to use first?
 */

export interface ScoreDeflationParams {
  admittedCount: number
  rejectedCount: number
  targetRate: number
}

export interface FactorConfig {
  /**
   * Higher = more responsive to small deviations
   * @default 0.4
   */
  sensitivity: number
  /**
   * Minimum deflation factor (80% reduction)
   * @default 0.8
   */
  maxDeflation: number
  /**
   * Maximum inflation factor (80% boost)
   * @default  0.8
   */
  maxInflation: number
}

const defaultFactor: FactorConfig = {
  sensitivity: 0.4,
  maxDeflation: 0.8,
  maxInflation: 0.8,
}

function getCurrentRate(params: ScoreDeflationParams) {
  if (params.admittedCount === 0 && params.rejectedCount === 0) return 0
  return params.admittedCount / (params.admittedCount + params.rejectedCount)
}

/**
 *  Dynamic deflation factor that responds smoothly to admission rate deviations
 */
export function getScoreDeflationFactor(
  params: ScoreDeflationParams & Partial<FactorConfig>
): number {
  const currentRate = getCurrentRate(params)
  const deviation = currentRate - params.targetRate

  // Use sigmoid/tanh for smooth, responsive adjustment
  // Maps large deviations to strong deflation/inflation
  const sensitivity = params.sensitivity ?? defaultFactor.sensitivity // Higher = more responsive to small deviations
  const maxDeflation = params.maxDeflation ?? defaultFactor.maxDeflation // Minimum deflation factor (80% reduction)
  const maxInflation = params.maxInflation ?? defaultFactor.maxInflation // Maximum inflation factor (80% boost)

  // Sigmoid function: maps (-∞, +∞) to (0, 1)
  const normalizedDeviation = Math.tanh(deviation * sensitivity)

  // Map to deflation range
  if (normalizedDeviation > 0) {
    // Over target rate - deflate scores
    const deflationStrength = normalizedDeviation
    return 1 - deflationStrength * (1 - maxDeflation)
  } else {
    // Under target rate - inflate scores
    const inflationStrength = -normalizedDeviation
    return 1 + inflationStrength * (maxInflation - 1)
  }
}

/**
 *  Exponential decay model for more aggressive response
 */
export function getExponentialDeflationFactor(
  params: ScoreDeflationParams
): number {
  const currentRate = getCurrentRate(params)
  const targetRate = params.targetRate
  const ratio = currentRate / targetRate

  // Exponential response - more dramatic for larger deviations
  if (ratio > 1) {
    // Over target - exponential deflation
    // ratio=2 → 0.5x, ratio=3 → 0.33x, ratio=4 → 0.25x
    return Math.max(0.1, 1 / ratio)
  } else {
    // Under target - exponential inflation
    // ratio=0.5 → 1.4x, ratio=0.25 → 1.75x
    const boost = Math.min(2.0, 1 + (1 - ratio))
    return boost
  }
}

/**
 *  Power law model - balanced between linear and exponential
 */
export function getPowerLawDeflationFactor(
  params: ScoreDeflationParams
): number {
  const currentRate =
    params.admittedCount / (params.admittedCount + params.rejectedCount)
  const targetRate = 0.25
  const ratio = currentRate / targetRate

  // Power law with exponent between 0.5 and 2.0 for different response curves
  const exponent = 1.5

  if (ratio > 1) {
    // Over target - power law deflation
    return Math.max(0.15, Math.pow(1 / ratio, exponent))
  } else {
    // Under target - power law inflation
    return Math.min(1.8, Math.pow(1 / ratio, 1 / exponent))
  }
}

/**
 *  Adaptive sensitivity that increases over time
 */
export function getAdaptiveDeflationFactor(
  params: ScoreDeflationParams & {
    maxCapacity: number
    baseSensitivity: number
  }
): number {
  const currentRate = getCurrentRate(params)
  const targetRate = params.targetRate
  const deviation = currentRate - targetRate

  // Increase sensitivity as game progresses
  const gameProgress = params.admittedCount / params.maxCapacity
  const baseSensitivity = params.baseSensitivity ?? 0.3
  const adaptiveSensitivity = baseSensitivity * (1 + gameProgress * 2) // 3x to 9x sensitivity

  const normalizedDeviation = Math.tanh(deviation * adaptiveSensitivity)

  if (normalizedDeviation > 0) {
    // Over target - stronger deflation as game progresses
    const maxDeflation = Math.max(0.1, 0.4 - gameProgress * 0.3) // 0.4 → 0.1
    return 1 - normalizedDeviation * (1 - maxDeflation)
  } else {
    // Under target - inflation
    const maxInflation = 1.6
    return 1 + -normalizedDeviation * (maxInflation - 1)
  }
}

/**
 *  Combined approach with safety bounds
 */
export function getScoreDeflationFactorCombined(
  params: ScoreDeflationParams
): number {
  const currentRate =
    params.admittedCount / (params.admittedCount + params.rejectedCount)
  const targetRate = params.targetRate

  // For your current 80% rate vs 25% target:
  const ratio = currentRate / targetRate // 3.2

  // Aggressive response for large deviations
  let deflationFactor: number

  if (ratio > 2.5) {
    // Severe over-admission - emergency deflation
    deflationFactor = 0.15
  } else if (ratio > 2.0) {
    // Heavy over-admission - strong deflation
    deflationFactor = 0.25
  } else if (ratio > 1.5) {
    // Moderate over-admission - medium deflation
    deflationFactor = 0.5
  } else if (ratio > 1.2) {
    // Slight over-admission - light deflation
    deflationFactor = 0.8
  } else if (ratio < 0.8) {
    // Under-admission - inflation
    deflationFactor = Math.min(1.5, 1.25 / ratio)
  } else {
    // Target range - minimal adjustment
    deflationFactor = 1.0
  }

  // Safety bounds
  return Stats.clamp(deflationFactor, 0.1, 2.0)
}
