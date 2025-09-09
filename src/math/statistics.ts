/**
 *  A collection of useful statistical methods and helpers for computing math.
 */
export namespace Stats {
  /**
   * Calculate the average (mean) value of an array.
   */
  export function average(items: number[]): number {
    if (!items || items.length === 0) return 0
    return items.reduce((total, current) => total + current, 0) / items.length
  }

  /**
   * Calculate the actual median value (middle value when sorted).
   */
  export function median(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    const sorted = [...nums].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    return sorted.length % 2 === 0
      ? (sorted[mid - 1]! + sorted[mid]!) / 2
      : sorted[mid]!
  }

  /**
   * Find the mode (most frequent value) in an array.
   */
  export function mode(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    const frequency: Record<number, number> = {}
    let maxCount = 0
    let modeValue = nums[0]

    nums.forEach((num) => {
      frequency[num] = (frequency[num] || 0) + 1
      if (frequency[num]! > maxCount) {
        maxCount = frequency[num]!
        modeValue = num
      }
    })

    return modeValue!
  }

  /**
   * Calculate standard deviation.
   */
  export function stdDev(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    const avg = Stats.average(nums)
    const variance =
      nums.reduce((sum, score) => sum + Math.pow(score - avg, 2), 0) /
      nums.length
    return Math.sqrt(variance)
  }

  /**
   * Calculate variance.
   */
  export function variance(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    const avg = Stats.average(nums)
    return (
      nums.reduce((sum, score) => sum + Math.pow(score - avg, 2), 0) /
      nums.length
    )
  }

  /**
   * Find minimum value in array.
   */
  export function min(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    return Math.min(...nums)
  }

  /**
   * Find maximum value in array.
   */
  export function max(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    return Math.max(...nums)
  }

  /**
   * Calculate range (max - min).
   */
  export function range(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    return Stats.max(nums) - Stats.min(nums)
  }

  /**
   * Calculate percentile (e.g., 0.9 for 90th percentile).
   */
  export function percentile(nums: number[], p: number): number {
    if (!nums || nums.length === 0) return 0
    const sorted = [...nums].sort((a, b) => a - b)
    const index = Math.ceil(sorted.length * p) - 1
    return sorted[Math.max(0, Math.min(index, sorted.length - 1))]!
  }

  /**
   * Calculate quartiles [Q1, Q2 (median), Q3].
   */
  export function quartiles(nums: number[]): [number, number, number] {
    return [
      Stats.percentile(nums, 0.25),
      Stats.median(nums),
      Stats.percentile(nums, 0.75),
    ]
  }

  /**
   * Calculate z-score for a value relative to an array.
   */
  export function zScore(value: number, nums: number[]): number {
    const avg = Stats.average(nums)
    const std = Stats.stdDev(nums)
    return std === 0 ? 0 : (value - avg) / std
  }

  /**
   * Sum all values in array.
   */
  export function sum(nums: number[]): number {
    if (!nums || nums.length === 0) return 0
    return nums.reduce((total, current) => total + current, 0)
  }

  /**
   * Round the number to the specified places.
   */
  export function round(num: number, places = 100): number {
    return Math.round(num * places) / places
  }

  /**
   * Convert a decimal to a percentage with specified decimal places.
   */
  export function percent(num: number, decimalPlaces = 2): number {
    const multiplier = Math.pow(10, decimalPlaces + 2) // +2 for percentage
    return Math.round(num * multiplier) / Math.pow(10, decimalPlaces)
  }

  /**
   * Clamp a number between min and max values.
   */
  export function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(value, max))
  }

  /**
   * Linear interpolation between two values.
   */
  export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t
  }

  /**
   * Calculate the sigmoid function.
   */
  export function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  /**
   * Calculate the tanh-based sigmoid used in your threshold calculation.
   */
  export function tanhSigmoid(ratio: number): number {
    return (1.0 + Math.tanh(1.0 - ratio)) / 2.0
  }

  /**
   * Normalize an array to 0-1 range.
   */
  export function normalize(nums: number[]): number[] {
    if (!nums || nums.length === 0) return []
    const minVal = Stats.min(nums)
    const maxVal = Stats.max(nums)
    const range = maxVal - minVal

    if (range === 0) return nums.map(() => 0.5)
    return nums.map((num) => (num - minVal) / range)
  }

  /**
   * Calculate correlation coefficient between two arrays.
   */
  export function correlation(x: number[], y: number[]): number {
    if (!x || !y || x.length !== y.length || x.length === 0) return 0

    const n = x.length
    const sumX = Stats.sum(x)
    const sumY = Stats.sum(y)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i]!, 0)
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0)
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0)

    const numerator = n * sumXY - sumX * sumY
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
    )

    return denominator === 0 ? 0 : numerator / denominator
  }

  /**
   * Calculate moving average with specified window size.
   */
  export function movingAverage(nums: number[], windowSize: number): number[] {
    if (!nums || nums.length === 0 || windowSize <= 0) return []

    const result: number[] = []
    for (let i = 0; i < nums.length; i++) {
      const start = Math.max(0, i - windowSize + 1)
      const window = nums.slice(start, i + 1)
      result.push(Stats.average(window))
    }
    return result
  }

  /**
   * Check if a number is within a tolerance of another number.
   */
  export function isNear(a: number, b: number, tolerance = 1e-10): boolean {
    return Math.abs(a - b) <= tolerance
  }

  /**
   * Get basic descriptive statistics for an array.
   */
  export function describe(nums: number[]) {
    if (!nums || nums.length === 0) {
      return {
        count: 0,
        mean: 0,
        median: 0,
        mode: 0,
        min: 0,
        max: 0,
        range: 0,
        stdDev: 0,
        variance: 0,
        quartiles: [0, 0, 0] as [number, number, number],
      }
    }

    return {
      count: nums.length,
      mean: Stats.average(nums),
      median: Stats.median(nums),
      mode: Stats.mode(nums),
      min: Stats.min(nums),
      max: Stats.max(nums),
      range: Stats.range(nums),
      stdDev: Stats.stdDev(nums),
      variance: Stats.variance(nums),
      quartiles: Stats.quartiles(nums),
    }
  }
}
