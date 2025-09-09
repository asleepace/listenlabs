import { Stats } from './statistics'

export interface PIDConfig {
  kP: number // Proportional gain - how strongly to react to current error
  kI: number // Integral gain - how strongly to react to accumulated error
  kD: number // Derivative gain - how strongly to react to rate of change
  minOutput: number
  maxOutput: number
}

const DEFAULT_PID_CONFIG: PIDConfig = {
  kP: 2.5, // Strong proportional response
  kI: 0.15, // Moderate integral to prevent steady-state error
  kD: 0.8, // Moderate derivative to smooth oscillations
  minOutput: 0.1,
  maxOutput: 2.0,
}

export class DeflationPIDController {
  private integralError = 0
  private lastError = 0
  private lastTime = Date.now()

  constructor(private config: PIDConfig = DEFAULT_PID_CONFIG) {}

  /**
   * Calculate deflation factor using PID control
   */
  getDeflationFactor(
    currentRate: number,
    targetRate: number,
    options: {
      reset?: boolean
      deltaTimeMs?: number
    } = {}
  ): number {
    const currentTime = Date.now()
    const deltaTime = options.deltaTimeMs || currentTime - this.lastTime
    const deltaTimeSeconds = Math.max(deltaTime / 1000, 0.01) // Prevent division by zero

    if (options.reset) {
      this.integralError = 0
      this.lastError = 0
    }

    // Error is positive when over target, negative when under
    const error = currentRate - targetRate

    // Proportional term - immediate response to current error
    const proportional = this.config.kP * error

    // Integral term - accumulated error over time (prevents steady-state error)
    this.integralError += error * deltaTimeSeconds
    // Prevent integral windup
    this.integralError = Stats.clamp(this.integralError, -5.0, 5.0)
    const integral = this.config.kI * this.integralError

    // Derivative term - rate of change (smooths oscillations)
    const derivative =
      (this.config.kD * (error - this.lastError)) / deltaTimeSeconds

    // PID output (positive means we need to deflate more)
    const pidOutput = proportional + integral + derivative

    // Convert PID output to deflation factor
    // When pidOutput is positive (over target), deflation < 1
    // When pidOutput is negative (under target), deflation > 1
    const deflationFactor = 1.0 - pidOutput * 0.2 // Scaling factor

    // Update state for next iteration
    this.lastError = error
    this.lastTime = currentTime

    // Clamp to safe bounds
    return Stats.clamp(
      deflationFactor,
      this.config.minOutput,
      this.config.maxOutput
    )
  }

  /**
   * Get debug information about controller state
   */
  getDebugInfo(
    currentRate: number,
    targetRate: number
  ): {
    error: number
    integralError: number
    lastError: number
    proportionalTerm: number
    integralTerm: number
    derivativeTerm: number
  } {
    const error = currentRate - targetRate
    return {
      error: Stats.round(error, 10000),
      integralError: Stats.round(this.integralError, 10000),
      lastError: Stats.round(this.lastError, 10000),
      proportionalTerm: Stats.round(this.config.kP * error, 10000),
      integralTerm: Stats.round(this.config.kI * this.integralError, 10000),
      derivativeTerm: Stats.round(
        this.config.kD * (error - this.lastError),
        10000
      ),
    }
  }

  /**
   * Reset controller state (useful when starting new game or major changes)
   */
  reset(): void {
    this.integralError = 0
    this.lastError = 0
    this.lastTime = Date.now()
  }

  /**
   * Tune PID parameters during runtime
   */
  updateConfig(newConfig: Partial<PIDConfig>): void {
    this.config = { ...this.config, ...newConfig }
  }
}

// Preset configurations for different scenarios
export const PID_PRESETS = {
  CONSERVATIVE: {
    kP: 1.8,
    kI: 0.08,
    kD: 0.6,
    minOutput: 0.2,
    maxOutput: 1.5,
  } as PIDConfig,

  AGGRESSIVE: {
    kP: 3.5,
    kI: 0.25,
    kD: 1.2,
    minOutput: 0.05,
    maxOutput: 2.5,
  } as PIDConfig,

  SMOOTH: {
    kP: 2.0,
    kI: 0.1,
    kD: 1.5, // High derivative for smooth response
    minOutput: 0.15,
    maxOutput: 1.8,
  } as PIDConfig,

  RESPONSIVE: {
    kP: 4.0, // High proportional for fast response
    kI: 0.3,
    kD: 0.4, // Low derivative allows faster changes
    minOutput: 0.1,
    maxOutput: 2.2,
  } as PIDConfig,
}

/**
 * Factory function to create controller with preset
 */
export function createDeflationController(
  preset: keyof typeof PID_PRESETS = 'CONSERVATIVE'
): DeflationPIDController {
  return new DeflationPIDController(PID_PRESETS[preset])
}

/**
 * Enhanced deflation that combines PID with context awareness
 */
export function getEnhancedPIDDeflation(
  controller: DeflationPIDController,
  params: {
    currentRate: number
    targetRate: number
    rawScore: number
    gameProgress: number
    quotaBalance: number
  }
): { deflationFactor: number; debugInfo: any } {
  // Get base PID deflation
  const baseDef = controller.getDeflationFactor(
    params.currentRate,
    params.targetRate
  )

  // Context modifiers
  const scoreInflationMod = params.rawScore > 1.0 ? 0.9 : 1.0
  const gameProgressMod = params.gameProgress > 0.7 ? 1.1 : 1.0 // Less aggressive near end
  const balanceMod = params.quotaBalance < 0.7 ? 0.95 : 1.0 // Slightly more aggressive if unbalanced

  const contextDeflation =
    baseDef * scoreInflationMod * gameProgressMod * balanceMod

  const debugInfo = {
    ...controller.getDebugInfo(params.currentRate, params.targetRate),
    baseDef: Stats.round(baseDef),
    scoreInflationMod: Stats.round(scoreInflationMod),
    gameProgressMod: Stats.round(gameProgressMod),
    balanceMod: Stats.round(balanceMod),
    finalDef: Stats.round(contextDeflation),
  }

  return {
    deflationFactor: Stats.clamp(contextDeflation, 0.05, 2.5),
    debugInfo,
  }
}
