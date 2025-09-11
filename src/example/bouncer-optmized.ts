import type { GameState, PersonAttributes, ScenarioAttributes } from '../types'
import type { BergainBouncer } from '../berghain'

interface ConstraintInfo {
  attribute: keyof ScenarioAttributes
  needed: number
  frequency: number
  difficulty: number
  priority: number
}

interface AdmissionScore {
  baseValue: number
  rarityMultiplier: number
  urgencyMultiplier: number
  correlationBonus: number
  finalScore: number
  reasoning: string[]
}

export class OptimizedBouncer implements BergainBouncer {
  private gameData: GameState['game']
  private constraints: Map<keyof ScenarioAttributes, ConstraintInfo>
  private counts: Map<keyof ScenarioAttributes, number>
  private correlations: Map<
    keyof ScenarioAttributes,
    Map<keyof ScenarioAttributes, number>
  >

  // Dynamic state
  private currentThreshold: number = 0.35
  private targetAdmissionRate: number = 0.2
  private debugInfo: any = {}

  // Error tracking for smoother threshold adjustment
  private rateErrorHistory: number[] = []

  constructor(gameState: GameState) {
    this.gameData = gameState.game
    this.constraints = new Map()
    this.counts = new Map()
    this.correlations = new Map()

    this.initializeConstraints()
    this.initializeCorrelations()
  }

  private initializeConstraints() {
    this.gameData.constraints.forEach((constraint) => {
      const frequency =
        this.gameData.attributeStatistics.relativeFrequencies[
          constraint.attribute
        ]!
      const difficulty = this.calculateDifficulty(
        constraint.minCount,
        frequency
      )

      this.constraints.set(constraint.attribute, {
        attribute: constraint.attribute,
        needed: constraint.minCount,
        frequency,
        difficulty,
        priority: difficulty,
      })

      this.counts.set(constraint.attribute, 0)
    })
  }

  private initializeCorrelations() {
    Object.entries(this.gameData.attributeStatistics.correlations).forEach(
      ([attr1, correlMap]) => {
        const corrMap = new Map<keyof ScenarioAttributes, number>()
        Object.entries(correlMap as any).forEach(([attr2, correlation]) => {
          if (attr1 !== attr2) {
            corrMap.set(
              attr2 as keyof ScenarioAttributes,
              correlation as number
            )
          }
        })
        this.correlations.set(attr1 as keyof ScenarioAttributes, corrMap)
      }
    )
  }

  private calculateDifficulty(quota: number, frequency: number): number {
    // Safer calculation with bounds
    const rarityFactor = Math.min(100, 1 / Math.max(frequency, 0.005)) // Cap at 200x
    const quotaPressure = quota / 1000
    return Math.min(1000, rarityFactor * quotaPressure * 10) // Cap total difficulty
  }

  private syncStateFromGameStatus(status: GameState['status']) {
    if (status.status !== 'running') return

    // Update counts based on current admitted count vs our tracked counts
    const totalAdmitted = status.admittedCount
    const ourTotal = Array.from(this.counts.values()).reduce(
      (sum, count) =>
        sum +
        Math.min(
          count,
          this.constraints.get(Array.from(this.constraints.keys())[0]!)
            ?.needed || 0
        ),
      0
    )

    // If there's a significant discrepancy, we're out of sync (shouldn't happen in practice)
    if (Math.abs(totalAdmitted - ourTotal) > 10) {
      console.warn(
        `Count mismatch detected: game=${totalAdmitted}, internal=${ourTotal}`
      )
    }
  }

  private updatePriorities(
    admittedCount: number,
    spotsLeft: number,
    totalProcessed: number
  ) {
    const peopleLeft = Math.max(0, 10000 - totalProcessed)

    this.constraints.forEach((constraint, attr) => {
      const current = this.counts.get(attr)!
      const needed = Math.max(0, constraint.needed - current)
      const progress = current / constraint.needed

      if (needed === 0) {
        this.constraints.set(attr, { ...constraint, priority: 0 })
        return
      }

      // Base priority from difficulty
      let priority = constraint.difficulty

      // Progress urgency - exponential penalty for being behind
      const expectedProgress = Math.min(admittedCount / 1000, 1.0)
      if (progress < expectedProgress * 0.6) {
        priority *= 4.0 // Severely behind
      } else if (progress < expectedProgress * 0.8) {
        priority *= 2.0 // Behind
      }

      // Scarcity pressure - mathematical feasibility check
      const expectedWithAttribute = peopleLeft * constraint.frequency
      const scarcityRatio = needed / Math.max(expectedWithAttribute, 1)

      if (scarcityRatio > 2.0) {
        priority *= 8.0 // Mathematically very tight
      } else if (scarcityRatio > 1.0) {
        priority *= 3.0 // Tight
      } else if (scarcityRatio > 0.7) {
        priority *= 1.5 // Getting tight
      }

      // Endgame pressure
      if (spotsLeft <= 200 && needed > 0) {
        const endgamePressure = Math.min(5.0, needed / spotsLeft)
        priority *= 1.0 + endgamePressure
      }

      // Cap priority to prevent explosion
      priority = Math.min(priority, 10000)

      this.constraints.set(attr, { ...constraint, priority })
    })
  }

  private calculateAdmissionScore(
    attributes: Partial<ScenarioAttributes>,
    admittedCount: number,
    spotsLeft: number
  ): AdmissionScore {
    const reasoning: string[] = []

    // Get useful attributes
    const usefulAttrs = Object.entries(attributes)
      .filter(([attr, hasAttr]) => {
        if (!hasAttr) return false
        const constraint = this.constraints.get(
          attr as keyof ScenarioAttributes
        )
        if (!constraint) return false
        const current = this.counts.get(attr as keyof ScenarioAttributes)!
        return current < constraint.needed
      })
      .map(([attr]) => attr as keyof ScenarioAttributes)

    if (usefulAttrs.length === 0) {
      return {
        baseValue: 0,
        rarityMultiplier: 0,
        urgencyMultiplier: 0,
        correlationBonus: 0,
        finalScore: 0,
        reasoning: ['No useful attributes'],
      }
    }

    // Calculate base value - sum of normalized priorities
    let baseValue = 0
    let maxPriority = 1

    usefulAttrs.forEach((attr) => {
      const constraint = this.constraints.get(attr)!
      const current = this.counts.get(attr)!
      const needed = constraint.needed - current

      // Normalized contribution based on priority and need percentage
      const contribution = constraint.priority * (needed / constraint.needed)
      baseValue += contribution
      maxPriority = Math.max(maxPriority, constraint.priority)

      reasoning.push(
        `${attr}: priority=${Math.round(constraint.priority)}, needed=${needed}`
      )
    })

    // Normalize base value
    baseValue = baseValue / maxPriority

    // Rarity multiplier
    let rarityMultiplier = 1.0
    const hasUltraRare = usefulAttrs.some(
      (attr) => this.constraints.get(attr)!.frequency < 0.08
    )
    const hasRare = usefulAttrs.some(
      (attr) => this.constraints.get(attr)!.frequency < 0.15
    )

    if (hasUltraRare) {
      rarityMultiplier = 3.0
      reasoning.push('Ultra-rare attribute (3x)')
    } else if (hasRare) {
      rarityMultiplier = 2.0
      reasoning.push('Rare attribute (2x)')
    }

    // Multi-attribute multiplier
    let urgencyMultiplier = 1.0
    if (usefulAttrs.length >= 3) {
      urgencyMultiplier = 2.5
      reasoning.push('3+ attributes (2.5x)')
    } else if (usefulAttrs.length === 2) {
      urgencyMultiplier = 1.6
      reasoning.push('2 attributes (1.6x)')
    }

    // Correlation bonus for negative correlations
    let correlationBonus = 0
    for (let i = 0; i < usefulAttrs.length; i++) {
      for (let j = i + 1; j < usefulAttrs.length; j++) {
        const attr1 = usefulAttrs[i]!
        const attr2 = usefulAttrs[j]!
        const correlation = this.correlations.get(attr1)?.get(attr2) || 0

        if (correlation < -0.4) {
          correlationBonus += Math.abs(correlation) * 1.5
          reasoning.push(
            `Negative correlation bonus: ${attr1}-${attr2} (${correlation.toFixed(
              2
            )})`
          )
        }
      }
    }

    const finalScore =
      baseValue * rarityMultiplier * urgencyMultiplier + correlationBonus

    return {
      baseValue: Math.round(baseValue * 100) / 100,
      rarityMultiplier,
      urgencyMultiplier,
      correlationBonus: Math.round(correlationBonus * 100) / 100,
      finalScore: Math.round(finalScore * 100) / 100,
      reasoning,
    }
  }

  private updateThreshold(currentRate: number, totalProcessed: number) {
    const rateError = currentRate - this.targetAdmissionRate
    this.rateErrorHistory.push(rateError)

    // Keep only recent history
    if (this.rateErrorHistory.length > 50) {
      this.rateErrorHistory.shift()
    }

    // Smoother PID-like adjustment
    const proportional = rateError * 0.3
    const integral =
      this.rateErrorHistory.length > 10
        ? (this.rateErrorHistory.reduce((sum, err) => sum + err, 0) /
            this.rateErrorHistory.length) *
          0.1
        : 0

    const adjustment = proportional + integral
    const maxAdjustment = 0.05 // Limit swing

    this.currentThreshold += Math.max(
      -maxAdjustment,
      Math.min(maxAdjustment, adjustment)
    )
    this.currentThreshold = Math.max(
      0.05,
      Math.min(0.85, this.currentThreshold)
    )

    // Endgame relaxation
    if (totalProcessed > 1500) {
      const endgameRelaxation = Math.min(0.2, (totalProcessed - 1500) / 5000)
      this.currentThreshold = Math.max(
        0.05,
        this.currentThreshold - endgameRelaxation
      )
    }
  }

  admit(status: GameState['status']): boolean {
    if (status.status !== 'running' || !status.nextPerson) return false

    this.syncStateFromGameStatus(status)

    const admittedCount = status.admittedCount
    const rejectedCount = status.rejectedCount
    const totalProcessed = admittedCount + rejectedCount
    const spotsLeft = 1000 - admittedCount
    const currentRate = totalProcessed > 0 ? admittedCount / totalProcessed : 0

    // Update dynamic priorities
    this.updatePriorities(admittedCount, spotsLeft, totalProcessed)

    // Calculate score
    const scoreResult = this.calculateAdmissionScore(
      status.nextPerson.attributes,
      admittedCount,
      spotsLeft
    )

    // Update threshold
    this.updateThreshold(currentRate, totalProcessed)

    // Decision logic
    let shouldAdmit = scoreResult.finalScore > this.currentThreshold

    // Emergency admission for critical endgame
    if (!shouldAdmit && spotsLeft <= 50) {
      const usefulAttrs = Object.entries(status.nextPerson.attributes).filter(
        ([attr, hasAttr]) => {
          if (!hasAttr) return false
          const constraint = this.constraints.get(
            attr as keyof ScenarioAttributes
          )
          if (!constraint) return false
          const current = this.counts.get(attr as keyof ScenarioAttributes)!
          return current < constraint.needed
        }
      )

      const hasUltraRare = usefulAttrs.some(
        ([attr]) =>
          this.constraints.get(attr as keyof ScenarioAttributes)!.frequency <
          0.08
      )

      if (hasUltraRare) {
        shouldAdmit = true
        scoreResult.reasoning.push('EMERGENCY: Ultra-rare in final 50 spots')
      }
    }

    // Update internal counts if admitting
    if (shouldAdmit) {
      Object.entries(status.nextPerson.attributes).forEach(
        ([attr, hasAttr]) => {
          if (hasAttr) {
            const current =
              this.counts.get(attr as keyof ScenarioAttributes) || 0
            this.counts.set(attr as keyof ScenarioAttributes, current + 1)
          }
        }
      )
    }

    // Store debug info
    this.debugInfo = {
      score: scoreResult,
      threshold: Math.round(this.currentThreshold * 1000) / 1000,
      currentRate: Math.round(currentRate * 1000) / 1000,
      rateError:
        Math.round((currentRate - this.targetAdmissionRate) * 1000) / 1000,
      spotsLeft,
      decision: shouldAdmit ? 'ADMIT' : 'REJECT',
    }

    return shouldAdmit
  }

  getProgress() {
    const quotas = Array.from(this.constraints.entries())
      .map(([attr, constraint]) => {
        const current = this.counts.get(attr) || 0
        return {
          attribute: attr,
          needed: Math.max(0, constraint.needed - current),
          current,
          progress: Math.round((current / constraint.needed) * 100) / 100,
          priority: Math.round(constraint.priority),
        }
      })
      .filter((q) => q.needed > 0)

    return {
      quotas: quotas.sort((a, b) => b.priority - a.priority),
      debug: this.debugInfo,
      currentThreshold: this.currentThreshold,
      targetRate: this.targetAdmissionRate,
    }
  }

  getOutput() {
    return this.getProgress()
  }
}
