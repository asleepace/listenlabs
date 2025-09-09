import type { ScenarioAttributes } from '../types'
import type { Metrics } from './metrics'
import { Stats } from './statistics'

export interface ScoreCalculationParams {
  attributes: Partial<ScenarioAttributes>
  metrics: Metrics
  criticalAttributes: CriticalAttributes
  allQuotasMet: boolean
  admittedCount: number
}

export interface CriticalAttribute {
  needed: number
  required: boolean
  modifier: number
}

export type CriticalAttributes = Partial<
  Record<keyof ScenarioAttributes, CriticalAttribute>
>

export interface ScoreConfig {
  urgencyDivisor: number
  maxUrgencyScore: number
  maxCriticalMultiplier: number
  rarityThresholds: {
    high: number // frequency < high = high rarity bonus
    medium: number // frequency < medium = medium rarity bonus
  }
  rarityBonuses: {
    high: number
    medium: number
    normal: number
  }
  progressThresholds: {
    low: number // progress < low = high urgency
    medium: number // progress < medium = medium urgency
  }
  progressBonuses: {
    low: number
    medium: number
    normal: number
  }
  criticalMultiplierCap: number
  multiAttributeBonus: number
  normalizationBase: number
  maxScore: number
  isEndGame: boolean
}

const DEFAULT_SCORE_CONFIG: ScoreConfig = {
  urgencyDivisor: 100,
  maxUrgencyScore: 1.5,
  maxCriticalMultiplier: 10.0,
  rarityThresholds: {
    high: 0.1,
    medium: 0.4,
  },
  rarityBonuses: {
    high: 2.2,
    medium: 1.5,
    normal: 1.0,
  },
  progressThresholds: {
    low: 0.2,
    medium: 0.5,
  },
  progressBonuses: {
    low: 2.2,
    medium: 1.6,
    normal: 1.0,
  },
  criticalMultiplierCap: 4.0,
  multiAttributeBonus: 0.3,
  normalizationBase: 30,
  maxScore: 2.0,
  isEndGame: false,
}

export type ScorePresets = typeof SCORE_PRESETS

/**
 * Preset configurations for different game phases or strategies
 */
export const SCORE_PRESETS = {
  OPTIMIZED: {
    urgencyDivisor: 75, // Lower from 100
    maxUrgencyScore: 2.2, // Higher from 1.5
    maxCriticalMultiplier: 12.0, // Higher from 10.0

    rarityBonuses: {
      high: 2.8, // Higher from 2.2
      medium: 1.8, // Higher from 1.5
      normal: 1.0,
    },

    progressBonuses: {
      low: 2.8, // Higher from 2.2
      medium: 1.8, // Higher from 1.6
      normal: 1.0,
    },

    criticalMultiplierCap: 5.0, // Higher from 4.0
    multiAttributeBonus: 0.4, // Higher from 0.3
    normalizationBase: 25, // Lower from 30
    maxScore: 2.5, // Higher from 2.0
  } as ScoreConfig,
  CONSERVATIVE: {
    urgencyDivisor: 150,
    maxUrgencyScore: 1.2,
    rarityBonuses: { high: 1.8, medium: 1.3, normal: 1.0 },
    progressBonuses: { low: 1.8, medium: 1.4, normal: 1.0 },
    criticalMultiplierCap: 3.0,
    multiAttributeBonus: 0.2,
    normalizationBase: 40,
    maxScore: 1.5,
  } as Partial<ScoreConfig>,

  AGGRESSIVE: {
    urgencyDivisor: 80,
    maxUrgencyScore: 2.0,
    rarityBonuses: { high: 2.5, medium: 1.8, normal: 1.0 },
    progressBonuses: { low: 2.5, medium: 1.8, normal: 1.0 },
    criticalMultiplierCap: 5.0,
    multiAttributeBonus: 0.4,
    normalizationBase: 25,
    maxScore: 2.5,
  } as Partial<ScoreConfig>,

  BALANCED: DEFAULT_SCORE_CONFIG,

  ENDGAME_FOCUSED: {
    urgencyDivisor: 60,
    maxUrgencyScore: 2.0,
    maxCriticalMultiplier: 15.0,
    criticalMultiplierCap: 6.0,
    rarityBonuses: { high: 3.0, medium: 2.0, normal: 1.0 },
    progressBonuses: { low: 3.0, medium: 2.0, normal: 1.0 },
    multiAttributeBonus: 0.5,
    normalizationBase: 20,
    maxScore: 3.0,
  } as Partial<ScoreConfig>,
}

/**
 * Calculate admission score for a person with given attributes
 */
export function calculateAdmissionScore(
  params: ScoreCalculationParams,
  config: Partial<ScoreConfig> = {}
): number {
  const cfg = { ...DEFAULT_SCORE_CONFIG, ...config }

  if (params.allQuotasMet) return 1.0

  const isEndGame = Boolean(config.isEndGame)
  const usefulAttributes: (keyof ScenarioAttributes)[] =
    params.metrics.getUsefulAttributes(params.attributes, isEndGame)
  if (usefulAttributes.length === 0) return 0.0

  let totalScore = 0
  let hasCriticalAttribute = false
  let maxCriticalMultiplier = 1.0

  usefulAttributes.forEach((attr) => {
    const needed = params.metrics.getNeeded(attr)
    const progress = params.metrics.getProgress(attr)

    // Base urgency score
    const urgencyScore = Math.min(
      needed / cfg.urgencyDivisor,
      cfg.maxUrgencyScore
    )

    // Critical modifier
    const criticalInfo = params.criticalAttributes[attr]
    let criticalMultiplier = 1.0

    if (criticalInfo) {
      criticalMultiplier = Math.min(
        criticalInfo.modifier,
        cfg.maxCriticalMultiplier
      )
      maxCriticalMultiplier = Math.max(
        maxCriticalMultiplier,
        criticalMultiplier
      )
      hasCriticalAttribute = true
    }

    // Check for negative correlation conflicts
    const negativelyCorrelated = params.metrics.getNegativelyCorrelated(
      attr,
      -0.5
    )

    const hasConflictingAttribute = negativelyCorrelated.some(
      (conflictAttr) => {
        // Check if person has the conflicting attribute
        const hasConflictingAttr = params.attributes[conflictAttr]

        // Check if conflicting attribute is ahead in progress
        const conflictProgress = params.metrics.getProgress(conflictAttr)
        const currentProgress = params.metrics.getProgress(attr)

        return hasConflictingAttr && conflictProgress > currentProgress
      }
    )

    // Rarity bonus
    const frequency = params.metrics.frequencies[attr]!
    const rarityBonus = getRarityBonus(frequency, cfg)

    // Add velocity-based urgency
    const expectedProgress = (params.admittedCount || 0) / 1000
    const actualProgress = progress
    // const velocity = actualProgress / Math.max(expectedProgress, 0.01)
    const velocity =
      expectedProgress > 0.05 ? actualProgress / expectedProgress : 1.0
    const velocityBonus = velocity < 0.8 ? 2.0 : velocity < 0.9 ? 1.5 : 1.0

    // Modify correlation logic for better conflict handling
    let correlationBonus = 1.0
    if (negativelyCorrelated.length > 0) {
      const hasAnyConflicting = negativelyCorrelated.some(
        (conflictAttr) => params.attributes[conflictAttr]
      )
      if (hasAnyConflicting && progress < 0.9) {
        correlationBonus = 1.8 // Boost for anyone breaking correlation patterns
      }
    }

    // Progress urgency
    const progressUrgency = getProgressUrgency(progress, frequency, cfg)
    const attributeScore =
      urgencyScore *
      rarityBonus *
      progressUrgency *
      correlationBonus *
      velocityBonus

    totalScore += attributeScore
  })

  // Apply critical multiplier once to total
  if (hasCriticalAttribute) {
    totalScore *= Math.min(maxCriticalMultiplier, cfg.criticalMultiplierCap)
  }

  // Multi-attribute bonus
  if (usefulAttributes.length > 1) {
    totalScore *= 1 + (usefulAttributes.length - 1) * cfg.multiAttributeBonus
  }

  // Normalization
  const normalizedScore =
    Math.log(totalScore + 1) / Math.log(cfg.normalizationBase)
  return Math.min(normalizedScore, cfg.maxScore)
}

/**
 * Calculate rarity bonus based on attribute frequency
 */
function getRarityBonus(frequency: number, config: ScoreConfig): number {
  if (frequency < config.rarityThresholds.high) {
    return config.rarityBonuses.high
  } else if (frequency < config.rarityThresholds.medium) {
    return config.rarityBonuses.medium
  } else {
    return config.rarityBonuses.normal
  }
}

function getProgressUrgency(
  progress: number,
  frequency: number,
  config: ScoreConfig
): number {
  const baseUrgency =
    progress < config.progressThresholds.low
      ? config.progressBonuses.low
      : progress < config.progressThresholds.medium
      ? config.progressBonuses.medium
      : config.progressBonuses.normal

  // Boost common attributes that are behind where they should be
  if (frequency > 0.4 && progress < 0.6) {
    return baseUrgency * 1.5
  }

  return baseUrgency
}

/**
 * Calculate endgame score for urgent quota filling
 */
export function calculateEndgameScore(
  params: ScoreCalculationParams & {
    totalSpotsLeft: number
    isEndgame: boolean
  },
  config: { maxEndgameScore?: number } = {}
): number {
  if (!params.isEndgame) return 0

  const maxScore = config.maxEndgameScore || 3.0
  const usefulAttributes = params.metrics.getUsefulAttributes(
    params.attributes,
    true
  )
  const incompleteQuotas = params.metrics.getIncompleteConstraints()

  let endgameScore = 0

  usefulAttributes.forEach((attr) => {
    const quota = incompleteQuotas.find((q) => q.attribute === attr)
    if (quota) {
      // Score based on how desperately we need this attribute
      const urgency = Math.min(
        quota.needed / Math.max(params.totalSpotsLeft, 1),
        5.0
      )
      const frequency = params.metrics.frequencies[attr]!
      const scarcity = 1 / Math.max(frequency, 0.01)

      endgameScore += urgency * scarcity
    }
  })

  return Math.min(endgameScore, maxScore)
}

/**
 * Get score analysis for debugging
 */
export function getScoreAnalysis(
  params: ScoreCalculationParams,
  config: Partial<ScoreConfig & { isEndGame: boolean }> = {}
): {
  totalScore: number
  breakdown: {
    attribute: keyof ScenarioAttributes
    urgencyScore: number
    rarityBonus: number
    progressUrgency: number
    criticalMultiplier: number
    attributeScore: number
  }[]
  modifiers: {
    hasCriticalAttribute: boolean
    criticalMultiplierApplied: number
    multiAttributeBonus: number
    usefulAttributeCount: number
  }
  normalizedScore: number
} {
  const cfg = { ...DEFAULT_SCORE_CONFIG, ...config }
  const usefulAttributes = params.metrics.getUsefulAttributes(
    params.attributes,
    Boolean(config.isEndGame)
  )

  const breakdown = usefulAttributes.map((attr) => {
    const needed = params.metrics.getNeeded(attr)
    const progress = params.metrics.getProgress(attr)
    const frequency = params.metrics.frequencies[attr]!

    const urgencyScore = Math.min(
      needed / cfg.urgencyDivisor,
      cfg.maxUrgencyScore
    )
    const rarityBonus = getRarityBonus(frequency, cfg)
    const progressUrgency = getProgressUrgency(progress, frequency, cfg)

    const criticalInfo = params.criticalAttributes[attr]
    const criticalMultiplier = criticalInfo
      ? Math.min(criticalInfo.modifier, cfg.maxCriticalMultiplier)
      : 1.0

    const attributeScore = urgencyScore * rarityBonus * progressUrgency

    return {
      attribute: attr,
      urgencyScore: Stats.round(urgencyScore),
      rarityBonus: Stats.round(rarityBonus),
      progressUrgency: Stats.round(progressUrgency),
      criticalMultiplier: Stats.round(criticalMultiplier),
      attributeScore: Stats.round(attributeScore),
    }
  })

  const totalAttributeScore = breakdown.reduce(
    (sum, item) => sum + item.attributeScore,
    0
  )
  const hasCriticalAttribute = breakdown.some(
    (item) => item.criticalMultiplier > 1
  )
  const maxCriticalMultiplier = Math.max(
    ...breakdown.map((item) => item.criticalMultiplier)
  )
  const criticalMultiplierApplied = hasCriticalAttribute
    ? Math.min(maxCriticalMultiplier, cfg.criticalMultiplierCap)
    : 1.0

  const multiAttributeBonusMultiplier = Math.pow(2, usefulAttributes.length - 1) // 2x, 4x, 8x for 2, 3, 4 attributes

  const totalScore =
    totalAttributeScore *
    criticalMultiplierApplied *
    multiAttributeBonusMultiplier
  const normalizedScore = Math.min(
    Math.log(totalScore + 1) / Math.log(cfg.normalizationBase),
    cfg.maxScore
  )

  return {
    totalScore: Stats.round(totalScore),
    breakdown,
    modifiers: {
      hasCriticalAttribute,
      criticalMultiplierApplied: Stats.round(criticalMultiplierApplied),
      multiAttributeBonus: Stats.round(multiAttributeBonusMultiplier),
      usefulAttributeCount: usefulAttributes.length,
    },
    normalizedScore: Stats.round(normalizedScore),
  }
}
