import { BASE_CONFIG, type GameConfig } from './game-config'

export const WAVE_CONFIG: GameConfig = {
  ...BASE_CONFIG,
  BASE_THRESHOLD: 0.55, // Higher base for more selectivity
  MIN_THRESHOLD: 0.35, // Lower floor for aggressive recovery
  MAX_THRESHOLD: 0.8, // Lower ceiling to prevent over-restriction

  TARGET_RANGE: 3500, // Earlier completion target
  URGENCY_MODIFIER: 2.5, // Reduced from 3.0 - let threshold do the work

  MULTI_ATTRIBUTE_BONUS: 0.7, // Increased - reward efficiency
  RARE_PERSON_BONUS: 0.6, // Slight increase for rare combos

  // Keep existing values for these:
  THRESHOLD_RAMP: 0.3,
  CORRELATION_BONUS: 0.3,
  NEGATIVE_CORRELATION_BONUS: 0.7,
  NEGATIVE_CORRELATION_THRESHOLD: -0.5,

  // Critical thresholds - slightly more aggressive
  CRITICAL_IN_LINE_RATIO: 0.75, // Reduced from 0.8
  CRITICAL_CAPACITY_RATIO: 0.75, // Reduced from 0.8
  CRITICAL_REQUIRED_THRESHOLD: 8, // Increased from 5
}
