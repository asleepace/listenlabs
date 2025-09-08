/**
 * Percentage value as a decimal (e.g. 0.5 = 50%)
 */
export type Percentage = number

/**
 * Whole number usually in the thousands.
 */
export type NumberOfPeople = number

/**
 * Base game configuration settings for the nightclub bouncer algorithm.
 */
export interface GameConfig {
  // Admission threshold settings

  /**
   * Base admission score threshold - neutral point for decisions.
   * Lower = more lenient admission, Higher = more selective
   * @range 0.3 - 0.7
   * @default 0.55
   */
  BASE_THRESHOLD: Percentage

  /**
   * Minimum threshold floor - prevents algorithm from becoming too lenient.
   * @range 0.2 - 0.5
   * @default 0.35
   */
  MIN_THRESHOLD: Percentage

  /**
   * Maximum threshold ceiling - prevents algorithm from becoming too restrictive.
   * @range 0.7 - 0.95
   * @default 0.80
   */
  MAX_THRESHOLD: Percentage

  /**
   * How aggressively threshold changes as venue fills up.
   * Lower = consistent throughout, Higher = gets stricter as venue fills
   * @range 0.2 - 0.8
   * @default 0.3
   */
  THRESHOLD_RAMP: Percentage

  // Game timing and targets

  /**
   * Target number of people processed when aiming to complete quotas.
   * Lower = rush early (risk running out of spots), Higher = spread out (risk missing rare attributes)
   * @range 2000 - 6000
   * @default 3500
   */
  TARGET_RANGE: NumberOfPeople

  /**
   * Multiplier for how much being behind schedule affects scoring.
   * Lower = relaxed timing, Higher = panic when behind
   * @range 1.0 - 6.0
   * @default 2.5
   */
  URGENCY_MODIFIER: number

  // Scoring bonuses

  /**
   * Bonus multiplier for people with positively correlated attributes.
   * @range 0.1 - 0.5
   * @default 0.3
   */
  CORRELATION_BONUS: Percentage

  /**
   * Bonus for rare combinations (negatively correlated but both needed).
   * @range 0.3 - 1.0
   * @default 0.7
   */
  NEGATIVE_CORRELATION_BONUS: Percentage

  /**
   * Correlation threshold below which attributes are considered negatively correlated.
   * @range -0.7 - -0.3
   * @default -0.5
   */
  NEGATIVE_CORRELATION_THRESHOLD: number

  /**
   * Bonus multiplier for people with multiple needed attributes.
   * Too high = over-value generalists, Too low = miss efficient multi-quota fills
   * @range 0.3 - 1.5
   * @default 0.7
   */
  MULTI_ATTRIBUTE_BONUS: Percentage

  /**
   * Extra bonus for rare attribute combinations.
   * @range 0.3 - 1.0
   * @default 0.6
   */
  RARE_PERSON_BONUS: Percentage

  // Game constants

  /**
   * Total maximum people that can be admitted to the venue.
   * @constant 1000
   */
  MAX_CAPACITY: NumberOfPeople

  /**
   * Total number of people in line to process.
   * @constant 10000
   */
  TOTAL_PEOPLE: NumberOfPeople

  // Critical attribute thresholds

  /**
   * Number of spots remaining when an attribute becomes strictly required.
   * Higher = more conservative, Lower = more aggressive
   * @range 3 - 15
   * @default 8
   */
  CRITICAL_REQUIRED_THRESHOLD: NumberOfPeople

  /**
   * Percentage of remaining people needed to trigger critical status.
   * @range 0.5 - 0.9
   * @default 0.75
   */
  CRITICAL_IN_LINE_RATIO: Percentage

  /**
   * Percentage of venue capacity when attributes become critical.
   * @range 0.6 - 0.9
   * @default 0.75
   */
  CRITICAL_CAPACITY_RATIO: Percentage

  // Misc settings

  /**
   * Minimum number of scores needed for statistical calculations.
   * @default 5
   */
  MIN_RAW_SCORES: number

  /**
   * Configuration identifier or description.
   * @default "Base Configuration"
   */
  MESSAGE: string

  /**
   * The specific game scenario to run.
   * @default "3"
   */
  SCENARIO: '1' | '2' | '3'
}

/**
 * Default base configuration optimized for balanced performance.
 * This configuration aims to complete quotas efficiently while avoiding
 * both early overfill and late-game gridlock scenarios.
 */
export const BASE_CONFIG: GameConfig = {
  // Admission thresholds - balanced approach
  BASE_THRESHOLD: 0.55,
  MIN_THRESHOLD: 0.35,
  MAX_THRESHOLD: 0.8,
  THRESHOLD_RAMP: 0.3,

  // Game timing - front-load quota completion
  TARGET_RANGE: 3500,
  URGENCY_MODIFIER: 2.5,

  // Scoring bonuses - reward efficiency
  CORRELATION_BONUS: 0.3,
  NEGATIVE_CORRELATION_BONUS: 0.7,
  NEGATIVE_CORRELATION_THRESHOLD: -0.5,
  MULTI_ATTRIBUTE_BONUS: 0.7,
  RARE_PERSON_BONUS: 0.6,

  // Game constants
  MAX_CAPACITY: 1000,
  TOTAL_PEOPLE: 10000,

  // Critical thresholds - proactive detection
  CRITICAL_REQUIRED_THRESHOLD: 8,
  CRITICAL_IN_LINE_RATIO: 0.75,
  CRITICAL_CAPACITY_RATIO: 0.75,

  // Misc
  MIN_RAW_SCORES: 5,
  MESSAGE: 'Base Configuration - Balanced Strategy',
  SCENARIO: '3',
}

/**
 * Conservative configuration for safer, more predictable runs.
 * Higher thresholds, lower bonuses, more cautious critical detection.
 */
export const CONSERVATIVE_CONFIG: GameConfig = {
  ...BASE_CONFIG,
  BASE_THRESHOLD: 0.65,
  MIN_THRESHOLD: 0.45,
  MAX_THRESHOLD: 0.85,
  URGENCY_MODIFIER: 2.0,
  MULTI_ATTRIBUTE_BONUS: 0.5,
  CRITICAL_REQUIRED_THRESHOLD: 12,
  MESSAGE: 'Conservative Configuration - Safe Strategy',
}

/**
 * Aggressive configuration for high-risk, high-reward runs.
 * Lower thresholds, higher bonuses, earlier critical detection.
 */
export const AGGRESSIVE_CONFIG: GameConfig = {
  ...BASE_CONFIG,
  BASE_THRESHOLD: 0.45,
  MIN_THRESHOLD: 0.25,
  MAX_THRESHOLD: 0.75,
  TARGET_RANGE: 3000,
  URGENCY_MODIFIER: 3.5,
  MULTI_ATTRIBUTE_BONUS: 0.9,
  CRITICAL_REQUIRED_THRESHOLD: 5,
  CRITICAL_IN_LINE_RATIO: 0.65,
  SCENARIO: '3',
  MESSAGE: 'Aggressive Configuration - High Risk Strategy',
}
