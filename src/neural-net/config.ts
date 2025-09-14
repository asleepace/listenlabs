/** @file config.ts */

/**
 *  Shared configuration for the neural-net bouncer.
 */
export namespace Conf {
  /**
   * The maximum number of rejections which can happen in a game.
   * @default 10_000
   */
  export const MAX_REJECTIONS = 10_000

  /**
   * The maximum number of admissions which can happen in a game.
   * @default 1_000
   */
  export const MAX_ADMISSIONS = 1_000

  /**
   * The target number of rejections we want to be at or under by the
   * time we admit our last person (lower is better / harder).
   * @default 5_000
   */
  export const TARGET_REJECTIONS = 5_000

  /**
   * The total number of features for our neural net.
   * @default 17
   */
  export const FEATURES = 17

  /**
   * Wiggle room for available spaces before we start running out of space.
   * @default 1
   */
  export const SAFETY_CUSHION = 1

  /**
   * Training specific configuration.
   */
  export const TRAINING = Object.freeze({
    LAMBDA_SHORTFALL: 50, // linear penalty per missing head
    QUAD_SHORTFALL: 0.05, // extra penalty for concentrated gaps
    BETA_SURPLUS: 1.0, // mild penalty per head above required
    LOSS_PENALTY: 100000, // flat penalty for losing (unmet or reject cap)

    /** Per-attribute weights to nudge the policy where we were systematically off. */
    SHORTFALL_WEIGHTS: {
      creative: 1.6, // push scarce 'creative'
      techno_lover: 1.2, // mild bump to avoid TL misses
      // others default to 1.0
    } as Record<string, number>,

    SURPLUS_WEIGHTS: {
      // creative: 1.6, // penalize 'creative' overshoot
      well_connected: 1.8,
      // others default to 1.0
    } as Record<string, number>,

    // --- helpers ---

    /**
     * Helper which will return the shortfall weight for a given attribute or use
     * the provided fallback if none exist.
     */
    getShortfallWeight(props: { attribute: string; default?: number }): number {
      if (props.attribute in this.SHORTFALL_WEIGHTS) {
        return this.SHORTFALL_WEIGHTS[props.attribute] ?? props.default
      } else {
        return props.default ?? 1.0
      }
    },

    /**
     * Helper which will return the shortfall weight for a given attribute or use
     * the provided fallback if none exist.
     */
    getSurplusWeight(props: { attribute: string; default?: number }): number {
      if (props.attribute in this.SURPLUS_WEIGHTS) {
        return this.SURPLUS_WEIGHTS[props.attribute] ?? props.default
      } else {
        return props.default ?? 1.0
      }
    },
  })
}
