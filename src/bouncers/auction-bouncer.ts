import type { BerghainBouncer } from '../berghain'
import type { GameState, GameStatusRunning, ScenarioAttributes } from '../types'

type Guest<T extends ScenarioAttributes = ScenarioAttributes> = {
  [K in keyof T]: boolean
}

interface State {
  admittedCount: number
  attributeCounts: Record<keyof Guest, number>
  peopleProcessed: number
}

const Config = {
  TOTAL_PEOPLE: 10_000,
  MAX_REJECTIONS: 10_000,
  MAX_ADMISSIONS: 1_000,
  BASE_THRESHOLD: 0.51,
  SCALING: 1.0,
}

/* ==================  HELPERS  ================== */

function average(...values: number[][]): number {
  const flat = values.flat(1)
  const size = flat.length
  if (size === 0) return 0
  return flat.reduce((a, b) => a + b, 0) / size
}

function round(value: number): number {
  return Math.round(value * 10_000) / 10_000
}

function clamp(lower: number, value: number, upper: number): number {
  return Math.max(lower, Math.min(value, upper))
}

function match<T>(predicate: boolean, yes: T, no: T): T {
  return predicate ? yes : no
}

function isAboveZero(value: number): boolean {
  return value > 0
}

function isBelowZero(value: number): boolean {
  return value < 0
}

function divide(a: number, b: number): number {
  if (b === 0) return 0
  return a / b
}

function getGuestAttributes(guest: Record<string, boolean>): string[] {
  return Object.entries(guest)
    .filter((tuple) => tuple[1] === true)
    .map((tuple) => tuple[0])
}

/* ==================  HELPERS  ================== */

interface Quota {
  readonly attribute: string
  readonly minCount: number
  needed: number
  totalSeen: number
  frequency: number
  progress(): number
  isFinished(): boolean
  rate(): number
}

function createContext({ game, ...initialState }: GameState) {
  const frequencies = game.attributeStatistics.relativeFrequencies
  const correlations = game.attributeStatistics.correlations
  const quotas: Quota[] = game.constraints.map((constraint) => {
    return {
      attribute: constraint.attribute,
      minCount: constraint.minCount,
      needed: constraint.minCount,
      frequency: frequencies[constraint.attribute],
      totalSeen: 0,
      didAdmit() {
        this.needed--
      },
      isFinished() {
        return this.needed <= 0
      },
      progress() {
        return divide(this.needed, this.minCount)
      },
      rate() {
        const admitted = this.minCount - this.needed
        return divide(admitted, this.totalSeen)
      },
    }
  })

  return {
    ...initialState,
    game,
    quotas,
    totalPeopleInLine: Config.TOTAL_PEOPLE,
    totalAvailableSpots: Config.MAX_ADMISSIONS,
    get totalAdmitted(): number {
      return Config.MAX_ADMISSIONS - this.totalAvailableSpots
    },
    get totalRejected(): number {
      return Config.TOTAL_PEOPLE - this.totalAdmitted
    },
    get totalNeededMax(): number {
      return this.quotas.reduce((total, person) => total + person.minCount, 0)
    },
    get totalQuotasMet(): number {
      return this.quotas.reduce((total, person) => total + Number(person.minCount <= 0), 0)
    },
    info: {
      attributes: [] as string[],
      score: 0,
      threshold: 0,
      admit: false,
    },

    shouldAdmitPerson(guest: ScenarioAttributes): boolean {
      this.seen(guest) // update attributes

      const score = this.getGuestScore(guest)
      const threshold = this.getThreshold()
      const shouldAdmitGuest = score >= threshold

      this.info = {
        attributes: getGuestAttributes(guest),
        admit: shouldAdmitGuest,
        threshold: round(threshold),
        score: round(score),
      }

      if (shouldAdmitGuest) {
        this.didAdmit(guest)
      }

      return shouldAdmitGuest
    },
    seen(person: ScenarioAttributes) {
      this.totalPeopleInLine--
      this.quotas.forEach((quota) => {
        if (!person[quota.attribute]) return
        quota.totalSeen++
      })
    },
    didAdmit(person: ScenarioAttributes) {
      this.totalAvailableSpots--
      this.quotas.forEach((quota) => {
        if (!person[quota.attribute]) return
        quota.needed--
      })
    },
    getQuota(attribute: string) {
      const quota = this.quotas.find((quota) => quota.attribute === attribute)
      if (!quota) throw new Error(`Missing quota: ${attribute}`)
      return quota
    },
    getProgressRatio() {
      return this.totalAdmitted / Config.MAX_ADMISSIONS
    },
    getThreshold() {
      return Config.BASE_THRESHOLD + this.getProgressRatio() * Config.SCALING
    },
    getGuestScore(guest: ScenarioAttributes): number {
      const impactScores = this.getExpectedImpact(guest)

      const components = this.quotas.map((quota) => {
        return this.getUrgencyScore(quota) * impactScores[quota.attribute]
      })

      return average(components)
    },
    getUrgencyScore(quota: Quota) {
      const needed = Math.max(0, quota.minCount)
      const expecting = this.totalPeopleInLine * quota.frequency
      if (expecting <= 0) return 0
      return needed / expecting
    },
    getExpectedImpact(guest: ScenarioAttributes) {
      const impact = {} as Record<string, number>
      const guestAttrs = getGuestAttributes(guest)
      for (const attrA in impact) {
        impact[attrA] ??= 0
        for (const attrB of guestAttrs) {
          impact[attrA] += correlations[attrB][attrA] * this.totalAvailableSpots
        }
      }
      return impact
    },
    getScarcityOrAbundanceMultiplier(attribute: string) {
      const quota = this.getQuota(attribute)
      const ratio = divide(quota.progress(), this.getProgressRatio())
      const modifer = ratio > 1.0 ? ratio - 1.0 : 1.0 + ratio
      // should be above 1 if ahead
      // should be below 1 if behind
      return clamp(0.01, modifer, 1.99) || 1.0
    },
    getInfo() {
      return this.info
    },
  }
}

/**
 *  Exports an auction bouncer which handles scoring people via an auction,
 *  where guests bid to enter with their attributes.
 */
export function AuctionBouncer(initialState: GameState): BerghainBouncer {
  /**
   *  Create internal state which can be passed arround
   */
  const ctx = createContext(initialState)

  /**
   *  Return an object to be used as the bouncer.s
   */
  return {
    admit({ status, nextPerson }) {
      if (status !== 'running') return false
      const guest = nextPerson.attributes

      // update context with person
      ctx.seen(nextPerson.attributes)

      // calculate
      const score = ctx.getGuestScore(guest)
      const threshold = ctx.getThreshold()
      const isAdmitted = score >= threshold

      // set variables for debugging

      if (!isAdmitted) return false
      ctx.didAdmit(guest)
      return true
    },
    getProgress() {
      return {
        peopleInLine: ctx.totalPeopleInLine,
        spotsRemaining: ctx.totalAvailableSpots,
        admitted: ctx.totalAdmitted,
        rejected: ctx.totalRejected,
        progress: round(ctx.getProgressRatio()),
        ...ctx.getInfo(),
      }
    },
    getOutput() {
      return {
        ...initialState,
        ...this.getProgress(),
      }
    },
  }
}
