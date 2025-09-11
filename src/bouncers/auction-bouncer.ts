import type { BerghainBouncer } from '../berghain'
import type { GameState, GameStatusRunning, ScenarioAttributes } from '../types'

type Guest<T extends ScenarioAttributes = ScenarioAttributes> = {
  [K in keyof T]: boolean
}

interface Constraint {
  attribute: keyof ScenarioAttributes
  minCount: number
}

interface State {
  admittedCount: number
  attributeCounts: Record<keyof Guest, number>
  peopleProcessed: number
}

interface Constraint {
  attribute: keyof Guest
  minCount: number
}

interface State {
  admittedCount: number
  attributeCounts: Record<keyof Guest, number>
  peopleProcessed: number
}

// 1. Constraint Urgency Score
function calculateUrgencyScore(
  constraint: Constraint,
  currentCount: number,
  peopleRemaining: number,
  frequency: number
): number {
  const requiredRemaining = Math.max(0, constraint.minCount - currentCount)
  const expectedFromRemaining = peopleRemaining * frequency

  return expectedFromRemaining > 0
    ? requiredRemaining / expectedFromRemaining
    : 0
}

// 2. Guest Value Scoring
function calculateGuestValue(
  guest: Guest,
  constraints: Constraint[],
  state: State,
  peopleRemaining: number,
  frequencies: Record<keyof Guest, number>
): number {
  let totalValue = 0

  for (const constraint of constraints) {
    const urgency = calculateUrgencyScore(
      constraint,
      state.attributeCounts[constraint.attribute],
      peopleRemaining,
      frequencies[constraint.attribute]
    )

    if (guest[constraint.attribute]) {
      totalValue += urgency
    }
  }

  return totalValue
}

// 3. Dynamic Admission Threshold
function calculateThreshold(
  admissionsUsed: number,
  maxAdmissions: number,
  baseThreshold: number = 0.5,
  scaling: number = 1.0
): number {
  const progressRatio = admissionsUsed / maxAdmissions
  return baseThreshold + progressRatio * scaling
}

// 4. Correlation-Adjusted Expected Impact
function calculateExpectedImpact(
  guest: Guest,
  correlations: Record<keyof Guest, Record<keyof Guest, number>>,
  remainingAdmissions: number
): Record<keyof Guest, number> {
  const impact: Record<keyof Guest, number> = {
    techno_lover: 0,
    well_connected: 0,
    creative: 0,
    berlin_local: 0,
  }

  const guestAttrs = Object.entries(guest)
    .filter(([_, hasAttr]) => hasAttr)
    .map(([attr, _]) => attr as keyof Guest)

  for (const targetAttr in impact) {
    const attr = targetAttr as keyof Guest
    for (const guestAttr of guestAttrs) {
      impact[attr] += correlations[guestAttr][attr] * remainingAdmissions
    }
  }

  return impact
}

function createContext(initialState: GameState) {
  return {
    ...initialState,

    update(status: GameStatusRunning<ScenarioAttributes>) {
      this.status = status
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
    admit(next) {
      ctx.update(next)
      return true
    },
    getProgress() {
      return {}
    },
    getOutput() {
      return {
        ...initialState,
        ...this.getProgress(),
      }
    },
  }
}
