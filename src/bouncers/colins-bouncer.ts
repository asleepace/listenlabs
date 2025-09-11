import { Berghain, type BergainBouncer } from '../berghain'
import type {
  GameConstraints,
  GameState,
  GameStatusRunning,
  ScenarioAttributes,
} from '../types'

const config = {
  totalPeople: 10_000,
  targetRange: 3_000,
  maxRejections: 4_000,
  maxAdmissions: 1_000,
  get ratio() {
    return this.targetRange / this.totalPeople
  },
}

// const game = {
//   constraints: [
//     {
//       attribute: 'techno_lover',
//       minCount: 650,
//     },
//     {
//       attribute: 'well_connected',
//       minCount: 450,
//     },
//     {
//       attribute: 'creative',
//       minCount: 300,
//     },
//     {
//       attribute: 'berlin_local',
//       minCount: 750,
//     },
//   ] as const,
//   attributeStatistics: {
//     relativeFrequencies: {
//       techno_lover: 0.6265000000000001,
//       well_connected: 0.4700000000000001,
//       creative: 0.06227,
//       berlin_local: 0.398,
//     },
//     correlations: {
//       techno_lover: {
//         techno_lover: 1,
//         well_connected: -0.4696169332674324,
//         creative: 0.09463317039891586,
//         berlin_local: -0.6549403815606182,
//       },
//       well_connected: {
//         techno_lover: -0.4696169332674324,
//         well_connected: 1,
//         creative: 0.14197259140471485,
//         berlin_local: 0.5724067808436452,
//       },
//       creative: {
//         techno_lover: 0.09463317039891586,
//         well_connected: 0.14197259140471485,
//         creative: 1,
//         berlin_local: 0.14446459505650772,
//       },
//       berlin_local: {
//         techno_lover: -0.6549403815606182,
//         well_connected: 0.5724067808436452,
//         creative: 0.14446459505650772,
//         berlin_local: 1,
//       },
//     },
//   },
// } as const

console.clear()

function round(x: number): number {
  return Math.round(x * 10_000) / 10_000
}

function getPeopleFromConstraints({ game }: GameState) {
  const frequencies = game.attributeStatistics.relativeFrequencies
  const correlations = game.attributeStatistics.correlations
  const keys = game.constraints.map((constraint) => constraint.attribute)

  // create empty combos
  const combos = keys.reduce(
    (obj, key) => ({ ...obj, [key]: 0 }),
    {} as Record<string, number>
  )

  // initialize people from constraints
  const people = game.constraints.map(({ attribute, minCount }) => {
    const frequency = frequencies[attribute]
    let totalInLine = Math.round(config.totalPeople * frequency * config.ratio)
    const comboCopy = { ...combos }
    delete comboCopy[attribute]
    return {
      totalInLine,
      attribute,
      frequency,
      get modifer() {
        const realFrequency = this.admitted - this.totalSeen
        return 1 - (realFrequency - this.frequency) / 2.0
      },
      minCount,
      upperBound: totalInLine,
      lowerBound: totalInLine,
      combos: comboCopy,
      admitted: 0,
      totalSeen: 0,

      getOutput() {
        return {
          [this.attribute]: {
            admitted: this.admitted,
            rejected: this.totalSeen - this.admitted,
            needed: this.minCount - this.admitted,
            modifier: this.modifer,
          },
        }
      },

      forCombos<T>(callback: (key: string, value: number) => T): T[] {
        return Object.entries(this.combos).map(([key, value]) => {
          return callback(key, value)
        })
      },
      isComplete() {
        return this.admitted >= this.minCount
      },
      totalNeeded() {
        if (this.isComplete()) return 0
        return this.minCount - this.admitted
      },
      getScore(person: ScenarioAttributes): number {
        if (this.isComplete()) return 0
        if (!person[this.attribute]) return 0

        const hasAttribute = person[this.attribute] && !this.isComplete()

        // incrament the seen variable if person has attribute
        if (hasAttribute) this.totalSeen++

        const totalNeeded = this.totalNeeded()

        // map over combos and calculate score, +1 for combos and +2 for rare
        // combos with more people
        const comboScore = this.forCombos((key, value): number => {
          if (!person[key]) return 1.0

          if (value < 0) return 1.5
          if (value > 0) return 0.5
          return 1.0
        })

        return comboScore.reduce((a, b) => a + b, 0) * this.modifer
      },
      didAdmit(person: ScenarioAttributes) {
        if (person[this.attribute]) this.admitted++
        this.forCombos((key, value) => {
          if (value > 0) {
            this.combos[key]--
          } else {
            this.combos[key]++
          }
          // remove when done
          if (this.combos[key] === 0) {
            delete this.combos[key]
          }
        })
      },
    }
  })

  // iterate over items and calculate lower bounds
  people.forEach((personA, i) => {
    // get relationship of constraint to other constraints
    const relationshipsA = correlations[personA.attribute]

    // calculate lower bound by finding relationships
    people.forEach((personB, j) => {
      if (i === j) return

      // NOTE: double check that the relationship is % of other people with attribute
      // or if that's the % of people all together?
      const relationshipsB = correlations[personB.attribute]

      // such as techno_lovers are 9% of the time creative
      // such as berlin_locals are -64% of the time not techno_lovers
      const b2a = Math.round(
        personB.minCount * relationshipsB[personA.attribute]
      )
      const a2b = Math.round(
        personA.minCount * relationshipsA[personB.attribute]
      )

      // console.log(`${personB.attribute} has ${b2a} ${personA.attribute} people...`)
      // console.log(`${personA.attribute} has ${a2b} ${personB.attribute} people...`)

      personA.combos[personB.attribute] = a2b
      personB.combos[personA.attribute] = b2a
    })
  })

  // map over one more time and calculate bounds
  people.forEach((personA) => {
    for (const attributeB in personA.combos) {
      const totalPeopleBToA = personA.combos[attributeB]
      if (totalPeopleBToA < 0) {
        personA.upperBound -= totalPeopleBToA
      } else {
        personA.lowerBound -= totalPeopleBToA
      }
    }
  })

  return people
}

type People = ReturnType<typeof getPeopleFromConstraints>[0]

function clamp(lowerBound: number, value: number, upperBound: number): number {
  return Math.max(lowerBound, Math.min(upperBound, value))
}

/**
 *  Custom implementation.
 */
export class ColinsBouncer implements BergainBouncer {
  public lastScore: number[] = []
  public lastAttributes?: ScenarioAttributes
  public lastThreshold = 0.5
  public lastTotal = 0
  public keys: string[]
  public people: People[]

  public get frequencies() {
    return this.state.game.attributeStatistics.relativeFrequencies
  }

  public get correlations() {
    return this.state.game.attributeStatistics.correlations
  }

  public get totalSpotsLeft() {
    return config.maxAdmissions - this.totalAdmitted
  }

  public get toalRejected() {
    return this.totalProcessed - this.totalAdmitted
  }

  public totalAdmitted = 0
  public totalProcessed = 0

  isAtMaxCapacity() {
    return this.totalAdmitted >= config.maxAdmissions
  }

  isFinishedWithQuotas() {
    return this.people.every((person) => person.isComplete())
  }

  constructor(public state: GameState) {
    this.keys = state.game.constraints.map((constraint) => constraint.attribute)
    this.people = getPeopleFromConstraints(state)
  }

  getThreshold() {
    const ratio = this.totalAdmitted / this.totalSpotsLeft
    return clamp(0.41, ratio, 0.91)
  }

  // override methods

  admit(next: GameStatusRunning<ScenarioAttributes>): boolean {
    if (next.status !== 'running') return false
    this.totalProcessed++
    if (this.isAtMaxCapacity()) return false
    if (this.isFinishedWithQuotas()) return true

    const score = this.people.map((person) =>
      person.getScore(next.nextPerson.attributes)
    )
    const threshold = this.getThreshold()
    const totalScore = score.reduce((a, b) => a * b, 1.0) / score.length

    this.lastAttributes = next.nextPerson.attributes
    this.lastScore = score
    this.lastThreshold = threshold
    this.lastTotal = totalScore

    const shouldAdmit = totalScore >= threshold

    if (shouldAdmit) {
      this.totalAdmitted++
      this.people.forEach((person) =>
        person.didAdmit(next.nextPerson.attributes)
      )
    }

    return shouldAdmit
  }

  getOutput() {
    return {
      ...this.state,
      ...this.getProgress(),
    }
  }

  getProgress() {
    const combined = this.people.reduce((output, person) => {
      return {
        ...output,
        ...person.getOutput(),
      }
    }, {})

    return {
      ...combined,
      admissionRate: round(this.totalProcessed / this.totalSpotsLeft),
      weights: this.lastScore,
      totalSpotsLeft: this.totalSpotsLeft,
      totalAdmissions: this.totalAdmitted,
      totalRejections: this.toalRejected,
      threshold: this.lastThreshold,
      score: this.lastTotal,
    }
  }
}
