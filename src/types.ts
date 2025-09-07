export type GameConstraints = {
  attribute: string
  minCount: number
}

export type Game = {
  gameId: string
  constraints: GameConstraints[]
  attributeStatistics: {
    relativeFrequencies: {
      [attributeId: string]: number // 0.0-1.0
    }
    correlations: {
      [attributeId1: string]: {
        [attributeId2: string]: number // -1.0-1.0
      }
    }
  }
}

export type PersonAttributesScenario1 = {
  well_dressed: true
  young: true
}

export type PersonAttributesScenario2 = {
  techno_lover: boolean
  well_connected: boolean
  creative: boolean
  berlin_local: boolean
}

export type PersonAttributesScenario3 = {
  underground_veteran: boolean
  international: boolean
  fashion_forward: boolean
  queer_friendly: boolean
  vinyl_collector: boolean
  german_speaker: boolean
}

export type Person<T = PersonAttributesScenario3> = {
  personIndex: number
  attributes: T
}

export type GameStatusRunning = {
  status: 'running'
  admittedCount: number
  rejectedCount: number
  nextPerson: Person
}

export type GameStatusCompleted = {
  status: 'completed'
  rejectedCount: number
  nextPerson: null
}

export type GameStatusFailed = {
  status: 'failed'
  reason: string
  nextPerson: null
}

export type GameStatus =
  | GameStatusRunning
  | GameStatusCompleted
  | GameStatusFailed

export type GameState = {
  file: string
  game: Game
  status: GameStatus
  output?: any
}

export type Keys = keyof Person['attributes']
