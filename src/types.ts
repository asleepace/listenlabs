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

export interface PersonAttributes {
  [key: string]: boolean
}

export interface PersonAttributesScenario1 extends PersonAttributes {
  well_dressed: true
  young: true
}

export interface PersonAttributesScenario2 extends PersonAttributes {
  techno_lover: boolean
  well_connected: boolean
  creative: boolean
  berlin_local: boolean
}

export interface PersonAttributesScenario3 extends PersonAttributes {
  underground_veteran: boolean
  international: boolean
  fashion_forward: boolean
  queer_friendly: boolean
  vinyl_collector: boolean
  german_speaker: boolean
}

export interface Person<T extends PersonAttributes> {
  personIndex: number
  attributes: T
}

export type GameStatusRunning<
  T extends
    | PersonAttributesScenario1
    | PersonAttributesScenario2
    | PersonAttributesScenario3
> = {
  status: 'running'
  admittedCount: number
  rejectedCount: number
  nextPerson: Person<T>
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
  | GameStatusRunning<ScenarioAttributes>
  | GameStatusCompleted
  | GameStatusFailed

export type GameState<T = any> = {
  file: string
  game: Game
  status: GameStatus
  scenario: '1' | '2' | '3'
  output?: T
  timestamp?: string
}

export type ScenarioAttributes =
  | PersonAttributesScenario1
  | PersonAttributesScenario2
  | PersonAttributesScenario3

export type Keys = keyof ScenarioAttributes
