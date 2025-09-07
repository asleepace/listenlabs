import { acceptOrRejectThenGetNext } from './src/play-game'
import { initialize, loadGameFile, saveGameFile, prettyPrint } from './src/disk'

import { NightclubGameCounter } from './matrix'

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

type PersonAttributesScenario1 = {
  well_dressed: true
  young: true
}

type PersonAttributesScenario2 = {
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

export type GameStatus =
  | {
      status: 'running'
      admittedCount: number
      rejectedCount: number
      nextPerson: Person
    }
  | {
      status: 'completed'
      rejectedCount: number
      nextPerson: null
    }
  | {
      status: 'failed'
      reason: string
      nextPerson: null
    }

export type GameState = {
  file: string
  game: Game
  status: GameStatus
  output?: any
}

export type Keys = keyof Person['attributes']

console.log('================ starting ================')
console.warn('[game] triggering new game!')

const game = await initialize({ scenario: '3' })
console.warn('[game] game file:', game)

// const savedGame = await loadGameFile({ file })
const counter = new NightclubGameCounter(game)
await runGameLoop(game.status).catch(console.warn)

console.log('=========================================')

//
// ====================== game loop ======================
//

async function runGameLoop(nextStatus: GameStatus): Promise<boolean> {
  if (nextStatus.status !== 'running') throw new Error('Invalid status!')

  const accept = counter.admit(nextStatus)

  const next = await acceptOrRejectThenGetNext({
    game: counter.state.game,
    index: nextStatus.nextPerson.personIndex,
    accept,
  })

  console.log(counter.getProgress())

  // if (next.status !== 'completed') {
  //   saveGameFile({
  //     ...,game,
  //     output: counter.getGameData(),
  //   }).catch(() => {})
  // }

  if (next.status === 'failed') {
    console.warn('================ ❌ ================')
    throw next
  }

  if (next.status === 'completed') {
    console.log('================ ✅ ================')
    const scoreFile = Bun.file(`./scores-${+new Date()}.json`)
    // scoreFile.write(JSON.stringify(counter.getGameData(), null, 2))
    console.log(next)
    return true
  }

  return runGameLoop(next)
}
