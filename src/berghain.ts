import https from 'https'
import http from 'http'

import type {
  GameStatus,
  Game,
  GameState,
  GameStatusRunning,
  ScenarioAttributes,
  GameStatusCompleted,
  GameStatusFailed,
} from './types'
import axios from 'axios'

export type ListenLabsConfig = {
  uniqueId: string
  baseUrl: string
  scenario: '1' | '2' | '3'
}

export interface BerghainBouncer {
  admit(next: GameStatusRunning<ScenarioAttributes>): boolean
  getProgress(): any
  getOutput(lastStatus: GameStatusCompleted | GameStatusFailed): any | Promise<any>
}

class MissingCurrentState extends Error {
  constructor() {
    super('MISSING_CURRENT_GAME: Make sure to create or load a game first!')
  }
}

class MissingBouncer extends Error {
  constructor() {
    super('MISSING_BOUNCER: Make sure to call .withBouncer(initializer) first!')
  }
}

class NetworkRequestFailed extends Error {
  constructor(response: Response, url: URL) {
    super(`NETWORK REQUEST FAILED (${response.status}): ${response.statusText} for ${url.href}`)
  }
}

const DEFAULT_CONFIG: ListenLabsConfig = {
  uniqueId: 'c9d5d12b-66e6-4d03-909b-5825fac1043b',
  baseUrl: 'https://berghain.challenges.listenlabs.ai/',
  scenario: '3',
}

const httpsAgent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 30_000,
  maxSockets: 1, // Single persistent connection
  maxFreeSockets: 1,
  timeout: 60_000,
  secureProtocol: 'TLSv1_2_method', // Avoid SSL negotiation overhead
})

const httpAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 1,
  maxFreeSockets: 1,
  timeout: 60000,
})

// debugging

export function prettyPrint(obj: any) {
  console.log(JSON.stringify(obj, null, 2))
}

// persistance

export async function saveGameFile<T extends {}>(state: GameState) {
  const file = Bun.file(state.file, { type: 'json' })
  await file.write(JSON.stringify(state, null, 2))
}

export async function loadGameFile(props: { file: string }): Promise<GameState> {
  const file = Bun.file(props.file, { type: 'json' })
  return await file.json()
}

// game controller

export class Berghain {
  /**
   *  Helper for initializing a new instance.
   */
  static initialize({ scenario }: Pick<ListenLabsConfig, 'scenario'>) {
    return new Berghain({ ...DEFAULT_CONFIG, scenario })
  }

  // instance variables

  private createBouncer?: (initialState: GameState) => BerghainBouncer | Promise<BerghainBouncer>
  private bouncer?: BerghainBouncer
  private current?: GameState
  private maxRetries = 1

  constructor(private config: ListenLabsConfig) {}

  get nextIndex(): number {
    return this.current!.status.nextPerson!.personIndex
  }

  /**
   * Load a previosely saved game file.
   */
  async loadSavedGame(props: { file: string }): Promise<GameState> {
    const savedGame = await loadGameFile(props)
    console.warn('====================== 💾 ======================')
    console.log('[game] loading from file:', props.file)
    prettyPrint(savedGame)
    this.current = savedGame
    return this.current
  }

  /**
   * Set the bouncer strategy to be used in the game.
   */
  withBouncer(initializer: (state: GameState) => BerghainBouncer | Promise<BerghainBouncer>): this {
    this.createBouncer = initializer
    return this
  }

  /**
   * Creates a new game for the specified scenario and loads the first person
   * in line, will also save the file to disk.
   */
  async startNewGame(): Promise<this> {
    const endpoint = new URL('/new-game', this.config.baseUrl)
    endpoint.searchParams.set('playerId', this.config.uniqueId)
    endpoint.searchParams.set('scenario', this.config.scenario)
    const game = await this.fetch<Game>(endpoint)

    this.current = {
      scenario: this.config.scenario,
      game,
    } as GameState

    console.log('[game] created new game:')
    prettyPrint(game)

    if (!game || !game.gameId) {
      throw game
    }

    const status = await this.acceptOrRejectThenGetNext({
      index: 0,
      accept: true,
    })

    // create initial game state
    const initialState: GameState = {
      file: `./data/scenario-${this.config.scenario}-${game.gameId}.json`,
      scenario: this.config.scenario,
      game,
      status,
      output: {},
    }
    // log file to output
    prettyPrint(initialState)
    this.current = initialState
    return this.runGameLoop()
  }

  async runGameLoop(): Promise<this> {
    if (!this.current) throw new MissingCurrentState()
    if (!this.createBouncer) throw new MissingBouncer()
    this.bouncer = await this.createBouncer(this.current)

    // reset the max retries on each run
    this.maxRetries = 1

    try {
      while (this.current.status.status === 'running') {
        // check if we should accept or reject
        const admit = this.bouncer.admit(this.current.status)

        // then check the next person.
        this.current.status = await this.acceptOrRejectThenGetNext({
          index: this.nextIndex,
          accept: admit,
        })
        if (this.current.status.status !== 'running') break
        console.log('- '.repeat(32))
        console.log(this.bouncer.getProgress()) // better to console here
      }

      if (this.current.status.status === 'failed') {
        console.warn('====================== ❌ ======================')
        console.log(this.bouncer.getOutput(this.current.status))
        console.log(this.bouncer.getProgress())
        prettyPrint(this.current.status)
        return this
      }

      if (this.current.status.status === 'completed') {
        console.warn('====================== ✅ ======================')
        console.log(this.bouncer.getOutput(this.current.status))
        console.log(this.bouncer.getProgress())
        prettyPrint(this.current.status)
        return this
      }
    } catch (e) {
      console.warn('====================== ⚠️ ======================')
      console.warn(e)
      if (e && typeof e === 'object' && 'data' in e) {
        console.warn(e.data)
      }
    } finally {
      return this
    }
  }

  /**
   *  Call this method to accept or reject the current in line, then get
   *  the next person and update the sate.
   *
   *  @url /decide-and-next?gameId=uuid&personIndex=0&accept=true
   */
  private async acceptOrRejectThenGetNext(props: { index: number; accept: boolean }): Promise<GameStatus> {
    try {
      if (!this.current?.game.gameId) throw new Error('Missing game id!')
      const endpoint = new URL('/decide-and-next', this.config.baseUrl)
      endpoint.searchParams.set('gameId', this.current.game.gameId)
      endpoint.searchParams.set('personIndex', String(props.index))
      endpoint.searchParams.set('accept', props.accept ? 'true' : 'false')
      // console.log(endpoint.href)
      return (await axios.get<GameStatus>(endpoint.href)).data
    } catch (e) {
      if (--this.maxRetries < 0) throw e
      console.warn(e)
      return await this.acceptOrRejectThenGetNext(props)
    }
  }

  /**
   * Simple fetch wrapper for fetching JSON, does not handle retries.
   */
  private async fetch<T>(url: URL): Promise<T> {
    const resp = await axios.get(url.href, {
      httpsAgent,
      httpAgent,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        Connection: 'keep-alive',
      },
    })

    resp.data
    if (resp.status >= 300 || resp.status < 200) {
      throw new NetworkRequestFailed({ status: resp.status, statusText: resp.statusText } as any, url)
    } else {
      return resp.data as T
    }
  }
}
