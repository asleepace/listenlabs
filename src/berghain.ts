import type { GameStatus, Game, GameState, GameStatusRunning } from './types'

export type ListenLabsConfig = {
  uniqueId: string
  baseUrl: string
  scenario: '1' | '2' | '3'
}

export interface BergainBouncer {
  admit(next: GameStatusRunning): boolean
  getProgress(): any
  getOutput(): any
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
    super(
      `NETWORK REQUEST FAILED (${response.status}): ${response.statusText} for ${url.href}`
    )
  }
}

const DEFAULT_CONFIG: ListenLabsConfig = {
  uniqueId: 'c9d5d12b-66e6-4d03-909b-5825fac1043b',
  baseUrl: 'https://berghain.challenges.listenlabs.ai/',
  scenario: '3',
}

// debugging

export function prettyPrint(obj: any) {
  console.log(JSON.stringify(obj, null, 2))
}

// persistance

export async function saveGameFile<T extends {}>(state: GameState) {
  const file = Bun.file(state.file, { type: 'json' })
  await file.write(JSON.stringify(state, null, 2))
}

export async function loadGameFile(props: {
  file: string
}): Promise<GameState> {
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

  private createBouncer?: (initialState: GameState) => BergainBouncer
  private bouncer?: BergainBouncer
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
  withBouncer(initializer: (state: GameState) => BergainBouncer): this {
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

    // @ts-ignore
    this.current = {
      game,
    }

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
      game,
      status,
      output: {},
    }
    // log file to output
    prettyPrint(initialState)
    this.current = initialState
    return this.runGameLoop()
  }

  async saveGame() {
    if (!this.current || !this.bouncer) return
    await saveGameFile({
      ...this.current,
      output: this.bouncer.getProgress(),
    })
  }

  async runGameLoop(): Promise<this> {
    if (!this.current) throw new MissingCurrentState()
    if (!this.createBouncer) throw new MissingBouncer()
    this.bouncer = this.createBouncer(this.current)

    // reset the max retries on each run
    this.maxRetries = 10

    try {
      const status = this.current.status.status

      while (status === 'running') {
        const admit = this.bouncer.admit(this.current.status)
        this.current.status = await this.acceptOrRejectThenGetNext({
          index: this.nextIndex,
          accept: admit,
        })
        if (this.current.status.status !== 'running') break
        prettyPrint(this.bouncer.getProgress())
      }

      if (this.current.status.status === 'failed') {
        console.warn('====================== ❌ ======================')
        prettyPrint(this.bouncer.getProgress())
        prettyPrint(this.current.status)
      }

      if (this.current.status.status === 'completed') {
        console.warn('====================== ✅ ======================')
        prettyPrint(this.bouncer.getProgress())
        prettyPrint(this.current.status)
        await this.saveGame()
      }
    } catch (e) {
      console.warn('====================== ⚠️ ======================')
      console.warn(e)
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
  private async acceptOrRejectThenGetNext(props: {
    index: number
    accept: boolean
  }): Promise<GameStatus> {
    try {
      if (!this.current?.game.gameId) throw new Error('Missing game id!')
      const endpoint = new URL('/decide-and-next', this.config.baseUrl)
      endpoint.searchParams.set('gameId', this.current.game.gameId)
      endpoint.searchParams.set('personIndex', String(props.index))
      endpoint.searchParams.set('accept', props.accept ? 'true' : 'false')
      console.log(endpoint.href)
      return await this.fetch<GameStatus>(endpoint)
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
    const response = await fetch(url, {
      headers: {
        'content-type': 'application/json',
      },
    })
    if (!response.ok) {
      throw new NetworkRequestFailed(response, url)
    } else {
      return (await response.json()) as T
    }
  }
}
