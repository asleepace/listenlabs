import { config } from "./config";
import type { Game, GameConstraints } from "..";


/**
 * /scenario=1&playerId=c9d5d12b-66e6-4d03-909b-5825fac1043b
 */
export async function createGame(props: { scenario: '1' | '2' | '3' }): Promise<Game> {
  const endpoint = new URL('/new-game', config.baseUrl)
  endpoint.searchParams.set('playerId', config.uniqueId)
  endpoint.searchParams.set('scenario', props.scenario)

  const response = await fetch(endpoint, {
    headers: {
      'content-type': 'application/json'
    }
  })

  if (!response.ok) throw new Error(`${response.status} ${response.statusText} - Failed to create new game`)
  
  const game = await response.json() as Game
  console.log({ game })
  return game
}