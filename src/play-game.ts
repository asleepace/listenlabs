import { config } from "./config";
import { type Person, type Game, type GameStatus } from "..";

/**
 *
 * /decide-and-next?gameId=uuid&personIndex=0&accept=true
 */
export async function acceptOrRejectThenGetNext(props: {
  index: number;
  accept: boolean;
  game: Game;
}): Promise<GameStatus> {
  try {
    const endpoint = new URL("/decide-and-next", config.baseUrl);
    endpoint.searchParams.set("gameId", props.game.gameId);
    endpoint.searchParams.set("personIndex", String(props.index));
    endpoint.searchParams.set("accept", props.accept ? "true" : "false");

    console.log(endpoint.href)
    const response = await fetch(endpoint);
    if (!response.ok)
      throw new Error(`${response.status} - ${response.statusText}`);
    const game = (await response.json()) as GameStatus;
    console.log(
      JSON.stringify(
        {
          href: endpoint.href,
          game: game,
        },
        null,
        2
      )
    );
    return game;
  } catch (e) {
    console.warn(e);
    return await acceptOrRejectThenGetNext(props);
  }
}
