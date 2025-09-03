import { type GameState } from "..";
import { createGame } from "./create-game";
import { acceptOrRejectThenGetNext } from "./play-game";

export function prettyPrint(obj: any) {
  console.log(JSON.stringify(obj, null, 2));
}

export async function saveGameFile(state: GameState) {
  const file = Bun.file(state.file, { type: "json" });
  await file.write(JSON.stringify(state, null, 2));
}

export async function loadGameFile(props: { file: string }) {
  const file = Bun.file(props.file, { type: "json" });
  const json = (await file.json()) as GameState;
  return json;
}

/**
 * Start a new game and save file.
 */
export async function initialize(props: Parameters<typeof createGame>[0]) {
  const game = await createGame(props);
  const status = await acceptOrRejectThenGetNext({
    index: 0,
    accept: true,
    game,
  });

  prettyPrint({ game, status });

  const file = `./game-${game.gameId}.json`

  await saveGameFile({
    file,
      game,
      status,
      metrics: {
        totalWellDressed: 0,
        totalYoung: 0,
        winner: false,
        score: 0,
      },
  });

  return file
}
