import { type GameState } from "..";
import { createGame } from "./create-game";
import { acceptOrRejectThenGetNext } from './play-game'

export function prettyPrint(obj: any) {
  console.log(JSON.stringify(obj, null, 2));
}

export async function saveGameFile(state: GameState) {
  const file = Bun.file("./state.json", { type: "json" });
  await file.write(JSON.stringify(state, null, 2));
}

export async function loadGameFile() {
  const file = Bun.file("./state.json", { type: "json" });
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

  await saveGameFile({
    game,
    status,
    metrics: {
      totalWellDressed: 0,
      totalYoung: 0,
    },
  });
}
