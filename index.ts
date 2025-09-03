import { config } from "./src/config";
import { createGame } from "./src/create-game";
import { acceptOrRejectThenGetNext } from "./src/play-game";
import { initialize, loadGameFile, saveGameFile, prettyPrint } from "./src/disk";

export type GameConstraints = {
  attribute: string;
  minCount: number;
};

export type Game = {
  gameId: string;
  constraints: GameConstraints[];
  attributeStatistics: {
    relativeFrequencies: {
      [attributeId: string]: number; // 0.0-1.0
    };
    correlations: {
      [attributeId1: string]: {
        [attributeId2: string]: number; // -1.0-1.0
      };
    };
  };
};

type PersonAttributesScenario1 = {
  well_dressed: true;
  young: true;
};

export type Person<T = PersonAttributesScenario1> = {
  personIndex: number;
  attributes: T;
};

export type GameStatus =
  | {
      status: "running";
      admittedCount: number;
      rejectedCount: number;
      nextPerson: Person;
    }
  | {
      status: "completed";
      rejectedCount: number;
      nextPerson: null;
    }
  | {
      status: "failed";
      reason: string;
      nextPerson: null;
    };

export type GameState = {
  file: string;
  game: Game;
  status: GameStatus;
  metrics: {
    totalWellDressed: number;
    totalYoung: number;
    winner: boolean;
    score: number;
  };
};


const sleep = (time: number) => new Promise<void>((resolve) => {
  setTimeout(() => resolve(), time)
})


function hasAllAttributes(person: Person) {
  return Object.values(person.attributes).every(Boolean)
}

function hasSomeAttribute(person: Person) {
  return Object.values(person.attributes).some(Boolean)
}

let totalWellDressed = 0
let totalYoung = 0

/**
 * Main function where we should let person in or not...
 */
function shouldLetPersonIn({ status: next, metrics }: GameState): boolean {
  if (next.nextPerson == null) return false;

  const totalCount = 'admittedCount' in next ? next.admittedCount : 0

  const { nextPerson } = next

  // NOTE: count total number of people
  if (next.nextPerson.attributes.well_dressed) {
    totalWellDressed++
  }
  if (next.nextPerson.attributes.young) {
    totalYoung++
  }

  // calculate totals

  if (totalYoung > 600 && totalWellDressed > 600) {
    return true
  }

  if (nextPerson.attributes.well_dressed && nextPerson.attributes.young) {
    return true
  }

  if (totalYoung < 595 && nextPerson.attributes.young) {
    return true
  }

  if (totalWellDressed < 595 && nextPerson.attributes.well_dressed) {
    return true
  }

  if (totalCount > 975) {
    if (totalYoung < 600 && nextPerson.attributes.young) return true
    if (totalWellDressed < 600 && nextPerson.attributes.well_dressed) return true
    return false 
  }

  const hasOneOrMoreAttribute = nextPerson.attributes.well_dressed || nextPerson.attributes.young 

  return hasOneOrMoreAttribute || nextPerson.personIndex % 8 === 0
}


function updateGameState(prevState: GameState, nextStatus: GameStatus): GameState {
  return {
    ...prevState,
    status: nextStatus,
    metrics: {
      totalWellDressed: totalWellDressed,
      totalYoung: totalYoung,
      winner: false,
      score: 'rejectedCount' in nextStatus ? nextStatus.rejectedCount : 0
     }
  }
}

/**
 * # Game Loop
 *
 *
 */
async function runGameLoop(state: GameState): Promise<boolean> {
  const accept = shouldLetPersonIn(state);
  const index = state.status.nextPerson?.personIndex ?? 0

  const next = await acceptOrRejectThenGetNext({
    game: state.game,
    index: index,
    accept,
  });

  if (next.status === "completed") {
    const nextState: GameState = updateGameState(state, next)
    await saveGameFile({ ...nextState, metrics: { totalWellDressed, totalYoung, winner: true, score: next.rejectedCount }})
    return true;
  }
  if (next.status === "failed") {
    await Bun.file(state.file).delete()
    throw next
  }

  const nextState: GameState = updateGameState(state, next)
  prettyPrint(nextState)

  await saveGameFile(nextState);
  return await runGameLoop(nextState);
}

// ================= GAME START ===================== //

async function triggerNewGame() {
  console.warn('[game] triggering new game!')
  const file = await initialize({ scenario: '1' })
  const savedGame = await loadGameFile({ file })
  await runGameLoop(savedGame);
  return savedGame
}

async function* createGameGenerator() {
  do {
    try {
      yield await triggerNewGame()
    } catch (e) {
      console.warn('[error] e:', e)
    } finally {
      await sleep(60_000)
    }
  } while (true)
}


const gameIterator = createGameGenerator()

for await (const result of gameIterator) {
  console.log(result.metrics)
}