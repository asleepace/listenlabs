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
  game: Game;
  status: GameStatus;
  metrics: {
    totalWellDressed: number;
    totalYoung: number;
  };
};



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

  const { nextPerson } = next

  if (next.nextPerson.attributes.well_dressed) {
    totalWellDressed++
  }

  if (next.nextPerson.attributes.young) {
    totalYoung++
  }

  if (totalWellDressed > 600 && totalYoung > 600) {
    return true
  }

  if (totalYoung > 600) {
    return nextPerson.attributes.well_dressed
  }

  if (totalWellDressed > 600) {
    return nextPerson.attributes.young
  }

  if (nextPerson.attributes.well_dressed && nextPerson.attributes.young) {
    return true
  }

  if (totalWellDressed < 600 && nextPerson.attributes.well_dressed) {
    return true
  }

  if (totalYoung < 600 && nextPerson.attributes.young) {
    return true
  }

  return nextPerson.attributes.well_dressed || nextPerson.attributes.young || Math.random() < 0.05
}


function updateGameState(prevState: GameState, nextStatus: GameStatus): GameState {
  return {
    ...prevState,
    status: nextStatus,
    metrics: {
      totalWellDressed: totalWellDressed,
      totalYoung: totalYoung,
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

  if (next.status === "completed") return true;
  if (next.status === "failed") {
    throw next
  }

  const nextState: GameState = updateGameState(state, next)
  prettyPrint({ ...nextState, stats: { totalWellDressed, totalYoung} })

  await saveGameFile(nextState);
  return await runGameLoop(nextState);
}

// ================= GAME START ===================== //

await initialize({ scenario: '1' })
const savedGame = await loadGameFile();
await runGameLoop(savedGame);
