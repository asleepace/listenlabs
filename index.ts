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

  if (totalYoung < 590 && nextPerson.attributes.young) {
    return true
  }

  if (totalWellDressed < 595 && nextPerson.attributes.well_dressed) {
    return true
  }

  const hasOneOrMoreAttribute = nextPerson.attributes.well_dressed || nextPerson.attributes.young 

  if (totalCount < 900) {
    return hasOneOrMoreAttribute || nextPerson.personIndex % 16 === 0
  } else {
    return hasOneOrMoreAttribute
  }
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
  prettyPrint(nextState)

  await saveGameFile(nextState);
  return await runGameLoop(nextState);
}

// ================= GAME START ===================== //

const file = await initialize({ scenario: '1' })
const savedGame = await loadGameFile({ file })
await runGameLoop(savedGame);
