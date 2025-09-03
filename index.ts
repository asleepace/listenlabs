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

/**
 * Main function where we should let person in or not...
 */
function shouldLetPersonIn({ status: next, metrics }: GameState): boolean {
  if (next.nextPerson == null) return false;

  const { nextPerson } = next

  if (hasAllAttributes(nextPerson)) return true

  const totalPeople = next.admittedCount

  const hasAnyAttribute = hasSomeAttribute(nextPerson)
  const isMoreYoungPeople = metrics.totalYoung > metrics.totalWellDressed
  const isMoreWellDressed = !isMoreYoungPeople

  switch (true) {
    case totalPeople < 99:
      return true

    case totalPeople < 250:
      return hasAnyAttribute

    case totalPeople < 800: {
      if (isMoreYoungPeople && nextPerson.attributes.well_dressed) return true
      if (isMoreWellDressed && nextPerson.attributes.young) return true
      return hasAnyAttribute
    }
    
    case totalPeople < 800: {
      if (isMoreYoungPeople && nextPerson.attributes.well_dressed) return true
      if (isMoreWellDressed && nextPerson.attributes.young) return true
      return false
    }
    default:
      return hasAnyAttribute
  }
}


function updateGameState(prevState: GameState, nextStatus: GameStatus, accepted: boolean): GameState {
  const nextPerson = nextStatus.nextPerson
  const nextTotalWellDressed = accepted && nextPerson?.attributes.well_dressed ? 1 : 0;
  const nextTotalYoung = accepted && nextPerson?.attributes.young ? 1 : 0;

  return {
    ...prevState,
    status: nextStatus,
    metrics: {
      totalWellDressed: prevState.metrics.totalWellDressed + nextTotalWellDressed,
      totalYoung: prevState.metrics.totalYoung + nextTotalYoung,
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
    throw new Error("Uh oh game failed...");
  }

  const nextMetrics = accept === false ? state.metrics : {
    totalWellDressed: state.metrics.totalWellDressed - (next.nextPerson.attributes.well_dressed ? 1 : 0),
    totalYoung: state.metrics.totalYoung - (next.nextPerson.attributes.young ? 1 : 0)
  }

  const nextState: GameState = updateGameState(state, next, accept)
  prettyPrint(nextState)

  await saveGameFile(nextState);
  return await runGameLoop(nextState);
}

// ================= GAME START ===================== //

await initialize({ scenario: '1' })
const savedGame = await loadGameFile();
await runGameLoop(savedGame);
