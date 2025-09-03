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
    percentWellDressed: number;
    percentYoung: number;
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

  const isWithinRange = Math.abs(metrics.totalYoung - metrics.totalWellDressed) < 10

  switch (true) {
    case totalPeople < 100:
      return Math.random() < 0.5

    case totalPeople < 250:
      return hasAnyAttribute

    case totalPeople < 500: {
      if ((isMoreYoungPeople || isWithinRange) && nextPerson.attributes.well_dressed) return true
      if ((isMoreWellDressed || isWithinRange) && nextPerson.attributes.young) return true
      return hasAnyAttribute || Math.random() < 0.25
    }
    
    case totalPeople < 750: {
      if ((isMoreYoungPeople || isWithinRange) && nextPerson.attributes.well_dressed) return true
      if ((isMoreWellDressed || isWithinRange) && nextPerson.attributes.young) return true
      return (hasAnyAttribute && Math.random() < 0.1)
    }

    default: {
      if (isMoreYoungPeople && nextPerson.attributes.well_dressed) return true
      if (isMoreWellDressed && nextPerson.attributes.young) return true
      return false
    }
  }
}


function updateGameState(prevState: GameState, nextStatus: GameStatus, accepted: boolean): GameState {
  const nextPerson = nextStatus.nextPerson

  const nextTotalWellDressed = prevState.metrics.totalWellDressed + (accepted && nextPerson?.attributes.well_dressed ? 1 : 0);
  const nextTotalYoung =  prevState.metrics.totalYoung + (accepted && nextPerson?.attributes.young ? 1 : 0);
  const admittedCount = 'admittedCount' in nextStatus ? nextStatus.admittedCount : 1

  return {
    ...prevState,
    status: nextStatus,
    metrics: {
      totalWellDressed: nextTotalWellDressed,
      totalYoung: nextTotalYoung,
      percentWellDressed: 1 - (nextTotalWellDressed / admittedCount),
      percentYoung: 1 - (nextTotalYoung / admittedCount)
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
