import { config } from "./src/config";
import { createGame } from "./src/create-game";
import { acceptOrRejectThenGetNext } from "./src/play-game";

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

type GameState = {
  game: Game;
  status: GameStatus;
  metrics: {
    totalWellDressed: number;
    totalYoung: number;
  };
};

function prettyPrint(obj: any) {
  console.log(JSON.stringify(obj, null, 2));
}

async function saveGameFile(state: GameState) {
  const file = Bun.file("./state.json", { type: "json" });
  await file.write(JSON.stringify(state, null, 2));
}

async function loadGameFile() {
  const file = Bun.file("./state.json", { type: "json" });
  const json = (await file.json()) as GameState;
  return json;
}

/**
 * Start a new game and save file.
 */
async function initialize() {
  const game = await createGame({ scenario: "1" });
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
      totalWellDressed: 600,
      totalYoung: 600,
    },
  });
}

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
  const hasAllAttributes = Object.values(next.nextPerson.attributes).every(
    (attr) => attr
  );

  if (hasAllAttributes) return true

  const hasAnyAttribute = hasSomeAttribute(next.nextPerson)
  const totalPeopleLeft = 10_000 - next.nextPerson.personIndex
  const isCloseToClose = totalPeopleLeft < 1_000 || next.admittedCount > 800
  const isStartOfNight = !isCloseToClose && (totalPeopleLeft > 9_000 || next.admittedCount < 100)

  if (isStartOfNight && (hasAnyAttribute || Math.random() < 0.2)) {
    return true
  }

  const hasLessWellDressed = metrics.totalWellDressed >= metrics.totalYoung
  const hasLessYoungPeople = metrics.totalYoung >= metrics.totalWellDressed
  const isYoung = next.nextPerson.attributes.young
  const isWellDressed = next.nextPerson.attributes.well_dressed 

  if (metrics.totalWellDressed > 0 && isWellDressed && (hasLessWellDressed || isCloseToClose || isStartOfNight)) {
    return true
  }

  if (metrics.totalYoung > 0 && isYoung && (hasLessYoungPeople || isCloseToClose || isStartOfNight)) {
    return true
  }

  return isCloseToClose ? hasAllAttributes : hasSomeAttribute(next.nextPerson)
}

/**
 * # Game Loop
 *
 *
 */
async function runGameLoop(state: GameState): Promise<boolean> {
  const accept = shouldLetPersonIn(state);

  const next = await acceptOrRejectThenGetNext({
    game: state.game,
    index: state.status.nextPerson?.personIndex ?? 0,
    accept,
  });

  if (next.status === "completed") return true;
  if (next.status === "failed") {
    throw new Error("Uh oh game failed...");
  }

  const nextMetrics = accept === false ? state.metrics : {
    totalWellDressed: state.metrics.totalWellDressed - (next.nextPerson.attributes.well_dressed ? 1 : 0),
    totalYoung: state.metrics.totalWellDressed - (next.nextPerson.attributes.young ? 1 : 0)
  }

  const nextState: GameState = {
    ...state,
    status: next,
    metrics: nextMetrics
  };

  prettyPrint(nextState)

  await saveGameFile(nextState);
  return await runGameLoop(nextState);
}

// ================= GAME START ===================== //

await initialize()
const savedGame = await loadGameFile();
await runGameLoop(savedGame);
