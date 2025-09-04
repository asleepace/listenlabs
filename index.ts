import { config } from "./src/config";
import { createGame } from "./src/create-game";
import { acceptOrRejectThenGetNext } from "./src/play-game";
import {
  initialize,
  loadGameFile,
  saveGameFile,
  prettyPrint,
} from "./src/disk";

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

function getSortedAttributes(
  attributes: Record<keyof Person["attributes"], number>
): [keyof Person["attributes"], number][] {
  return Object.entries(attributes)
    .filter((attr) => attr[1] > 0)
    .sort((attr1, attr2) => attr1[1] - attr2[1])
    .reverse() as [keyof Person["attributes"], number][];
}

function doesSetBHaveAllSetA<T>(setA: Set<T>, setB: Set<T>): boolean {
  for (const key of setB.values()) {
    if (setA.has(key)) continue;
    return false;
  }
  return true;
}

const getKeys = (attr: Person["attributes"]) => {
  return new Set(
    Object.entries(attr)
      .filter(([key, attr]) => attr === true)
      .map(([key]) => key) as (keyof Person["attributes"])[]
  );
};

class GameCounter {
  public data: Record<keyof Person["attributes"], number> = {
    underground_veteran: 0,
    international: 0,
    fashion_forward: 0,
    queer_friendly: 0,
    vinyl_collector: 0,
    german_speaker: 0,
  };

  private canLetAnyoneIn = false;
  private totalEntries = 0;
  private totalAttributes = 0;

  constructor(initialState: GameState) {
    // initialize with game constraints
    for (const constraint of initialState.game.constraints) {
      this.data[constraint.attribute as keyof Person["attributes"]] = Number(
        constraint.minCount
      );
    }

    this.totalAttributes = Object.keys(this.data).length;
  }

  public count(person: Person) {
    if (this.canLetAnyoneIn) return true;
    for (const [key, value] of Object.entries(person.attributes) as [
      keyof Person["attributes"],
      boolean
    ][]) {
      if (value === false) continue;
      this.data[key]--;
    }

    this.canLetAnyoneIn = Object.values(this.data).every((val) => val < 0);
  }

  get minPeopleToMeetQuota(): number {
    return Object.values(this.data).reduce((total, current) => {
      return Math.max(total, current);
    }, 0);
  }

  public shouldLetIn(person: Person): boolean {
    /**
     * NOTE: this is an inner funcion:
     * @returns
     */
    const determineToLetPersonInOrSomething = (): boolean => {
      // sort attributes in descending order (greatest to smallest)
      const sortedAttributes = getSortedAttributes(this.data);

      console.log(sortedAttributes);

      // the quotas have been met
      if (sortedAttributes.length === 0) return true;

      // extract a set of the persons attributes
      const personAttributes = getKeys(person.attributes);

      // if a person has all attributes then accept
      if (personAttributes.size === this.totalAttributes) {
        return true;
      }

      // this is the least common item so we want to grab these too
      const admitVinylCollector =
        this.data.vinyl_collector > 0 && person.attributes.vinyl_collector;

      // admit vinyl collectors
      if (
        this.totalEntries < 100 &&
        admitVinylCollector &&
        personAttributes.size > 2
      ) {
        return true;
      }

      if (this.minPeopleToMeetQuota < 1_000 - this.totalEntries) {
        // iterate over 100, 200, 300, 400, 500 people looking for largest
        // attributes starting with 1x, 2x, 3x, 4x, 5x
        for (let i = 1; i < this.totalAttributes; i++) {
          if (this.totalEntries > i * 100) continue;
          const wantedAttrs = sortedAttributes.slice(0, i);
          const hasBoth = wantedAttrs.every(([key]) =>
            personAttributes.has(key)
          );
          if (hasBoth) return true;
        }
      }

      // otherwise we want to return if they have them all in common.
      return sortedAttributes.every(([key]) => personAttributes.has(key));
    };

    if (determineToLetPersonInOrSomething()) {
      this.count(person);
      return true;
    } else {
      return false;
    }
  }

  public metrics(status: GameState["status"]) {
    if (status.status !== "running") throw status;
    return {
      data: this.data,
      score: status.rejectedCount,
      total: status.admittedCount,
    };
  }
}

type PersonAttributesScenario1 = {
  well_dressed: true;
  young: true;
};

type PersonAttributesScenario2 = {
  techno_lover: boolean;
  well_connected: boolean;
  creative: boolean;
  berlin_local: boolean;
};

type PersonAttributesScenario3 = {
  underground_veteran: boolean;
  international: boolean;
  fashion_forward: boolean;
  queer_friendly: boolean;
  vinyl_collector: boolean;
  german_speaker: boolean;
};

export type Person<T = PersonAttributesScenario3> = {
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
    score: number;
    total: number;
    data: Record<string, number>;
  };
};

const sleep = (time: number) =>
  new Promise<void>((resolve) => {
    setTimeout(() => resolve(), time);
  });

/**
 * # Game Loop
 *
 *
 */
async function runGameLoop(
  state: GameState,
  counter: GameCounter
): Promise<boolean> {
  if (state.status.status !== "running") throw new Error("Invalid status!");

  const accept = counter.shouldLetIn(state.status.nextPerson);
  const index = state.status.nextPerson?.personIndex ?? 0;

  const next = await acceptOrRejectThenGetNext({
    game: state.game,
    index: index,
    accept,
  });

  if (next.status === "completed") {
    console.log("================ success ================");
    console.log(next);
    return true;
  }

  const nextState: GameState = {
    ...state,
    status: next,
    metrics: counter.metrics(next),
  };

  prettyPrint(nextState);
  await saveGameFile(nextState);

  if (next.status === "failed") {
    console.log("================ failure ================");
    await Bun.file(state.file).delete();
    throw next;
  }

  await saveGameFile(nextState);
  return await runGameLoop(nextState, counter);
}

// ================= GAME START ===================== //

async function triggerNewGame() {
  console.log("================ starting ================");
  console.warn("[game] triggering new game!");
  const file = await initialize({ scenario: "3" });
  console.warn("[game] game file:", file);
  const savedGame = await loadGameFile({ file });
  const counter = new GameCounter(savedGame);
  await runGameLoop(savedGame, counter);
  console.log("[game] scenario 3 finished!");
  console.log("================ end ================");
}

await triggerNewGame();
