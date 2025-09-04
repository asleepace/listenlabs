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

  constructor(initialState: GameState) {
    // initialize with game constraints
    for (const constraint of initialState.game.constraints) {
      this.data[constraint.attribute as keyof Person["attributes"]] = Number(
        constraint.minCount
      );
    }
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

  get availableSpaces(): number {
    return 1_000 - this.totalEntries;
  }

  get totalPeopleNeeded(): number {
    return Object.values(this.data).reduce((total, current) => {
      return total + (current < 0 ? 0 : current);
    }, 0);
  }

  public shouldLetIn(person: Person): boolean {
    const YES = () => {
      this.count(person);
      return true;
    };
    const NO = () => {
      return false;
    };
    const personKeys = getKeys(person.attributes);

    // person is a unicorn, let them in...
    if (personKeys.size === 6) {
      return YES();
    }

    // check if we need to find the exact people
    const isUnderStrictLimit = this.availableSpaces >= this.totalEntries + 50;

    const shouldPickCollector =
      this.data.vinyl_collector > 0 && person.attributes.vinyl_collector;
    const shouldPickGerman =
      this.data.german_speaker > 0 && person.attributes.german_speaker;
    const shouldPickInternational =
      this.data.international > 0 && person.attributes.international;
    const shouldPickQueerFriendly =
      this.data.queer_friendly > 0 && person.attributes.queer_friendly;

    if (isUnderStrictLimit && shouldPickGerman && shouldPickCollector) {
      return YES();
    }

    // pick internationl queer pairs
    if (
      isUnderStrictLimit &&
      shouldPickInternational &&
      shouldPickQueerFriendly
    ) {
      return YES();
    }

    // pick international german pairs
    if (isUnderStrictLimit && shouldPickInternational && shouldPickGerman) {
      return YES();
    }

    // handle limiting factor which is this key
    // if (this.totalEntries < 50 && personKeys.size >= 3) return YES();
    if (isUnderStrictLimit) {
      if (
        this.totalEntries < 100 &&
        personKeys.size >= 3 &&
        person.attributes.german_speaker
      )
        return YES();
      if (this.totalEntries < 250 && personKeys.size >= 4) return YES();
      if (this.totalEntries < 500 && personKeys.size >= 5) return YES();

      if (
        isUnderStrictLimit &&
        this.data.vinyl_collector > 50 &&
        person.attributes.vinyl_collector
      ) {
        return YES();
      }
    }

    // determine which keys are left
    const wantedKeys = getKeys({
      underground_veteran: this.data.underground_veteran > 0,
      international: this.data.international > 0,
      fashion_forward: this.data.fashion_forward > 0,
      queer_friendly: this.data.queer_friendly > 0,
      vinyl_collector: this.data.vinyl_collector > 0,
      german_speaker: this.data.german_speaker > 0,
    });

    // attempt to knock down large items which can take out a lot
    // if (this.totalEntries < 400 && !hasToFindExactPeople && personKeys.size + 1 >= wantedKeys.size)
    //   return YES();

    // we can just return everyone now
    if (wantedKeys.size === 0) return YES();

    // check how many keys both share in common
    let hasAllKeys = true;
    wantedKeys.forEach((key) => {
      if (!personKeys.has(key)) {
        hasAllKeys = false;
      }
    });

    if (hasAllKeys) {
      return YES();
    } else {
      return NO();
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
