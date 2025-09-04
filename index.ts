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


function toBitmask(attr: PersonAttributesScenario2) {
  const OxTecho = attr.techno_lover ? 0b1000 : 0b0000
  const OxConnt = attr.well_connected ? 0b0100 : 0b0000
  const OxCreat = attr.creative ? 0b0010 : 0b0000
  const OxLocal = attr.berlin_local ? 0b0001 : 0b0000
  return OxTecho | OxConnt | OxCreat | OxLocal
}


function doesSetBHaveAllSetA<T>(setA: Set<T>, setB: Set<T>): boolean {
  for (const key of setB.values()) {
    if (setA.has(key)) continue
    return false
  }
  return true
}

function countMatchingBitsEfficient(mask1: number, mask2: number): number {
  let matching = mask1 & mask2;
  let count = 0;
  while (matching) {
      count += matching & 1;
      matching >>= 1;
  }
  return count;
}

const getKeys = (attr: Person['attributes']) => {
  return new Set(Object.entries(attr).filter(([key, attr]) => attr === true).map(([key]) => key) as (keyof Person['attributes'])[])
}

class GameCounter {
  public data: Record<keyof Person['attributes'], number> = {
    berlin_local: 0,
    techno_lover: 0,
    creative: 0,
    well_connected: 0,
  }

  private canLetAnyoneIn = false;
  private totalEntries = 0;

  get bitmask() {
    return toBitmask({
      berlin_local: this.data.berlin_local > 0,
      techno_lover: this.data.techno_lover > 0,
      creative: this.data.creative > 0,
      well_connected: this.data.well_connected > 0,
    })
  }

  constructor(initialState: GameState) {
    // initialize with game constraints
    for (const constraint of initialState.game.constraints) {
      this.data[constraint.attribute as keyof Person['attributes']] = Number(constraint.minCount);
    }
  }

  public count(person: Person) {
    if (this.canLetAnyoneIn) return true;
    for (const [key, value] of Object.entries(person.attributes) as [keyof Person['attributes'], boolean][]) {
      if (value === false) continue;
      this.data[key]--;
    }

    this.canLetAnyoneIn = Object.values(this.data).every((val) => val < 0);
  }

  get availableSpaces(): number {
    return 1_000 - this.totalEntries
  }

  get totalPeopleNeeded(): number {
    return Object.values(this.data).reduce((total, current) => {
      return total + (current < 0 ? 0 : current)
    }, 0)
  }

  public shouldLetIn(person: Person): boolean {
    const YES = () => {
      this.count(person)
      return true
    }
    const NO = () => { return false }
    const personKeys = getKeys(person.attributes)

    // person is a unicorn, let them in...
    if (personKeys.size === 4) {
      return YES()
    }

    // check if we need to find the exact people
    const hasToFindExactPeople = this.totalEntries > 900   

    // handle limiting factor which is this key
    if (this.data.creative > 50 && person.attributes.creative) return YES()

    if (this.totalEntries < 200 && personKeys.size >= 3) return YES()


    // determine which keys are left
    const wantedKeys = getKeys({
      berlin_local: this.data.berlin_local > 0,
      techno_lover: this.data.techno_lover > 0,
      creative: this.data.creative > 0,
      well_connected: this.data.well_connected > 0,
    })

    // attempt to knock down large items which can take out a lot
    if (!hasToFindExactPeople && personKeys.size + 1 >= wantedKeys.size) return YES()

    // we can just return everyone now
    if (wantedKeys.size === 0) return YES()

    // check how many keys both share in common

    for (const key of wantedKeys) {
      if (personKeys.has(key)) continue
      return NO()
    }

    return YES()
  }

  public metrics(status: GameState["status"]) {
    if (status.status !== "running")
      throw new Error("Invalid status for metrics:");
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

export type Person<T = PersonAttributesScenario2> = {
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

  if (next.status === 'completed') {
    console.log('================ success ================')
    console.log(next)
    return true
  }

  const nextState: GameState = {
    ...state,
    status: next,
    metrics: counter.metrics(next),
  };

  prettyPrint(nextState);
  await saveGameFile(nextState);

  if (next.status === "failed") {
    console.log('================ failure ================')
    await Bun.file(state.file).delete();
    throw next;
  }

  await saveGameFile(nextState);
  return await runGameLoop(nextState, counter);
}

// ================= GAME START ===================== //

async function triggerNewGame() {
  console.warn("[game] triggering new game!");
  const file = await initialize({ scenario: "2" });
  const savedGame = await loadGameFile({ file });
  const counter = new GameCounter(savedGame);
  await runGameLoop(savedGame, counter);
}

await triggerNewGame();
