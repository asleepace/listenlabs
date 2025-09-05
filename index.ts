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

function getMeanWithoutZeros(numbers: number[]) {
  const total = numbers.reduce((total, current) => current + total, 0);
  return total / numbers.length;
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

  private admittedCount = 0;
  get frequencies() {
    return this.state.game.attributeStatistics.relativeFrequencies as Record<
      keyof Person["attributes"],
      number
    >;
  }

  // initialize with game constraints
  constructor(public state: GameState) {
    for (const constraint of state.game.constraints) {
      // Number(constraint.minCount);
      this.data[constraint.attribute as keyof Person["attributes"]] = 0;
    }
  }

  // count people to let enter
  public count(person: Person) {
    for (const [key, value] of Object.entries(person.attributes) as [
      keyof Person["attributes"],
      boolean
    ][]) {
      if (value === false) continue;
      this.data[key]++;
    }
  }

  get minPeopleToMeetQuota(): number {
    return Object.values(this.data).reduce((total, current) => {
      return Math.max(total, current);
    }, 0);
  }

  calculatePersonScore = (person: Person): number => {
    if (this.state.status.status !== 'running') throw new Error('not_running')

    const constraints = this.state.game.constraints;
    const frequencies = this.state.game.attributeStatistics.relativeFrequencies;
    const currentCounts = this.data;
    const admittedCount = this.state.status.admittedCount;
    const remainingSpaces = 1000 - admittedCount;
  
    const personVector: number[] = [];
    const normalizedScores: number[] = [];
  
    constraints.forEach((constraint) => {
      const attr = constraint.attribute as keyof Person["attributes"];
      
      if (!person.attributes[attr]) {
        normalizedScores.push(0);
        return;
      }
  
      const currentCount = currentCounts[attr] || 0;
      const remaining = Math.max(0, constraint.minCount - currentCount);
      
      if (remaining === 0) {
        normalizedScores.push(0);
        return;
      }
  
      // Normalize each component to 0-1 scale
      const needsRatio = Math.min(remaining / remainingSpaces, 1); // 0-1
      const rarityScore = 1 - (frequencies[attr] || 0.001); // 0-1 (rare = closer to 1)
      
      // Urgency: completion ratio inverted
      const completionRatio = currentCount / constraint.minCount;
      const urgencyScore = Math.max(0, 1 - completionRatio); // 0-1
      
      // Combine with weights, keep in reasonable range
      const score = needsRatio * 0.4 + rarityScore * 0.4 + urgencyScore * 0.2;
      normalizedScores.push(score);
    });
  
    // Sum gives max possible score of (number of attributes)
    return normalizedScores.reduce((sum, score) => sum + score, 0);
  };
  /**
   * Core game logic here...
   * @param person 
   * @returns 
   */
  public shouldLetIn(person: Person): boolean {
    if (this.state.status.status !== "running") throw new Error("Game not running");

    const score = this.calculatePersonScore(person);
    const remainingSpaces = 1000 - this.state.status.admittedCount;
    const scorePercent = score / 6.0
    const spaceRatio = remainingSpaces / 1000; // 1.0 = full capacity, 0.0 = no space
    const threshold = 0.30 + (0.99 * Math.pow(1 - spaceRatio, 2));
    const shouldAccept = scorePercent >= threshold;

    console.log({
      person: person.personIndex,
      score: Math.round(score * 100) / 100,
      scorePercent,
      threshold,
      decision: shouldAccept,
      remainingSpaces
    });

    // Only update internal state if accepting
    if (shouldAccept) {
      this.count(person);
    }

    return shouldAccept;
  }

  // Remove admittedCount tracking - let the game state handle this
  // Remove the increment from shouldLetIn method

  public metrics(status: GameState["status"]) {
    if (status.status !== "running") throw new Error("invalid status");
    this.state.status = status
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
