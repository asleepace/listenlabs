import { type GameState } from '../types'

export interface Episode {
  states: GameState[]
  actions: boolean[] // admit/reject
  reward: number // -rejectedCount if successful, -20000 if failed
}

export class SelfPlayTrainer {
  generateEpisode(): Episode {
    return {
      states: [],
      actions: [],
      reward: -20000,
    }
  }
  updateWeights(episodes: Episode[]) {}
}
