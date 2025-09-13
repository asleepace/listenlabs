import type {
  Game,
  GameStatusRunning,
  GameStatusCompleted,
  GameStatusFailed,
  PersonAttributesScenario2,
  Person,
} from '../types'

import { NeuralNet, createBerghainNet } from './neural-net'
import { NeuralNetBouncer } from './neural-net-bouncer'
import { StateEncoder } from './state-encoder'

interface Episode {
  states: number[][]
  actions: boolean[]
  reward: number
  rejections: number
  completed: boolean
}

interface TrainingConfig {
  episodes: number
  batchSize: number
  learningRate: number
  explorationStart: number
  explorationEnd: number
  explorationDecay: number
  successThreshold: number
  elitePercentile: number
}

export class SelfPlayTrainer {
  private net: NeuralNet
  private game: Game
  private encoder: StateEncoder
  private config: TrainingConfig

  // Training history
  private episodeHistory: Episode[] = []
  private bestEpisode: Episode | null = null
  private trainingStats: {
    epoch: number
    avgReward: number
    avgRejections: number
    successRate: number
    bestRejections: number
  }[] = []

  constructor(game: Game, config?: Partial<TrainingConfig>) {
    this.game = game
    this.encoder = new StateEncoder(game)
    this.net = createBerghainNet()

    this.config = {
      episodes: 100,
      batchSize: 32,
      learningRate: 0.001,
      explorationStart: 0.3,
      explorationEnd: 0.05,
      explorationDecay: 0.995,
      successThreshold: 5000, // Max rejections for "success"
      elitePercentile: 0.2, // Top 20% of episodes
      ...config,
    }

    this.net.setLearningRate(this.config.learningRate)
  }

  // Generate synthetic person based on statistics
  private generatePerson(index: number): Person<PersonAttributesScenario2> {
    const attributes: PersonAttributesScenario2 = {} as any
    const stats = this.game.attributeStatistics

    // First, sample each attribute independently based on frequency
    const samples: Record<string, boolean> = {}
    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < freq
    }

    // Apply correlations to make it more realistic
    // This is a simplified correlation application
    for (const [attr1, correlations] of Object.entries(stats.correlations)) {
      if (samples[attr1]) {
        for (const [attr2, corr] of Object.entries(correlations)) {
          if (attr1 !== attr2 && Math.abs(corr) > 0.3) {
            // Strong correlation - adjust probability
            const adjustedProb = stats.relativeFrequencies[attr2] * (1 + corr * 0.5)
            samples[attr2] = Math.random() < adjustedProb
          }
        }
      }
    }

    // Convert to proper format
    for (const key of Object.keys(stats.relativeFrequencies)) {
      attributes[key as keyof PersonAttributesScenario2] = samples[key]
    }

    return {
      personIndex: index,
      attributes,
    }
  }

  // Simulate one episode
  private runEpisode(explorationRate: number): Episode {
    const bouncer = new NeuralNetBouncer(this.game, {
      explorationRate,
      baseThreshold: 0.5,
    })
    bouncer.setNetwork(this.net)

    const states: number[][] = []
    const actions: boolean[] = []
    let admitted = 0
    let rejected = 0

    // Run until completion or failure
    while (admitted < 1000 && rejected < 20000) {
      const person = this.generatePerson(admitted + rejected)

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: person,
      }

      // Encode state
      const state = this.encoder.encode(status)
      states.push(state)

      // Make decision
      const admit = bouncer.admit(status)
      actions.push(admit)

      // Update counts
      if (admit) {
        admitted++
      } else {
        rejected++
      }

      // Check constraints if venue is full
      if (admitted === 1000) {
        const progress = bouncer.getProgress()
        const satisfied = progress.constraints.every((c: any) => c.satisfied)

        if (satisfied) {
          // Success!
          return {
            states,
            actions,
            reward: -rejected, // Negative rejections as reward
            rejections: rejected,
            completed: true,
          }
        } else {
          // Failed constraints
          return {
            states,
            actions,
            reward: -20000,
            rejections: rejected,
            completed: false,
          }
        }
      }
    }

    // Failed - too many rejections
    return {
      states,
      actions,
      reward: -20000,
      rejections: rejected,
      completed: false,
    }
  }

  // Train on successful episodes
  private trainOnEpisodes(episodes: Episode[]): number {
    if (episodes.length === 0) return 0

    // Sort by reward (higher is better)
    const sorted = [...episodes].sort((a, b) => b.reward - a.reward)

    // Take elite episodes
    const eliteCount = Math.max(1, Math.floor(episodes.length * this.config.elitePercentile))
    const elite = sorted.slice(0, eliteCount)

    // Prepare training data
    const inputs: number[][] = []
    const targets: number[] = []

    for (const episode of elite) {
      for (let i = 0; i < episode.states.length; i++) {
        inputs.push(episode.states[i])
        targets.push(episode.actions[i] ? 1 : 0)
      }
    }

    // Train in batches
    let totalLoss = 0
    const batchCount = Math.ceil(inputs.length / this.config.batchSize)

    for (let i = 0; i < batchCount; i++) {
      const start = i * this.config.batchSize
      const end = Math.min(start + this.config.batchSize, inputs.length)

      const batchInputs = inputs.slice(start, end)
      const batchTargets = targets.slice(start, end)

      const loss = this.net.trainBatch(batchInputs, batchTargets, 1)
      totalLoss += loss
    }

    return totalLoss / batchCount
  }

  // Main training loop
  async train(epochs: number = 10): Promise<void> {
    console.log('Starting self-play training...')

    let exploration = this.config.explorationStart

    for (let epoch = 0; epoch < epochs; epoch++) {
      const episodeBatch: Episode[] = []
      let successCount = 0
      let totalRejections = 0

      // Run episodes
      for (let ep = 0; ep < this.config.episodes; ep++) {
        const episode = this.runEpisode(exploration)
        episodeBatch.push(episode)

        if (episode.completed) {
          successCount++
          totalRejections += episode.rejections

          // Track best episode
          if (!this.bestEpisode || episode.rejections < this.bestEpisode.rejections) {
            this.bestEpisode = episode
          }
        }
      }

      // Calculate stats
      const avgRejections = successCount > 0 ? totalRejections / successCount : 20000
      const successRate = successCount / this.config.episodes

      // Train on episodes
      const loss = this.trainOnEpisodes(episodeBatch)

      // Update exploration
      exploration = Math.max(this.config.explorationEnd, exploration * this.config.explorationDecay)

      // Log progress
      console.log(`Epoch ${epoch + 1}/${epochs}:`)
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`)
      console.log(`  Avg rejections (successful): ${avgRejections.toFixed(0)}`)
      console.log(`  Best rejections: ${this.bestEpisode?.rejections || 'N/A'}`)
      console.log(`  Training loss: ${loss.toFixed(4)}`)
      console.log(`  Exploration rate: ${exploration.toFixed(3)}`)

      // Store stats
      this.trainingStats.push({
        epoch: epoch + 1,
        avgReward: -avgRejections,
        avgRejections,
        successRate,
        bestRejections: this.bestEpisode?.rejections || 20000,
      })

      // Early stopping if we're doing well
      if (successRate > 0.9 && avgRejections < 1000) {
        console.log('Early stopping - excellent performance achieved!')
        break
      }
    }

    console.log('\nTraining complete!')
    if (this.bestEpisode) {
      console.log(`Best episode: ${this.bestEpisode.rejections} rejections`)
    }
  }

  // Get trained network
  getNetwork(): NeuralNet {
    return this.net
  }

  // Get training statistics
  getStats(): typeof this.trainingStats {
    return this.trainingStats
  }

  // Save best weights
  getBestWeights(): any {
    return this.net.toJSON()
  }

  // Test the trained network
  test(episodes: number = 100): {
    successRate: number
    avgRejections: number
    minRejections: number
    maxRejections: number
  } {
    const bouncer = new NeuralNetBouncer(this.game, {
      explorationRate: 0, // No exploration during testing
      baseThreshold: 0.5,
    })
    bouncer.setNetwork(this.net)

    let successes = 0
    let totalRejections = 0
    let minRejections = Infinity
    let maxRejections = 0

    for (let i = 0; i < episodes; i++) {
      const episode = this.runEpisode(0)

      if (episode.completed) {
        successes++
        totalRejections += episode.rejections
        minRejections = Math.min(minRejections, episode.rejections)
        maxRejections = Math.max(maxRejections, episode.rejections)
      }
    }

    return {
      successRate: successes / episodes,
      avgRejections: successes > 0 ? totalRejections / successes : 20000,
      minRejections: minRejections === Infinity ? 20000 : minRejections,
      maxRejections,
    }
  }
}

// Example usage
export async function trainBouncer(game: Game): Promise<NeuralNetBouncer> {
  const trainer = new SelfPlayTrainer(game, {
    episodes: 50,
    batchSize: 32,
    learningRate: 0.001,
    explorationStart: 0.3,
    explorationEnd: 0.05,
    explorationDecay: 0.99,
  })

  // Train for 20 epochs
  await trainer.train(20)

  // Test performance
  const results = trainer.test(100)
  console.log('\nTest Results:')
  console.log(`Success rate: ${(results.successRate * 100).toFixed(1)}%`)
  console.log(`Average rejections: ${results.avgRejections.toFixed(0)}`)
  console.log(`Best: ${results.minRejections}, Worst: ${results.maxRejections}`)

  // Create bouncer with trained weights
  const bouncer = new NeuralNetBouncer(game)
  bouncer.setNetwork(trainer.getNetwork())

  return bouncer
}
