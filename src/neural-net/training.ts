/** @file training.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, Person } from '../types'
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
  explorationDecay: number // per-epoch decay
  successThreshold: number
  elitePercentile: number
}

export class SelfPlayTrainer {
  private net: NeuralNet
  private game: Game
  private encoder: StateEncoder
  private config: TrainingConfig

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
    this.net = createBerghainNet(this.encoder.getFeatureSize())
    console.log('[Net] featureSize =', this.encoder.getFeatureSize())

    this.config = {
      episodes: 100,
      batchSize: 32,
      learningRate: 0.001,
      explorationStart: 0.9,
      explorationEnd: 0.2,
      explorationDecay: 0.97, // per-epoch
      successThreshold: 5000,
      elitePercentile: 0.2,
      ...config,
    }

    this.net.setLearningRate(this.config.learningRate)
  }

  private generatePerson(index: number): Person<PersonAttributesScenario2> {
    const attributes: PersonAttributesScenario2 = {} as any
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    // sample independent
    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < freq
    }
    // apply coarse correlations
    for (const [a1, correlations] of Object.entries(stats.correlations)) {
      if (samples[a1]) {
        for (const [a2, corr] of Object.entries(correlations)) {
          if (a1 !== a2 && Math.abs(corr) > 0.3) {
            const p = Math.min(0.999, Math.max(0.001, stats.relativeFrequencies[a2] * (1 + corr * 0.5)))
            samples[a2] = Math.random() < p
          }
        }
      }
    }
    // to typed map
    for (const k of Object.keys(stats.relativeFrequencies)) {
      attributes[k as keyof PersonAttributesScenario2] = samples[k]
    }
    return { personIndex: index, attributes }
  }

  private runEpisode(explorationRate: number): Episode {
    // bouncer has NO per-step decay; explorationRate is fixed for the episode
    const bouncer = new NeuralNetBouncer(this.game, { explorationRate, baseThreshold: 0.35, minThreshold: 0.25 })
    bouncer.setNetwork(this.net)

    const states: number[][] = []
    const actions: boolean[] = []

    let admitted = 0
    let rejected = 0

    // true running counts used for encoding (to match bouncer’s tracker)
    const trueCounts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (trueCounts[k] = 0))

    while (admitted < 1000 && rejected < 20000) {
      const person = this.generatePerson(admitted + rejected)

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: person,
      }

      // Encode with TRUE counts (not estimates)
      const state = this.encoder.encode(status, trueCounts)
      states.push(state)

      const admit = bouncer.admit(status)
      actions.push(admit)

      if (admit) {
        admitted++
        // update trueCounts
        for (const [attr, has] of Object.entries(person.attributes)) {
          if (has) trueCounts[attr] = (trueCounts[attr] || 0) + 1
        }
      } else {
        rejected++
      }

      // capacity reached
      if (admitted === 1000) {
        const progress = bouncer.getProgress()
        const satisfied = progress.constraints.every((c: any) => c.satisfied)
        if (satisfied) {
          return { states, actions, reward: -rejected, rejections: rejected, completed: true }
        } else {
          const shortfall = progress.constraints.reduce(
            (s: number, c: any) => s + Math.max(0, c.required - c.current),
            0
          )
          const lambda = 10
          return { states, actions, reward: -(rejected + lambda * shortfall), rejections: rejected, completed: false }
        }
      }
    }

    // too many rejections
    const progress = bouncer.getProgress()
    const shortfall = progress.constraints.reduce((s: number, c: any) => s + Math.max(0, c.required - c.current), 0)
    const lambda = 10
    return { states, actions, reward: -(rejected + lambda * shortfall), rejections: rejected, completed: false }
  }

  // Train on top-performing (elite) episodes with label balancing & shuffling
  private trainOnEpisodes(episodes: Episode[]): number {
    if (episodes.length === 0) return 0

    const sorted = [...episodes].sort((a, b) => b.reward - a.reward)
    if (sorted[0].reward === sorted[sorted.length - 1].reward) return 0

    const eliteCount = Math.max(1, Math.floor(episodes.length * this.config.elitePercentile))
    const elite = sorted.slice(0, eliteCount)

    const rewards = elite.map((e) => e.reward)
    const rMin = Math.min(...rewards)
    const rMax = Math.max(...rewards)
    const scale = rMax > rMin ? (r: number) => 0.25 + 0.75 * ((r - rMin) / (rMax - rMin)) : (_: number) => 1

    const X: number[][] = []
    const y: number[] = []
    for (const ep of elite) {
      const w = Math.max(1, Math.round(scale(ep.reward) * 1))
      for (let rep = 0; rep < w; rep++) {
        for (let i = 0; i < ep.states.length; i++) {
          X.push(ep.states[i])
          y.push(ep.actions[i] ? 1 : 0)
        }
      }
    }
    if (X.length === 0) return 0

    // minimally upsample positives to 20% to avoid trivial rejecter
    const POS_MIN_RATIO = 0.2
    const posIdx: number[] = []
    for (let i = 0; i < y.length; i++) if (y[i] === 1) posIdx.push(i)
    const desiredPos = Math.ceil(POS_MIN_RATIO * y.length)
    if (posIdx.length > 0 && posIdx.length < desiredPos) {
      const need = desiredPos - posIdx.length
      for (let k = 0; k < need; k++) {
        const j = posIdx[k % posIdx.length]
        X.push(X[j].slice())
        y.push(1)
      }
    }

    // shuffle
    for (let i = X.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[X[i], X[j]] = [X[j], X[i]]
      ;[y[i], y[j]] = [y[j], y[i]]
    }

    // batched training
    const batchSize = Math.max(1, this.config.batchSize)
    const batchCount = Math.ceil(X.length / batchSize)
    let totalLoss = 0

    for (let b = 0; b < batchCount; b++) {
      const start = b * batchSize
      const end = Math.min(start + batchSize, X.length)
      const loss = this.net.trainBatch(X.slice(start, end), y.slice(start, end), 1)
      totalLoss += loss
    }

    return totalLoss / Math.max(1, batchCount)
  }

  async train(epochs: number = 10): Promise<void> {
    console.log('Starting self-play training...')

    let exploration = this.config.explorationStart

    for (let epoch = 0; epoch < epochs; epoch++) {
      const episodeBatch: Episode[] = []
      let successCount = 0
      let totalRejections = 0

      for (let ep = 0; ep < this.config.episodes; ep++) {
        const episode = this.runEpisode(exploration)
        episodeBatch.push(episode)

        if (episode.completed) {
          successCount++
          totalRejections += episode.rejections
          if (!this.bestEpisode || episode.rejections < this.bestEpisode.rejections) this.bestEpisode = episode
        }
      }

      const avgRejections = successCount > 0 ? totalRejections / successCount : 20000
      const successRate = successCount / this.config.episodes

      const loss = this.trainOnEpisodes(episodeBatch)

      // per-epoch decay ONLY
      exploration = Math.max(this.config.explorationEnd, exploration * this.config.explorationDecay)

      console.log(`Epoch ${epoch + 1}/${epochs}:`)
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`)
      console.log(`  Avg rejections (successful): ${avgRejections.toFixed(0)}`)
      console.log(`  Best rejections: ${this.bestEpisode?.rejections || 'N/A'}`)
      console.log(`  Training loss: ${loss.toFixed(4)}`)
      console.log(`  Exploration rate: ${exploration.toFixed(3)}`)

      this.trainingStats.push({
        epoch: epoch + 1,
        avgReward: -avgRejections,
        avgRejections,
        successRate,
        bestRejections: this.bestEpisode?.rejections || 20000,
      })

      if (successRate > 0.9 && avgRejections < 1000) {
        console.log('Early stopping - excellent performance achieved!')
        break
      }
    }

    console.log('\nTraining complete!')
    if (this.bestEpisode) console.log(`Best episode: ${this.bestEpisode.rejections} rejections`)
  }

  getNetwork(): NeuralNet {
    return this.net
  }

  getStats(): typeof this.trainingStats {
    return this.trainingStats
  }

  getBestWeights(): any {
    return this.net.toJSON()
  }

  test(episodes: number = 100): {
    successRate: number
    avgRejections: number
    minRejections: number
    maxRejections: number
  } {
    const bouncer = new NeuralNetBouncer(this.game, { explorationRate: 0, baseThreshold: 0.5 })
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
      maxRejections: successes > 0 ? maxRejections : 20000,
    }
  }
}

export async function trainBouncer(game: Game): Promise<NeuralNetBouncer> {
  const trainer = new SelfPlayTrainer(game, {
    episodes: 50,
    batchSize: 32,
    learningRate: 0.001,
    explorationStart: 0.9,
    explorationEnd: 0.2,
    explorationDecay: 0.97, // per-epoch
  })

  await trainer.train(20)

  const results = trainer.test(100)
  console.log('\nTest Results:')
  console.log(`Success rate: ${(results.successRate * 100).toFixed(1)}%`)
  console.log(`Average rejections: ${results.avgRejections.toFixed(0)}`)
  console.log(`Best: ${results.minRejections}, Worst: ${results.maxRejections}`)

  const bouncer = new NeuralNetBouncer(game)
  bouncer.setNetwork(trainer.getNetwork())
  return bouncer
}
