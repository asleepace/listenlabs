/** @file training.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, Person } from '../types'
import { NeuralNet, createBerghainNet } from './neural-net'
import { NeuralNetBouncer } from './neural-net-bouncer'
import { StateEncoder } from './state-encoder'

interface Episode {
  states: number[][]
  actions: boolean[]
  people: Person<PersonAttributesScenario2>[]
  reward: number
  rejections: number
  completed: boolean
  admittedAtEnd: number
  nudgeCount?: number
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
  teacherAssistProb?: number
}

export class SelfPlayTrainer {
  private net: NeuralNet
  private game: Game
  private encoder: StateEncoder
  private config: TrainingConfig

  private bestEpisode: Episode | null = null
  private trainingStats: {
    epoch: number
    avgReward: number
    avgRejections: number
    successRate: number
    bestRejections: number
  }[] = []

  private currentAssistProb = 0

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
      explorationDecay: 0.97,
      successThreshold: 5000,
      elitePercentile: 0.2,
      teacherAssistProb: 0.02,
      ...config,
    }

    this.net.setLearningRate(this.config.learningRate)
  }

  // synthetic person
  private generatePerson(index: number): Person<PersonAttributesScenario2> {
    const attributes: PersonAttributesScenario2 = {} as any
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < freq
    }
    for (const [a1, correlations] of Object.entries(stats.correlations)) {
      if (samples[a1]) {
        for (const [a2, corr] of Object.entries(correlations)) {
          if (a1 !== a2 && Math.abs(corr) > 0.3) {
            const p = stats.relativeFrequencies[a2] * (1 + corr * 0.5)
            samples[a2] = Math.random() < p
          }
        }
      }
    }
    for (const k of Object.keys(stats.relativeFrequencies)) {
      attributes[k as keyof PersonAttributesScenario2] = !!samples[k]
    }
    return { personIndex: index, attributes }
  }

  // --- tiny oracle used only as a nudge during training ---
  // REPLACE your greedyNudgeShouldAdmit(...) with this version.
  private greedyNudgeShouldAdmit(
    counts: Record<string, number>,
    personAttrs: Record<string, boolean>,
    admittedSoFar: number
  ): boolean {
    const R = Math.max(1, 1000 - admittedSoFar) // remaining seats including this decision
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const f = this.game.attributeStatistics.relativeFrequencies[c.attribute] || 0
      // expected final if we REJECT this person now:
      const expectedIfReject = cur + f * (R - 1)
      const criticalNow = expectedIfReject < c.minCount
      if (criticalNow && personAttrs[c.attribute]) return true
    }
    return false
  }

  // one episode — side-effect-free decision first, then apply once to bouncer
  private runEpisode(explorationRate: number, useTeacherAssist = true): Episode {
    const bouncer = new NeuralNetBouncer(this.game, {
      explorationRate,
      baseThreshold: 0.45,
      minThreshold: 0.35,
      maxThreshold: 0.75,
      urgencyFactor: 2.0,
    })
    bouncer.setNetwork(this.net)

    const states: number[][] = []
    const actions: boolean[] = []
    const people: Person<PersonAttributesScenario2>[] = []

    let admitted = 0
    let rejected = 0
    let nudgeCount = 0

    const trueCounts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (trueCounts[k] = 0))

    while (admitted < 1000 && rejected < 20000) {
      const person = this.generatePerson(admitted + rejected)
      people.push(person)

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: person,
      }

      const features = this.encoder.encode(status, trueCounts)
      states.push(features)

      // side-effect free score
      const prob = bouncer.predictProbability(features)
      const threshold = bouncer.computeThreshold(status)

      // exploration
      let admit = Math.random() < explorationRate ? bouncer.exploratoryAdmit(status) : prob > threshold

      // teacher assist (decayed over epochs)
      if (!admit && useTeacherAssist && Math.random() < this.currentAssistProb) {
        if (this.greedyNudgeShouldAdmit(trueCounts, person.attributes, admitted)) {
          admit = true
          nudgeCount++
        }
      }

      // apply final decision exactly once to bouncer internals
      bouncer.applyFinalDecision(admit, person.attributes)
      actions.push(admit)

      // mirror to our episode-level counts
      if (admit) {
        admitted++
        for (const [attr, has] of Object.entries(person.attributes)) {
          if (has) trueCounts[attr] = (trueCounts[attr] || 0) + 1
        }
      } else {
        rejected++
      }

      if (admitted === 1000) {
        const progress = bouncer.getProgress()
        const satisfied = progress.constraints.every((c: any) => c.satisfied)

        const shortfall = progress.constraints.reduce((s: number, c: any) => s + Math.max(0, c.required - c.current), 0)
        if (satisfied) {
          console.log(`[episode end] admitted=${admitted}, rej=${rejected}, shortfall=${shortfall}, success=true`)
          return {
            states,
            actions,
            people,
            reward: -rejected,
            rejections: rejected,
            completed: true,
            admittedAtEnd: admitted,
            nudgeCount,
          }
        } else {
          const lambda = 30
          const reward = -(rejected + lambda * shortfall)
          console.log(`[episode end] admitted=${admitted}, rej=${rejected}, shortfall=${shortfall}, reward=${reward}`)
          return {
            states,
            actions,
            people,
            reward,
            rejections: rejected,
            completed: false,
            admittedAtEnd: admitted,
            nudgeCount,
          }
        }
      }
    }

    // failed by rejections
    const progress = bouncer.getProgress()
    const shortfall = progress.constraints.reduce((s: number, c: any) => s + Math.max(0, c.required - c.current), 0)
    const lambda = 30
    const reward = -(rejected + lambda * shortfall)
    console.log(
      `[episode end] admitted=${admitted}, rej=${rejected}, shortfall=${shortfall}, reward=${reward} (failed by rejections)`
    )
    return {
      states,
      actions,
      people,
      reward,
      rejections: rejected,
      completed: false,
      admittedAtEnd: admitted,
      nudgeCount,
    }
  }

  // oracle labels: urgency at each step
  // REPLACE your buildOracleLabels(...) with this version.
  private buildOracleLabels(ep: Episode): number[] {
    const counts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (counts[k] = 0))

    const labels: number[] = []
    let admittedSoFar = 0

    for (let i = 0; i < ep.people.length; i++) {
      const person = ep.people[i]
      const R = Math.max(1, 1000 - admittedSoFar)

      let label = 0
      for (const c of this.game.constraints) {
        const cur = counts[c.attribute] || 0
        const f = this.game.attributeStatistics.relativeFrequencies[c.attribute] || 0
        const expectedIfReject = cur + f * (R - 1)
        const criticalNow = expectedIfReject < c.minCount
        if (criticalNow && person.attributes[c.attribute]) {
          label = 1
          break
        }
      }
      labels.push(label)

      // advance counts using the ACTUAL action taken in this episode
      if (ep.actions[i]) {
        admittedSoFar++
        for (const [attr, has] of Object.entries(person.attributes)) {
          if (has) counts[attr] = (counts[attr] || 0) + 1
        }
        if (admittedSoFar >= 1000) break
      }
    }

    return labels
  }

  // train on elites with oracle labels and ~60/40 balance
  private trainOnEpisodes(episodes: Episode[]): number {
    if (episodes.length === 0) return 0
    const sorted = [...episodes].sort((a, b) => b.reward - a.reward)
    if (sorted[0].reward === sorted[sorted.length - 1].reward) return 0

    const eliteCount = Math.max(1, Math.floor(episodes.length * this.config.elitePercentile))
    const elite = sorted.slice(0, eliteCount)

    const X: number[][] = []
    const y: number[] = []
    for (const ep of elite) {
      const labels = this.buildOracleLabels(ep)
      const n = Math.min(labels.length, ep.states.length)
      for (let i = 0; i < n; i++) {
        X.push(ep.states[i])
        y.push(labels[i])
      }
    }
    if (X.length === 0) return 0

    // ~60/40 balance via downsampling larger class
    const TARGET_POS = 0.6
    const posIdx: number[] = []
    const negIdx: number[] = []
    for (let i = 0; i < y.length; i++) (y[i] === 1 ? posIdx : negIdx).push(i)
    if (!posIdx.length || !negIdx.length) {
      console.log('[train] skipped: oracle produced single-class dataset')
      return 0
    }

    const needTotal = Math.min(Math.floor(posIdx.length / TARGET_POS), Math.floor(negIdx.length / (1 - TARGET_POS)))
    const wantTotal = Math.max(2, needTotal)
    const wantPos = Math.floor(TARGET_POS * wantTotal)
    const wantNeg = wantTotal - wantPos

    function pick<T>(arr: T[], k: number): T[] {
      const a = arr.slice()
      for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[a[i], a[j]] = [a[j], a[i]]
      }
      return a.slice(0, Math.max(0, Math.min(k, a.length)))
    }

    const keepPos = pick(posIdx, wantPos)
    const keepNeg = pick(negIdx, wantNeg)
    const keep = keepPos.concat(keepNeg)

    const Xb: number[][] = []
    const yb: number[] = []
    for (const i of keep) {
      Xb.push(X[i])
      yb.push(y[i])
    }

    for (let i = Xb.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[Xb[i], Xb[j]] = [Xb[j], Xb[i]]
      ;[yb[i], yb[j]] = [yb[j], yb[i]]
    }

    const batchSize = Math.max(1, this.config.batchSize)
    const batchCount = Math.ceil(Xb.length / batchSize)
    let totalLoss = 0
    for (let b = 0; b < batchCount; b++) {
      const s = b * batchSize
      const e = Math.min(s + batchSize, Xb.length)
      totalLoss += this.net.trainBatch(Xb.slice(s, e), yb.slice(s, e), 1)
    }

    const posCount = yb.filter((v) => v === 1).length
    const negCount = yb.length - posCount
    console.log(`[train] elite samples: ${Xb.length} (pos=${posCount}, neg=${negCount})`)

    return totalLoss / Math.max(1, batchCount)
  }

  async train(epochs: number = 10): Promise<void> {
    console.log('Starting self-play training...')
    let exploration = this.config.explorationStart

    for (let epoch = 0; epoch < epochs; epoch++) {
      // fast decay teacher assist
      const baseAssist = this.config.teacherAssistProb ?? 0
      this.currentAssistProb = baseAssist * Math.pow(0.9, epoch)

      const episodeBatch: Episode[] = []
      let successCount = 0
      let totalRejections = 0

      for (let ep = 0; ep < this.config.episodes; ep++) {
        const episode = this.runEpisode(exploration, true)
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

      const avgAdmittedAll = episodeBatch.reduce((s, e) => s + e.admittedAtEnd, 0) / episodeBatch.length
      const totalNudges = episodeBatch.reduce((s, e) => s + (e.nudgeCount ?? 0), 0)
      const bestEp = episodeBatch.reduce((b, e) => (e.reward > b.reward ? e : b), episodeBatch[0])
      const approxShortfallBest = Math.max(0, Math.floor((-bestEp.reward - bestEp.rejections) / 30))

      exploration = Math.max(this.config.explorationEnd, exploration * this.config.explorationDecay)

      console.log(`Epoch ${epoch + 1}/${epochs}:`)
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`)
      console.log(`  Avg rejections (successful): ${avgRejections.toFixed(0)}`)
      console.log(`  Best rejections: ${this.bestEpisode?.rejections || 'N/A'}`)
      console.log(`  Training loss: ${loss.toFixed(4)}`)
      console.log(`  Exploration rate: ${exploration.toFixed(3)}`)
      console.log(`  Avg admitted (all episodes): ${avgAdmittedAll.toFixed(1)}`)
      console.log(
        `  Best episode — admitted: ${bestEp.admittedAtEnd}, rejections: ${bestEp.rejections}, reward: ${bestEp.reward}, ~shortfall≈${approxShortfallBest}`
      )
      console.log(
        `  Teacher nudges used this epoch: ${totalNudges}, assistProb(now)=${this.currentAssistProb.toFixed(4)}`
      )

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
  getStats() {
    return this.trainingStats
  }
  getBestWeights(): any {
    return this.net.toJSON()
  }

  test(episodes: number = 100) {
    let successes = 0,
      totalRejections = 0,
      minRej = Infinity,
      maxRej = 0
    for (let i = 0; i < episodes; i++) {
      const episode = this.runEpisode(0, false) // no exploration, no teacher
      if (episode.completed) {
        successes++
        totalRejections += episode.rejections
        minRej = Math.min(minRej, episode.rejections)
        maxRej = Math.max(maxRej, episode.rejections)
      }
    }
    return {
      successRate: successes / episodes,
      avgRejections: successes > 0 ? totalRejections / successes : 20000,
      minRejections: minRej === Infinity ? 20000 : minRej,
      maxRejections: successes > 0 ? maxRej : 20000,
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
    explorationDecay: 0.97,
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
