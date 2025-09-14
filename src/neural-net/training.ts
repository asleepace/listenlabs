/** @file training.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, Person, ScenarioAttributes } from '../types'
import { NeuralNet, createBerghainNet } from './neural-net'
import { NeuralNetBouncer } from './neural-net-bouncer'
import { StateEncoder } from './state-encoder'
import { initializeScoring } from './scoring'

interface Episode {
  admittedPrefix: number[] // admitted count before each decision
  states: number[][]
  actions: boolean[]
  reward: number
  rejections: number
  completed: boolean
  admittedAtEnd: number
  nudgeCount?: number
  countsPerStep: Array<Record<string, number>> // snapshot of true counts at each step (pre-action)
  peoplePerStep: Array<Record<string, boolean>> // person attributes at each step
}

interface TrainingConfig {
  episodes: number
  batchSize: number
  learningRate: number
  explorationStart: number // per-epoch exploration
  explorationEnd: number
  explorationDecay: number // per-epoch decay
  successThreshold: number
  elitePercentile: number
  teacherAssistProb?: number // legacy; not used directly
  assistGain?: number // k for adaptive assist
  oracleRelabelFrac?: number // fraction of elite samples to relabel with oracle [0..1]
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

// ----- Reward shaping knobs -----
const LAMBDA_SHORTFALL = 50 // linear penalty per missing head
const QUAD_SHORTFALL = 0.05 // extra penalty for concentrated gaps
const BETA_SURPLUS = 1.0 // mild penalty per head above required
const LOSS_PENALTY = 100000 // flat penalty for losing (unmet or reject cap)

/** Per-attribute weights to nudge the policy where we were systematically off. */
const SHORTFALL_WEIGHTS: Record<string, number> = {
  creative: 1.6, // push scarce 'creative'
  techno_lover: 1.2, // mild bump to avoid TL misses
  // others default to 1.0
}

const SURPLUS_WEIGHTS: Record<string, number> = {
  // creative: 1.6, // penalize 'creative' overshoot
  well_connected: 1.8,
  // others default to 1.0
}

export class SelfPlayTrainer {
  private net: NeuralNet
  private game: Game
  private encoder: StateEncoder
  private config: TrainingConfig
  private resumedFromWeights = false

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
      explorationDecay: 0.97,
      successThreshold: 5000,
      elitePercentile: 0.2,
      teacherAssistProb: 0.0,
      assistGain: 2.0,
      oracleRelabelFrac: 0.35,
      ...config,
    }

    this.net.setLearningRate(this.config.learningRate)
  }

  loadWeights(weights: any): void {
    const n: any = this.net as any
    if (typeof n.fromJSON === 'function') n.fromJSON(weights)
    else if (typeof n.loadJSON === 'function') n.loadJSON(weights)
    else if (typeof n.load === 'function') n.load(weights)
    else console.warn('[trainer] Unable to load weights: no fromJSON/loadJSON/load on NeuralNet')
    this.resumedFromWeights = true
  }

  // ---------- synthetic generator ----------
  private generatePerson(index: number): Person<PersonAttributesScenario2> {
    const attributes: PersonAttributesScenario2 = {} as any
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    // Base sampling
    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < freq
    }
    // Correlation adjustments
    for (const [a1, correlations] of Object.entries(stats.correlations)) {
      if (samples[a1]) {
        for (const [a2, corr] of Object.entries(correlations)) {
          if (a1 !== a2 && Math.abs(corr) > 0.3) {
            const p = clamp((stats.relativeFrequencies as any)[a2] * (1 + corr * 0.5), 0, 1)
            samples[a2] = Math.random() < p
          }
        }
      }
    }
    for (const k of Object.keys(stats.relativeFrequencies)) {
      ;(attributes as any)[k] = !!samples[k]
    }
    return { personIndex: index, attributes }
  }

  // ---------- expected-gap helpers ----------
  private worstExpectedGap(counts: Record<string, number>, remaining: number): { gap: number; culprit?: string } {
    let worst = 0
    let culprit: string | undefined
    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const f = this.game.attributeStatistics.relativeFrequencies[c.attribute] || 0
      const expectedFinal = cur + f * remaining
      const gap = Math.max(0, c.minCount - expectedFinal)
      if (gap > worst) {
        worst = gap
        culprit = c.attribute
      }
    }
    return { gap: worst, culprit }
  }

  // ---------- oracle policies ----------
  /** Admit iff the candidate materially reduces expected worst shortfall (or no gap). */
  private oracleShouldAdmit(
    counts: Record<string, number>,
    person: Record<string, boolean>,
    admitted: number
  ): boolean {
    const remaining = Math.max(1, 1000 - admitted)
    const { gap: before } = this.worstExpectedGap(counts, remaining)

    const afterCounts: Record<string, number> = { ...counts }
    for (const [attr, has] of Object.entries(person)) if (has) afterCounts[attr] = (afterCounts[attr] || 0) + 1
    const { gap: after } = this.worstExpectedGap(afterCounts, remaining - 1)

    if (after + 1e-9 < before) return true
    if (before === 0) return true
    return false
  }

  /** Adaptive assist probability driven by current expected-gap. */
  private assistProb(counts: Record<string, number>, admitted: number): number {
    const remaining = Math.max(1, 1000 - admitted)
    const { gap } = this.worstExpectedGap(counts, remaining)
    const g = gap / remaining // normalized [0..1+]
    return clamp(this.config.assistGain! * g, 0, 0.5)
  }

  // ---------- run one episode ----------
  /**
   * Run one self-play episode.
   * - explorationRate:  ε for the network's exploration (passed into NeuralNetBouncer)
   * - usePolicyFusion:  allow seat/urgency scoring to influence final action
   * - useTeacherAssist: allow oracle nudges to correct decisions probabilistically
   */
  private runEpisode({
    explorationRate,
    usePolicyFusion = true,
    useTeacherAssist = true,
  }: {
    explorationRate: number
    useTeacherAssist?: boolean
    usePolicyFusion?: boolean
  }): Episode {
    const bouncer = new NeuralNetBouncer(this.game, {
      explorationRate,
      baseThreshold: 0.35,
      minThreshold: 0.25,
      maxThreshold: 0.7,
      urgencyFactor: 2.0,
    })
    bouncer.setNetwork(this.net)

    const states: number[][] = []
    const actions: boolean[] = []
    const countsPerStep: Array<Record<string, number>> = []
    const peoplePerStep: Array<Record<string, boolean>> = []
    const admittedPrefix: number[] = []

    let admitted = 0
    let rejected = 0
    let nudgeCount = 0

    // ground-truth running counts for reward shaping
    const trueCounts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (trueCounts[k] = 0))

    // seat/urgency scoring engine
    const scoring = initializeScoring(this.game, {
      maxAdmissions: 1_000,
      maxRejections: 20_000,
      targetRejections: 5_500, // steer to desired reject band
      safetyCushion: 1, // +1 cushion to avoid off-by-one misses near the end
      weights: {
        // leave empty unless you want to override scoring weights here
      },
    })

    while (scoring.inProgress()) {
      // ONE person per step: same sample for scoring + NN
      const person = this.generatePerson(admitted + rejected)

      // snapshots (pre-action)
      countsPerStep.push({ ...trueCounts })
      peoplePerStep.push({ ...person.attributes })
      admittedPrefix.push(admitted)

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: person,
      }

      // encode state with current true counts
      const state = this.encoder.encode(status, trueCounts)
      states.push(state)

      // --- hybrid decision (scoring + network) ---
      const guest = person.attributes as ScenarioAttributes
      let admit = bouncer.admit(status, trueCounts)

      if (usePolicyFusion) {
        const policyVote = scoring.shouldAdmit(guest, 1.0, 0.5)
        const helpsWorstGap = this.oracleShouldAdmit(trueCounts, person.attributes, admitted)
        const scarce = scoring.isRunningOutOfAvailableSpots()

        if (!scarce) {
          // Early/mid game: permissive — let either strategy open the door
          admit = policyVote || admit
        } else {
          // Late/low-seats: must help the worst shortfall, and pass at least one gate
          admit = (policyVote || admit) && helpsWorstGap
        }
      }
      // teacher assist (oracle) during training only
      if (useTeacherAssist) {
        const pAssist = this.assistProb(trueCounts, admitted)
        if (Math.random() < pAssist) {
          const oracle = this.oracleShouldAdmit(trueCounts, person.attributes, admitted)
          if (oracle !== admit) {
            admit = oracle
            if (admit) nudgeCount++
          }
        }
      }

      actions.push(admit)

      // apply decision to both trackers
      if (admit) {
        admitted++
        for (const [attr, has] of Object.entries(person.attributes)) {
          if (has) trueCounts[attr] = (trueCounts[attr] || 0) + 1
        }
      } else {
        rejected++
      }
      scoring.update({ guest, admit })

      // filled venue → score episode
      if (admitted === 1000) {
        const { reward, completed } = this.scoreEpisode(trueCounts, rejected, /*hitRejectCap=*/ false)
        return {
          admittedPrefix,
          states,
          actions,
          reward,
          rejections: rejected,
          completed,
          admittedAtEnd: admitted,
          nudgeCount,
          countsPerStep,
          peoplePerStep,
        }
      }
    }

    // hit reject cap / line ended
    const { reward, completed } = this.scoreEpisode(trueCounts, rejected, /*hitRejectCap=*/ true)
    return {
      admittedPrefix,
      states,
      actions,
      reward,
      rejections: rejected,
      completed,
      admittedAtEnd: admitted,
      nudgeCount,
      countsPerStep,
      peoplePerStep,
    }
  }

  /** Compute final reward with shortfall, loss, and surplus penalties. */
  private scoreEpisode(
    counts: Record<string, number>,
    rejected: number,
    hitRejectCap: boolean
  ): { reward: number; completed: boolean } {
    let totalShortfall = 0
    let quadShortfall = 0
    let totalSurplus = 0

    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const deficit = Math.max(0, c.minCount - cur)
      const surplus = Math.max(0, cur - c.minCount)

      const sw = SHORTFALL_WEIGHTS[c.attribute] ?? 1.0
      const vw = SURPLUS_WEIGHTS[c.attribute] ?? 1.0

      totalShortfall += sw * deficit
      quadShortfall += sw * deficit * deficit
      totalSurplus += vw * surplus
    }

    let reward = -rejected
    reward -= LAMBDA_SHORTFALL * totalShortfall
    reward -= QUAD_SHORTFALL * quadShortfall
    reward -= BETA_SURPLUS * totalSurplus

    // // Uncomment for debugging misses:
    // if (totalShortfall > 0) {
    //   const misses = this.game.constraints
    //     .map((c) => ({ attr: c.attribute, need: Math.max(0, c.minCount - (counts[c.attribute] || 0)) }))
    //     .filter((x) => x.need > 0)
    //     .sort((a, b) => b.need - a.need)
    //   console.log('[episode.miss]', misses.slice(0, 3))
    // }

    const satisfiedAll = totalShortfall === 0
    if (!satisfiedAll || hitRejectCap) {
      reward -= LOSS_PENALTY
    }

    return { reward, completed: satisfiedAll && !hitRejectCap }
  }

  // ---------- training over elite episodes ----------
  private trainOnEpisodes(episodes: Episode[]): number {
    if (episodes.length === 0) return 0

    const sorted = [...episodes].sort((a, b) => b.reward - a.reward)
    if (sorted[0].reward === sorted[sorted.length - 1].reward) return 0

    const eliteCount = Math.max(1, Math.floor(episodes.length * this.config.elitePercentile))
    const elite = sorted.slice(0, eliteCount)

    const X: number[][] = []
    const y: number[] = []
    const relabelFrac = clamp(this.config.oracleRelabelFrac ?? 0, 0, 1)

    const pushSample = (state: number[], label: number, repeats = 1) => {
      for (let r = 0; r < repeats; r++) {
        X.push(state)
        y.push(label)
      }
    }

    for (const ep of elite) {
      for (let i = 0; i < ep.states.length; i++) {
        const modelLabel = ep.actions[i] ? 1 : 0

        const counts = ep.countsPerStep[i]
        const person = ep.peoplePerStep[i]
        const admittedSoFar = ep.admittedPrefix[i]

        let label = modelLabel
        let repeats = 1

        if (Math.random() < relabelFrac) {
          const oracleLabel = this.oracleShouldAdmit(counts, person, admittedSoFar) ? 1 : 0
          label = oracleLabel
          // oversample steps where oracle ≠ model
          if (oracleLabel !== modelLabel) repeats += 5
        }

        // Rare-attr boost: if 'creative' still unmet and present, upweight
        const creativeConstraint = this.game.constraints.find((c) => c.attribute === 'creative')
        const needCreative = (creativeConstraint ? creativeConstraint.minCount : 0) - (counts['creative'] || 0)
        if (person['creative'] && needCreative > 0) {
          repeats += label === 1 ? 3 : 1
        }

        pushSample(ep.states[i], label, repeats)
      }
    }

    if (X.length === 0) return 0

    // ensure at least ~35% positives to avoid collapse to "deny"
    const POS_MIN = 0.35
    const posIdx: number[] = []
    for (let i = 0; i < y.length; i++) if (y[i] === 1) posIdx.push(i)
    const wantPos = Math.ceil(POS_MIN * y.length)
    if (posIdx.length > 0 && posIdx.length < wantPos) {
      const need = wantPos - posIdx.length
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

    // minibatches
    const batchSize = Math.max(1, this.config.batchSize)
    const batchCount = Math.ceil(X.length / batchSize)
    let totalLoss = 0

    for (let b = 0; b < batchCount; b++) {
      const start = b * batchSize
      const end = Math.min(start + batchSize, X.length)
      const loss = this.net.trainBatch(X.slice(start, end), y.slice(start, end), 1)
      totalLoss += loss
    }

    const posCount = y.filter((v) => v === 1).length
    const negCount = y.length - posCount
    console.log(`[train] elite samples: ${X.length} (pos=${posCount}, neg=${negCount})`)

    return totalLoss / Math.max(1, batchCount)
  }

  async train(
    epochs: number = 10,
    onEpochEnd?: (summary: {
      epoch: number
      loss: number
      successRate: number
      avgRejections: number
      exploration: number
      totalNudges: number
      bestEpisode: { admitted: number; rejections: number; reward: number }
    }) => void
  ): Promise<void> {
    console.log('Starting self-play training...')

    // If resuming, don’t crank exploration back up
    let exploration = this.resumedFromWeights ? this.config.explorationEnd : this.config.explorationStart

    for (let epoch = 0; epoch < epochs; epoch++) {
      const batch: Episode[] = []
      let successCount = 0
      let totalRejections = 0

      for (let ep = 0; ep < this.config.episodes; ep++) {
        const finishingStart = Math.max(0, epochs - 3)
        const isFinishing = epoch >= finishingStart

        const episode = this.runEpisode({
          explorationRate: exploration,
          usePolicyFusion: !isFinishing,
          useTeacherAssist: !isFinishing,
        })
        batch.push(episode)

        if (episode.completed) {
          successCount++
          totalRejections += episode.rejections
          if (!this.bestEpisode || episode.rejections < this.bestEpisode.rejections) this.bestEpisode = episode
        }
      }

      const avgRejections = successCount > 0 ? totalRejections / successCount : 20000
      const successRate = successCount / this.config.episodes
      const loss = this.trainOnEpisodes(batch)

      exploration = Math.max(this.config.explorationEnd, exploration * this.config.explorationDecay)

      // diagnostics
      const avgAdmittedAll = batch.reduce((s, e) => s + e.admittedAtEnd, 0) / batch.length
      const totalNudges = batch.reduce((s, e) => s + (e.nudgeCount ?? 0), 0)
      const bestEp = batch.reduce((b, e) => (e.reward > b.reward ? e : b), batch[0])

      console.log(`Epoch ${epoch + 1}/${epochs}:`)
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`)
      console.log(`  Avg rejections (successful): ${avgRejections.toFixed(0)}`)
      console.log(`  Best rejections: ${this.bestEpisode?.rejections || 'N/A'}`)
      console.log(`  Training loss: ${loss.toFixed(4)}`)
      console.log(`  Exploration rate: ${exploration.toFixed(3)}`)
      console.log(`  Avg admitted (all episodes): ${avgAdmittedAll.toFixed(1)}`)
      console.log(
        `  Best episode — admitted: ${bestEp.admittedAtEnd}, rejections: ${bestEp.rejections}, reward: ${bestEp.reward}`
      )
      console.log(
        `  Teacher nudges used this epoch: ${totalNudges}, assistProb(now)=${this.assistGainPreview().toFixed(4)}`
      )

      this.trainingStats.push({
        epoch: epoch + 1,
        avgReward: -avgRejections,
        avgRejections,
        successRate,
        bestRejections: this.bestEpisode?.rejections || 20000,
      })

      onEpochEnd?.({
        epoch: epoch + 1,
        loss,
        successRate,
        avgRejections,
        exploration,
        totalNudges,
        bestEpisode: {
          admitted: bestEp.admittedAtEnd,
          rejections: bestEp.rejections,
          reward: bestEp.reward,
        },
      })

      if (successRate > 0.9 && avgRejections < 1000) {
        console.log('Early stopping - excellent performance achieved!')
        break
      }
    }

    console.log('\nTraining complete!')
    if (this.bestEpisode) console.log(`Best episode: ${this.bestEpisode.rejections} rejections`)
  }

  private assistGainPreview(): number {
    const counts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (counts[k] = 0))
    return this.assistProb(counts, 0)
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

  // Configurable test
  test(
    episodes: number = 100,
    opts?: { explorationRate?: number; usePolicyFusion?: boolean; useTeacherAssist?: boolean }
  ): {
    successRate: number
    avgRejections: number
    minRejections: number
    maxRejections: number
  } {
    const { explorationRate = 0, usePolicyFusion = false, useTeacherAssist = false } = opts ?? {}

    let successes = 0
    let totalRejections = 0
    let minRejections = Infinity
    let maxRejections = 0

    for (let i = 0; i < episodes; i++) {
      const episode = this.runEpisode({
        explorationRate,
        usePolicyFusion,
        useTeacherAssist,
      })
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
    explorationDecay: 0.97,
    assistGain: 3.0,
    elitePercentile: 0.05,
    oracleRelabelFrac: 1.0, // keep high for a few epochs; then drop to ~0.5
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
