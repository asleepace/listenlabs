/** @file training.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, Person, ScenarioAttributes } from '../types'
import { NeuralNet, createBerghainNet } from './neural-net'
import { NeuralNetBouncer } from './neural-net-bouncer'
import { StateEncoder } from './state-encoder'
import { initializeScoring } from './scoring'
import { Conf } from './config'
import { clamp, toFixed } from './util'
import { Disk } from '../utils/disk'

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
  dataset?: ScenarioAttributes[]
}

function shuffle<T>(data: T[]) {
  // Unbiased Fisher–Yates shuffle
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[data[i], data[j]] = [data[j], data[i]]
  }
  return data
}

export async function getSampleGame(filePath = 'data/samples/sample-01.json'): Promise<PersonAttributesScenario2[]> {
  console.log('[dataset] loading data:', filePath)
  const guests = await Disk.getJsonFile<PersonAttributesScenario2[][]>(filePath)
  if (!guests || !Array.isArray(guests)) throw new Error(`Training: Invalid Game Data!`)
  const copy = (): PersonAttributesScenario2 => ({
    berlin_local: false,
    well_connected: false,
    creative: false,
    techno_lover: false,
  })
  // convert tuples to objects
  const guestList = guests.map((attributes) => {
    return attributes.reduce(
      (out, attribute) => ({
        ...out,
        [attribute as any]: true,
      }),
      copy()
    )
  })
  SelfPlayTrainer.lastDatasetPath = filePath
  console.log('[dataset] total entries:', guestList.length)
  return shuffle(guestList)
}

export class SelfPlayTrainer {
  static readonly DISABLE_FUSION_AT_EPOCH = false
  static readonly MAX_SAMPLES_PER_EPOCH = 250_000
  static lastDatasetPath?: string

  private net: NeuralNet
  private game: Game
  private encoder: StateEncoder
  private config: TrainingConfig
  private resumedFromWeights = false

  dataset?: ScenarioAttributes[]
  private datasetPtr = 0

  public get hasDataset() {
    return !!this.dataset
  }

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

    this.dataset = config?.dataset
    const hasDataset = !!this.dataset

    this.config = {
      episodes: 100,
      batchSize: 32,
      learningRate: 0.0003, // was 0.001
      // lower exploration if dataset is present to increase stability
      explorationStart: hasDataset ? 0.4 : 0.7,
      explorationEnd: hasDataset ? 0.1 : 0.2,
      explorationDecay: 0.95, // slightly slower anneal
      successThreshold: 5000,
      elitePercentile: 0.1, // was 0.2 (more variety than 0.05, still selective)
      teacherAssistProb: 0.0,
      assistGain: 3.0,
      oracleRelabelFrac: 0.5, // was 0.35/1.0 — 0.5 is a stable middle ground
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

  resetDatasetOrdering() {
    if (!this.dataset) throw new Error('Training: Missing data set!')
    this.dataset = shuffle(this.dataset)
    this.datasetPtr = 0
  }

  nextPersonInDataset(personIndex: number): Person<PersonAttributesScenario2> {
    if (!this.dataset) throw new Error('Training: Missing data set!')
    if (this.datasetPtr >= this.dataset.length) this.resetDatasetOrdering()
    const attributes = this.dataset.at(this.datasetPtr++) as PersonAttributesScenario2
    if (!attributes) throw new Error('Training: Failed to load next guest: ' + this.datasetPtr)
    return { attributes, personIndex }
  }

  // ---------- synthetic generator ----------
  private generatePerson(index: number): Person<PersonAttributesScenario2> {
    if (this.dataset) return this.nextPersonInDataset(index)

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
            const p = clamp(stats.relativeFrequencies[a2] * (1 + corr * 0.5), [0, 1])
            samples[a2] = Math.random() < p
          }
        }
      }
    }
    for (const k of Object.keys(stats.relativeFrequencies)) {
      attributes[k] = !!samples[k]
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
    const remaining = Math.max(1, Conf.MAX_ADMISSIONS - admitted)
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
    const remaining = Math.max(1, Conf.MAX_ADMISSIONS - admitted)
    const { gap } = this.worstExpectedGap(counts, remaining)
    const g = gap / remaining // normalized [0..1+]
    return clamp(this.config.assistGain! * g, [0, 0.65])
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
      baseThreshold: 0.32,
      minThreshold: 0.22,
      maxThreshold: 0.62,
      urgencyFactor: 2.0,
    })
    bouncer.setNetwork(this.net)

    const states: number[][] = []
    const actions: boolean[] = []
    const countsPerStep: Array<Record<string, number>> = []
    const peoplePerStep: Array<Record<string, boolean>> = []
    const admittedPrefix: number[] = []

    let nudgeCount = 0

    // seat/urgency scoring engine
    const scoring = initializeScoring(this.game, {
      maxAdmissions: Conf.MAX_ADMISSIONS,
      maxRejections: Conf.MAX_REJECTIONS,
      targetRejections: Conf.TARGET_REJECTIONS,
      safetyCushion: Conf.SAFETY_CUSHION,
      weights: {
        // add overrides here
      },
    })

    while (scoring.inProgress()) {
      // ONE person per step: same sample for scoring + NN
      const person = this.generatePerson(scoring.nextIndex)
      const counts = scoring.getCounts()

      // snapshots (pre-action)
      countsPerStep.push({ ...counts })
      peoplePerStep.push({ ...person.attributes })
      admittedPrefix.push(scoring.admitted)

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: scoring.admitted,
        rejectedCount: scoring.rejected,
        nextPerson: person,
      }

      // encode state with current true counts
      const state = this.encoder.encode(status, counts)
      states.push(state)

      // --- hybrid decision (scoring + network) ---
      const guest = person.attributes as ScenarioAttributes
      let admit = bouncer.admit(status, counts)

      if (usePolicyFusion) {
        const policyVote = scoring.shouldAdmit(guest, 1.0, 0.5)
        const helpsWorstGap = this.oracleShouldAdmit(counts, person.attributes, scoring.admitted)
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
        const pAssist = this.assistProb(counts, scoring.admitted)
        if (Math.random() < pAssist) {
          const oracle = this.oracleShouldAdmit(counts, person.attributes, scoring.admitted)
          if (oracle !== admit) {
            admit = oracle
            if (admit) nudgeCount++
          }
        }
      }

      actions.push(admit)

      // apply decision to both trackers
      scoring.update({ guest, admit })
    }

    // the game is considered won if we have finished all of our quotas by the
    // time we run out of space for admissions or rejections.
    const isSuccess = scoring.isFinishedWithQuotas()
    const counts = scoring.getCounts()
    const rejected = scoring.rejected
    const admitted = scoring.admitted

    const { reward, completed } = this.scoreEpisode(counts, rejected, isSuccess)
    return {
      admittedPrefix,
      completed,
      states,
      actions,
      reward,
      rejections: rejected,
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
    isCompleted: boolean
  ): { reward: number; completed: boolean; shortfalls: Array<{ attr: string; need: number }> } {
    let totalShortfall = 0,
      quadShortfall = 0,
      totalSurplus = 0
    const shortfalls: Array<{ attr: string; need: number }> = []

    for (const c of this.game.constraints) {
      const cur = counts[c.attribute] || 0
      const deficit = Math.max(0, c.minCount - cur)
      const surplus = Math.max(0, cur - c.minCount)
      if (deficit > 0) shortfalls.push({ attr: c.attribute, need: deficit })

      const sw = Conf.TRAINING.getShortfallWeight({ attribute: c.attribute, default: 1.0 })
      const vw = Conf.TRAINING.getSurplusWeight({ attribute: c.attribute, default: 1.0 })

      totalShortfall += sw * deficit
      quadShortfall += sw * deficit * deficit
      totalSurplus += vw * surplus
    }

    let reward = -rejected
    reward -= Conf.TRAINING.LAMBDA_SHORTFALL * totalShortfall
    reward -= Conf.TRAINING.QUAD_SHORTFALL * quadShortfall
    reward -= Conf.TRAINING.BETA_SURPLUS * totalSurplus

    const satisfiedAll = shortfalls.length === 0
    if (!satisfiedAll || !isCompleted) reward -= Conf.TRAINING.LOSS_PENALTY

    return { reward, completed: satisfiedAll && isCompleted, shortfalls }
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
    const relabelFrac = clamp(this.config.oracleRelabelFrac ?? 0, [0, 1])

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
          // Oversample hard examples more aggressively
          if (oracleLabel !== modelLabel) repeats += 4
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

    if (X.length > SelfPlayTrainer.MAX_SAMPLES_PER_EPOCH) {
      const keep = SelfPlayTrainer.MAX_SAMPLES_PER_EPOCH
      const perm = Array.from({ length: X.length }, (_, i) => i)
      for (let i = 0; i < keep; i++) {
        const j = i + Math.floor(Math.random() * (perm.length - i))
        ;[perm[i], perm[j]] = [perm[j], perm[i]]
      }
      const picked = perm.slice(0, keep)
      const X2 = picked.map((i) => X[i])
      const y2 = picked.map((i) => y[i])
      X.length = 0
      y.length = 0
      X.push(...X2)
      y.push(...y2)
    }

    // ensure at least ~40% positives to avoid collapse to "deny"
    // If you notice the NN getting too “admit-happy”, raise the positive floor a touch:
    const posIdx: number[] = []
    for (let i = 0; i < y.length; i++) if (y[i] === 1) posIdx.push(i)
    const wantPos = Math.ceil(Conf.POS_MIN * y.length)
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
    let bestSuccess = 0,
      stall = 0

    for (let epoch = 0; epoch < epochs; epoch++) {
      if (this.dataset) this.resetDatasetOrdering()
      const batch: Episode[] = []
      let successCount = 0
      let totalRejections = 0

      for (let ep = 0; ep < this.config.episodes; ep++) {
        const finishingStart = Math.max(0, epochs - 3)

        // Keep assist/fusion on the whole time for stability during bring-up
        const isFinishing = SelfPlayTrainer.DISABLE_FUSION_AT_EPOCH ? epoch >= finishingStart : false

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

      const avgRejections = successCount > 0 ? totalRejections / successCount : Conf.MAX_REJECTIONS
      const successRate = successCount / this.config.episodes
      const loss = this.trainOnEpisodes(batch)

      exploration = Math.max(this.config.explorationEnd, exploration * this.config.explorationDecay)

      // diagnostics
      const avgAdmittedAll = batch.reduce((s, e) => s + e.admittedAtEnd, 0) / batch.length
      const totalNudges = batch.reduce((s, e) => s + (e.nudgeCount ?? 0), 0)
      const bestEp = batch.reduce((b, e) => (e.reward > b.reward ? e : b), batch[0])
      const bestEpInfo = this.scoreEpisode(
        bestEp.countsPerStep.at(-1) ?? {}, // or track final counts explicitly
        bestEp.rejections,
        bestEp.completed
      )

      console.log(`Epoch ${epoch + 1}/${epochs}:`)
      console.log(`  Using dataset:`, this.hasDataset ? SelfPlayTrainer.lastDatasetPath : false)
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`)
      console.log(`  Avg rejections (successful):`, +avgRejections.toFixed(0))
      console.log(`  Avg admitted (all episodes):`, +avgAdmittedAll.toFixed(0))
      console.log(`  Best rejections: ${this.bestEpisode?.rejections || 'N/A'}`)
      console.log(`  Training loss: ${loss.toFixed(4)}`)
      console.log(`  Exploration rate: ${exploration.toFixed(3)}`)
      console.log(`  Success:`, successRate !== 0)
      const label = bestEp.completed ? 'SUCCESS' : 'FAIL'
      console.log(
        `  Best episode — (${label}) admitted: ${bestEp.admittedAtEnd}, rejections: ${bestEp.rejections}, reward: ${bestEp.reward}`
      )
      if (!bestEp.completed && bestEpInfo.shortfalls.length) {
        console.log('  Missed quotas:', bestEpInfo.shortfalls.map((s) => `${s.attr}:${s.need}`).join(', '))
      }
      console.log(
        `  Teacher nudges used this epoch: ${totalNudges}, assistProb(now)=${this.assistGainPreview().toFixed(4)}`
      )

      this.trainingStats.push({
        epoch: epoch + 1,
        avgReward: -avgRejections,
        avgRejections,
        successRate,
        bestRejections: this.bestEpisode?.rejections || Conf.MAX_REJECTIONS,
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

      if ((epoch + 1) % 3 === 0) {
        const cur = this.net.getLearningRate()
        this.net.setLearningRate(cur * 0.7)
        console.log(`[lr] decayed to ${this.net.getLearningRate()}`)
      }

      if (successRate >= bestSuccess) {
        bestSuccess = successRate
        stall = 0
      } else stall++
      if (bestSuccess >= 0.99 && stall >= 2) {
        console.log('Early stopping: success regressed, keeping best weights.')
        break
      }

      if (successRate > 0.9 && avgRejections < Conf.TARGET_REJECTIONS) {
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
    avgAdmissions: number
    minRejections: number
    maxRejections: number
  } {
    const { explorationRate = 0, usePolicyFusion = false, useTeacherAssist = false } = opts ?? {}

    let successes = 0
    let totalRejections = 0
    let totalAdmissions = 0
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
        totalAdmissions += episode.admittedAtEnd
        minRejections = Math.min(minRejections, episode.rejections)
        maxRejections = Math.max(maxRejections, episode.rejections)
      }
    }

    return {
      successRate: successes / episodes,
      avgRejections: successes > 0 ? totalRejections / successes : Conf.MAX_REJECTIONS,
      avgAdmissions: successes > 0 ? totalAdmissions / successes : 0,
      minRejections: successes > 0 ? minRejections : Conf.MAX_REJECTIONS,
      maxRejections: successes > 0 ? maxRejections : Conf.MAX_REJECTIONS,
    }
  }
}

export async function trainBouncer(game: Game): Promise<NeuralNetBouncer> {
  const trainer = new SelfPlayTrainer(game, {
    episodes: 50,
    batchSize: 32,
    learningRate: 0.0003,
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
  if (trainer.hasDataset) {
    console.log(`Dataset file:`, SelfPlayTrainer.lastDatasetPath)
    console.log(`Dataset count:`, trainer.dataset?.length)
  }
  console.log(`Success rate: ${(results.successRate * 100).toFixed(1)}%`)
  console.log(`Average admissions:`, toFixed(results.avgAdmissions))
  console.log(`Average rejections:`, toFixed(results.avgRejections))
  console.log(`Rejections Worst=${results.maxRejections}, Best=${results.minRejections}`)

  const bouncer = new NeuralNetBouncer(game)
  bouncer.setNetwork(trainer.getNetwork())
  return bouncer
}
