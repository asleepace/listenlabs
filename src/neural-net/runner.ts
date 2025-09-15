/** @file runner.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, BerghainBouncer, ScenarioAttributes } from '../types'

import { NeuralNetBouncer } from './neural-net-bouncer'
import { getSampleGame, SelfPlayTrainer } from './training'
import { initializeScoring } from './scoring'
import * as fs from 'fs'
import * as path from 'path'
import { createBerghainNet, NeuralNet } from './neural-net'
import { clamp, parseFlags, toFixed } from './util'
import { Conf } from './config'
import { StateEncoder } from './state-encoder'
import { Disk } from '../utils/disk'
import { Try } from '@asleepace/try'

type RunGameIteration = {
  status: GameStatusRunning<ScenarioAttributes>
  scoring: ReturnType<typeof initializeScoring>
}

export class NeuralNetBouncerRunner {
  public bouncer: NeuralNetBouncer | null = null
  public game: Game | null = null
  public weightsPath: string
  public logPath: string

  constructor(private dataDir: string = './bouncer-data', private scenario: '2' = '2') {
    if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true })
    this.weightsPath = path.resolve(path.join(dataDir, `weights-scenario-${scenario}.json`))
    this.logPath = path.resolve(path.join(dataDir, `training-log-${scenario}.json`))
  }

  private initializeGame(): Game {
    return {
      gameId: `scenario-${this.scenario}-${Date.now()}`,
      constraints: [
        { attribute: 'techno_lover', minCount: 650 },
        { attribute: 'well_connected', minCount: 450 },
        { attribute: 'creative', minCount: 300 },
        { attribute: 'berlin_local', minCount: 750 },
      ],
      attributeStatistics: {
        relativeFrequencies: {
          techno_lover: 0.6265,
          well_connected: 0.47,
          creative: 0.06227,
          berlin_local: 0.398,
        },
        correlations: {
          techno_lover: { techno_lover: 1, well_connected: -0.4696, creative: 0.0946, berlin_local: -0.6549 },
          well_connected: { techno_lover: -0.4696, well_connected: 1, creative: 0.142, berlin_local: 0.5724 },
          creative: { techno_lover: 0.0946, well_connected: 0.142, creative: 1, berlin_local: 0.1445 },
          berlin_local: { techno_lover: -0.6549, well_connected: 0.5724, creative: 0.1445, berlin_local: 1 },
        },
      },
    }
  }

  // ---------- training / resume ----------
  private writeCheckpoint(trainer: SelfPlayTrainer, epoch: number) {
    const ckptPath = this.weightsPath.replace(/\.json$/, `.epoch-${epoch}.json`)
    fs.writeFileSync(ckptPath, JSON.stringify(trainer.getBestWeights(), null, 2))
    // also refresh "latest" weights each epoch
    fs.writeFileSync(this.weightsPath, JSON.stringify(trainer.getBestWeights(), null, 2))
  }

  private appendLog(entry: any) {
    let log: any = null
    if (fs.existsSync(this.logPath)) {
      try {
        log = JSON.parse(fs.readFileSync(this.logPath, 'utf-8'))
      } catch (e) {
        console.warn(e)
      }
    }
    if (!log) log = { runs: [] }
    if (!log?.runs) log.runs = []
    log.runs.push(entry)
    fs.writeFileSync(this.logPath, JSON.stringify(log, null, 2))
  }

  async train(epochs = 20, episodesPerEpoch = 50, flags: Record<string, string> = {}): Promise<void> {
    console.log('=== Neural Network Bouncer Training ===\n')
    this.game = this.initializeGame()

    const assistGain = flags.assistGain ? Number(flags.assistGain) : 2.0
    const oracleRelabelFrac = flags.oracleRelabelFrac ? Number(flags.oracleRelabelFrac) : 0.35
    const elitePercentile = flags.elitePercentile ? Number(flags.elitePercentile) : 0.2
    const resumeFlag = flags.resume === 'true'

    const explorationStart = flags.explorationStart ? Number(flags.explorationStart) : 0.4
    const explorationEnd = flags.explorationEnd ? Number(flags.explorationEnd) : 0.1
    const explorationDecay = flags.explorationDecay ? Number(flags.explorationDecay) : 0.9

    // optional dataset for training
    const samples = Array(8)
      .fill(0)
      .map((_, i) => `data/samples/sample-0${i + 1}.json`)
    const randomSample = samples.at(Math.floor(samples.length * Math.random()))

    const datafile = (flags.datafile || flags.data || randomSample) as string
    const optionalDataset = await Try.catch(async () => {
      return await getSampleGame(datafile)
    })

    console.log('[resume] found dataset:', optionalDataset.isOk())

    const trainer = new SelfPlayTrainer(this.game, {
      episodes: episodesPerEpoch,
      batchSize: 64,
      learningRate: 0.0003,
      explorationStart,
      explorationEnd,
      explorationDecay,
      successThreshold: 5_000,
      elitePercentile,
      assistGain,
      oracleRelabelFrac,
      dataset: optionalDataset.unwrapOr(undefined),
    })

    if (resumeFlag && fs.existsSync(this.weightsPath)) {
      try {
        const bestPath = this.weightsPath.replace(/\.json$/, '.best.json')
        const pathToLoad = fs.existsSync(bestPath) ? bestPath : this.weightsPath
        const weights = JSON.parse(fs.readFileSync(pathToLoad, 'utf-8'))
        console.log(`Loaded weights from: ${pathToLoad}`)
        trainer.loadWeights(weights)
        trainer.getNetwork().setLearningRate(0.0001)
        console.log(`[resume] Warm-started from ${this.weightsPath}`)
      } catch (e) {
        console.warn('[resume] Failed to load saved weights; starting fresh.', (e as Error).message)
      }
    }

    const startedAt = new Date().toISOString()

    await trainer.train(epochs, (summary) => {
      this.writeCheckpoint(trainer, summary.epoch)
      this.appendLog({
        type: 'epoch',
        scenario: this.scenario,
        startedAt,
        timestamp: new Date().toISOString(),
        summary,
      })
    })

    const testResults = trainer.test(100)
    console.log('\n=== Testing Trained Network ===\n')

    const pure = trainer.test(100, { explorationRate: 0, usePolicyFusion: false, useTeacherAssist: false })
    console.log('Pure NN (no fusion, no assist):')
    console.log(`  Success Rate: ${(pure.successRate * 100).toFixed(1)}%`)
    console.log(`  Average Rejections:`, +pure.avgRejections.toFixed(0))
    console.log(`  Average Admissions:`, +pure.avgAdmissions.toFixed(0))
    console.log(`  Best Performance: ${pure.minRejections} rejections`)
    console.log(`  Worst Performance: ${pure.maxRejections} rejections`)

    const hybridEval = trainer.test(100, { explorationRate: 0, usePolicyFusion: true, useTeacherAssist: false })
    console.log('\nHybrid (policy fusion on, no assist):')
    console.log(`  Success Rate: ${(hybridEval.successRate * 100).toFixed(1)}%`)
    console.log(`  Average Rejections:`, +hybridEval.avgRejections.toFixed(0))
    console.log(`  Average Admissions:`, +hybridEval.avgAdmissions.toFixed(0))
    console.log(`  Best Performance: ${hybridEval.minRejections} rejections`)
    console.log(`  Worst Performance: ${hybridEval.maxRejections} rejections`)

    // Save final weights (+ run summary)
    fs.writeFileSync(this.weightsPath, JSON.stringify(trainer.getBestWeights(), null, 2))
    console.log(`\nWeights saved to: ${this.weightsPath}`)

    this.appendLog({
      type: 'run',
      scenario: this.scenario,
      startedAt,
      finishedAt: new Date().toISOString(),
      params: { epochs, episodesPerEpoch, assistGain, oracleRelabelFrac, elitePercentile, resumeFlag },
      testResults,
      trainingStats: trainer.getStats(),
    })

    this.bouncer = new NeuralNetBouncer(this.game, { explorationRate: 0, baseThreshold: 0.5, softGates: true })
    this.bouncer.setNetwork(trainer.getNetwork())
  }

  /** Explicit resume command: always attempts to load weights first. */
  async resume(epochs = 10, episodesPerEpoch = 50, flags: Record<string, string> = {}): Promise<void> {
    flags.resume = 'true'
    await this.train(epochs, episodesPerEpoch, flags)
  }

  // ---------- load / diagnose / runGame ----------
  load(): boolean {
    if (!fs.existsSync(this.weightsPath)) {
      console.log('No saved weights found. Please train first.')
      return false
    }
    try {
      const weights = JSON.parse(fs.readFileSync(this.weightsPath, 'utf-8'))
      this.game = this.initializeGame()

      // Build a fresh bouncer with zero exploration
      this.bouncer = new NeuralNetBouncer(this.game, {
        explorationRate: 0,
        baseThreshold: 0.28,
        minThreshold: 0.18,
        maxThreshold: 0.6,
        urgencyFactor: 1.5, // a little less tightening
        optimism: 0.8, // bigger slack before gates kick in
        isProduction: true,
        softGates: true,
      })

      // Prefer static loader; fall back to instance methods if present.
      let net: NeuralNet
      try {
        net = NeuralNet.fromJSON(weights)
      } catch (e) {
        console.warn('[runner] failed to load neural net from JSON:', e)
        const enc = new StateEncoder(this.game)
        net = createBerghainNet(enc.getFeatureSize())
        ;(net as any).fromJSON?.(weights) || (net as any).loadJSON?.(weights) || (net as any).load?.(weights)
      }

      if (!net) {
        throw new Error('Runner: Failed to create neural net!')
      }

      this.bouncer.setNetwork(net)

      console.log('Weights loaded successfully!')
      return true
    } catch (error) {
      console.error('[runner.load] Failed to load weights:', (error as Error).message)
      return false
    }
  }

  // Greedy score used by diagnose()
  private greedyOracleScore(
    personAttrs: Record<string, boolean>,
    counts: Record<string, number>,
    game: Game,
    admitted: number
  ): number {
    const remaining = Math.max(0, Conf.MAX_ADMISSIONS - admitted)
    let score = 0
    for (const c of game.constraints) {
      const current = counts[c.attribute] || 0
      const need = Math.max(0, c.minCount - current)
      if (need <= 0) continue
      const pressure = remaining > 0 ? need / remaining : 0
      if (personAttrs[c.attribute]) score += 1 + 3 * pressure // favor urgent attributes
    }
    return score
  }

  diagnose(samples = 100_000) {
    this.game = this.initializeGame()
    const counts: Record<string, number> = {}
    Object.keys(this.game.attributeStatistics.relativeFrequencies).forEach((k) => (counts[k] = 0))

    // build a candidate pool
    const pool: PersonAttributesScenario2[] = []
    for (let i = 0; i < samples; i++) {
      const attrs = this.generatePerson(i)
      const person: PersonAttributesScenario2 = {} as any
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        person[key as keyof PersonAttributesScenario2] = false
      }
      for (const a of attrs) person[a as keyof PersonAttributesScenario2] = true
      pool.push(person)
    }

    // greedy pick
    const chosen: PersonAttributesScenario2[] = []
    while (chosen.length < Conf.MAX_ADMISSIONS && pool.length > 0) {
      let bestIdx = -1
      let bestScore = -Infinity
      for (let i = 0; i < pool.length; i++) {
        const s = this.greedyOracleScore(pool[i], counts, this.game, chosen.length)
        if (s > bestScore) {
          bestScore = s
          bestIdx = i
        }
      }
      const pick = pool.splice(bestIdx, 1)[0]
      chosen.push(pick)
      for (const [k, v] of Object.entries(pick)) if (v) counts[k] = (counts[k] || 0) + 1
    }

    console.log('\n=== Feasibility (Greedy Oracle over sampled pool) ===')
    for (const c of this.game.constraints) {
      console.log(`  ${c.attribute}: ${counts[c.attribute] || 0}/${c.minCount}`)
    }
    const satisfied = this.game.constraints.every((c) => (counts[c.attribute] || 0) >= c.minCount)
    console.log(`  Result: ${satisfied ? 'POSSIBLE (greedy met all)' : 'LIKELY INFEASIBLE (even greedy failed)'}`)
  }

  getBouncer(): BerghainBouncer | null {
    return this.bouncer
  }

  assertBouncerExists(): asserts this is { bouncer: NeuralNetBouncer } {
    if (!this.bouncer) throw new Error('Bouncer has not been initialized!')
  }

  handleModeBouncer({ scoring, status }: RunGameIteration) {
    this.assertBouncerExists()
    const guest = status.nextPerson.attributes
    const count = scoring.getCounts()
    this.bouncer.setCounts(count)
    const admit = this.bouncer.admit(status, count)
    scoring.update({ guest, admit })
  }

  handleModeHybrid({ scoring, status }: RunGameIteration) {
    this.assertBouncerExists()
    const guest = status.nextPerson.attributes
    const count = scoring.getCounts()
    this.bouncer.setCounts(count)
    const policyVote = scoring.shouldAdmit(guest, 1.0, 0.5)
    const netVote = this.bouncer.admit(status, count)
    let admit = policyVote || netVote
    if (scoring.isRunningOutOfAvailableSpots()) {
      const before = scoring.worstExpectedShortfall(scoring.getPeopleLeftInLine())
      const after = scoring.worstExpectedShortfall(Math.max(0, scoring.getPeopleLeftInLine() - 1), guest)
      const helpsWorstGap = after + 1e-9 < before
      admit = admit && helpsWorstGap
    }
    scoring.update({ guest, admit })
  }

  handleModeScore({ scoring, status }: RunGameIteration) {
    const guest = status.nextPerson.attributes
    const admit = scoring.shouldAdmit(guest, 1.0, 0.5)
    scoring.update({ guest, admit })
  }

  // ---------- handle running a course on all sample data ----------
  async curriculum({ positional, runner }: { positional: any[]; runner: NeuralNetBouncerRunner }) {
    // discover and order datasets
    const allFiles = (await Disk.getFilePathsInDir(`data/samples/*`))
      .filter((f) => f.endsWith('.json'))
      .sort((a, b) => a.localeCompare(b))

    const epochsPerSample = 4
    const samples = allFiles.map((fp) => `${fp}:${epochsPerSample}`)
    samples.push('none:2') // end with a short self-play consolidation
    const defaultSampleString = samples.join(',')

    const getLastPath = (path: string) => {
      return path.split('samples/').at(-1) ?? path
    }

    console.log('[trainer] curriculum:', samples.map(getLastPath))

    const phasesArg: string = positional[0] || defaultSampleString

    // Base episodes per epoch. We’ll optionally scale this per phase for uneven sizes.
    const baseEpisodes = parseInt(positional[1] || '150', 10)

    // Build game and trainer fresh (no runner.train side-effects)
    const game = (runner as any).initializeGame()
    const { SelfPlayTrainer, getSampleGame } = await import('./training')

    const trainer = new SelfPlayTrainer(game, {
      episodes: baseEpisodes,
      explorationStart: 0.1,
      explorationEnd: 0.05,
      oracleRelabelFrac: 0.6,
      elitePercentile: 0.12,
      assistGain: 3,
    })

    const weightsPath = runner.weightsPath

    // Track best checkpoint across the entire curriculum
    let bestAcross = { rej: Infinity as number, json: trainer.getBestWeights() }

    const parsePhases = (arg: string) =>
      arg.split(',').map((s) => {
        const [file, e] = s.split(':')
        return { file, epochs: Math.max(1, parseInt(e || '3', 10)) }
      })

    const phases = parsePhases(phasesArg)

    for (const phase of phases) {
      let dataset: any[] | undefined = undefined
      let phaseEpisodes = baseEpisodes

      if (phase.file !== 'none' && phase.file) {
        try {
          // load + shuffle dataset
          dataset = await getSampleGame(phase.file)
          // OPTIONAL: scale episodes to dataset size (rough guidance)
          // ~1 episode ~= 1000 decisions; tune to your env:
          const k = Math.max(0.5, Math.min(2.0, dataset.length / 6000)) // 6k is mid-sized
          phaseEpisodes = Math.max(80, Math.round(baseEpisodes * k))
        } catch (e) {
          console.warn(`[curriculum] failed to load ${getLastPath(phase.file)}:`, (e as Error).message)
          continue
        }
      }

      // swap dataset (or switch to self-play)
      trainer.setDataset(dataset as any)
      if (phase.file !== 'none') (SelfPlayTrainer as any).lastDatasetPath = phase.file

      // temporarily adjust episodes for this phase
      const prevEpisodes = (trainer as any).config.episodes
      ;(trainer as any).config.episodes = phaseEpisodes

      console.log(`[phase] ${getLastPath(phase.file)} — epochs=${phase.epochs}, episodes/epoch=${phaseEpisodes}`)

      await trainer.train(phase.epochs, async (summary) => {
        // save rolling weights every epoch
        await Disk.saveJsonFile(weightsPath, trainer.getBestWeights())
        console.log(`[phase save] ${getLastPath(phase.file)} @ epoch ${summary.epoch} → `, { weightsPath })
      })

      // restore episodes for next phase
      ;(trainer as any).config.episodes = prevEpisodes

      // quick pure-NN eval (consistent with your runner’s test)
      const evalPure = trainer.test(60, { explorationRate: 0, usePolicyFusion: false, useTeacherAssist: false })
      console.log(
        `[phase eval] ${getLastPath(phase.file)} → success=${(evalPure.successRate * 100).toFixed(1)}% , avgRej=`,
        toFixed(evalPure.avgRejections)
      )

      if (evalPure.successRate > 0 && evalPure.avgRejections < bestAcross.rej) {
        bestAcross = { rej: evalPure.avgRejections, json: trainer.getBestWeights() }
        await Disk.saveJsonFile(weightsPath.replace(/\.json$/, '.best.json'), bestAcross.json)
        console.log(
          `[best-so-far] avgRej=${bestAcross.rej.toFixed(0)} → ${weightsPath.replace(/\.json$/, '.best.json')}`
        )
      }
    }

    // Final save (best across all phases)
    await Disk.saveJsonFile(weightsPath, bestAcross.json)
    console.log(`[curriculum] done. best avg rejections ~ `, toFixed(bestAcross.rej, 0), ` | saved → ${weightsPath}`)
  }

  /**
   *  ========= RUN GAME =========
   */
  runGame(sampleData?: any[], mode: 'score' | 'bouncer' | 'hybrid' = 'score') {
    if (!this.game) throw new Error('Game not initialized')

    // --- initialize scoring for all bookkeeping ---
    const scoring = initializeScoring(this.game, {
      maxRejections: Conf.MAX_REJECTIONS,
      maxAdmissions: Conf.MAX_ADMISSIONS,
      targetRejections: Conf.TARGET_REJECTIONS,
      safetyCushion: 1,
    })

    // after:
    function fisherYates<T>(arr: T[]) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
      }
      return arr
    }

    let loopPool: any[] | undefined
    let ptr = 0

    const getData = sampleData
      ? () => {
          if (!loopPool) loopPool = fisherYates(sampleData.slice())
          if (ptr >= loopPool.length) {
            loopPool = fisherYates(loopPool) // reshuffle each wrap
            ptr = 0
          }
          return loopPool[ptr++]
        }
      : () => this.generatePerson(scoring.nextIndex)

    const getOrGenerateNextGuest = (): ScenarioAttributes => {
      if (!this.game) throw new Error('Game not initialized')
      const personData = getData()

      // normalize
      const attributes: PersonAttributesScenario2 = {} as any
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        attributes[key] = false
      }
      if (Array.isArray(personData)) {
        for (const a of personData) if (a in attributes) attributes[a] = true
      } else if (personData && typeof personData === 'object') {
        for (const [k, v] of Object.entries(personData)) if (k in attributes) attributes[k] = !!v
      } else {
        console.log(personData)
        throw new Error('unknown type of data!')
      }

      return attributes as ScenarioAttributes
    }

    while (scoring.inProgress()) {
      // --- get or generate the next guest ---
      const guest = getOrGenerateNextGuest()

      // --- all quotas have been met (auto-admit) ---
      if (scoring.isFinishedWithQuotas()) {
        scoring.update({ guest, admit: true })
        continue
      }

      // --- generate mock game status ---
      const status: GameStatusRunning<ScenarioAttributes> = {
        status: 'running',
        admittedCount: scoring.admitted,
        rejectedCount: scoring.rejected,
        nextPerson: { personIndex: scoring.nextIndex + 1, attributes: guest },
      }

      // --- handle each mode ---
      switch (mode) {
        case 'bouncer': {
          this.handleModeBouncer({ scoring, status })
          continue
        }
        case 'hybrid': {
          this.handleModeHybrid({ scoring, status })
          continue
        }
        case 'score': {
          this.handleModeScore({ scoring, status })
          continue
        }
      }
    }

    // NOTE: The game is considered won as long as we meet all the quotas and we
    // have one of the following conditions:
    //  - 1,000 total admissions
    //  - 10,000 total rejections
    return scoring.getSummary()
  }

  /** Sample a synthetic person as a list of attribute names set to true. */
  private generatePerson(index: number): string[] {
    if (!this.game) throw new Error('Game not initialized')

    const attributes: string[] = []
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    // base samples
    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < clamp(freq, [0, 1])
    }

    // apply correlations (positive -> boost, negative -> reduce)
    for (const [attr1, correlations] of Object.entries(stats.correlations)) {
      if (samples[attr1]) {
        for (const [attr2, corr] of Object.entries(correlations)) {
          if (attr1 !== attr2 && Math.abs(corr) > 0.3) {
            const base = stats.relativeFrequencies[attr2] ?? 0
            const adjustedProb = clamp(base * (1 + corr * 0.5), [0, 1])
            samples[attr2] = Math.random() < adjustedProb
          }
        }
      }
    }

    for (const [attr, hasAttr] of Object.entries(samples)) if (hasAttr) attributes.push(attr)
    return attributes
  }
}

export async function main() {
  const runner = new NeuralNetBouncerRunner('./bouncer-data', '2')
  const argv = process.argv.slice(2)
  const command = argv[0] || 'help'
  const rest = argv.slice(1)

  // split rest into positionals and flags
  const positional: string[] = []
  const flagArgs: string[] = []
  for (const a of rest) {
    if (a.startsWith('--')) flagArgs.push(a)
    else positional.push(a)
  }
  const flags = parseFlags(flagArgs)
  const mode = (flags.mode as 'score' | 'bouncer' | 'hybrid') || 'score'

  switch (command) {
    case 'curriculum': {
      await runner.curriculum({ positional, runner })
      break
    }
    case 'train': {
      const epochs = parseInt(positional[0] || '20', 10)
      const episodes = parseInt(positional[1] || '50', 10)
      await runner.train(epochs, episodes, flags)
      break
    }
    case 'resume': {
      const epochs = parseInt(positional[0] || '10', 10)
      const episodes = parseInt(positional[1] || '50', 10)
      await runner.resume(epochs, episodes, flags)
      break
    }
    case 'test': {
      if (!runner.load()) {
        console.log('Please train the model first: bun run src/neural-net/runner train 20')
        break
      }
      const datafile = positional[0]
      let sampleData: any[] | null = null
      if (datafile) {
        try {
          const data = fs.readFileSync(datafile, 'utf-8')
          sampleData = JSON.parse(data)
          console.log(`Loaded ${sampleData?.length} samples from ${datafile}`)
        } catch (error) {
          console.error('Error loading sample data:', (error as Error).message)
        }
      }
      console.log('\n=== Running Test Game ===\n')
      const result = runner.runGame(sampleData || undefined, mode)
      console.log('Mode:', mode)
      console.log('Game Result:')
      console.log(`  Status: ${result.status}`)
      console.log(`  Success:`, result.success)
      console.log(`  Final Rejections:`, result.finalRejections)
      console.log('  Constraints:')
      result.constraints.forEach((c: any) => {
        const ok = c.satisfied ? '✓' : '✗'
        console.log(`    ${ok} ${c.attribute}: ${c.current}/${c.required}`)
      })
      break
    }
    case 'benchmark': {
      if (!runner.load()) {
        console.log('Please train the model first: bun run src/neural-net/runner train 20')
        break
      }

      console.log('\n=== Running Benchmark (10 games) ===\n')
      console.log(`Mode: ${mode}`)

      const results: any[] = []
      let successes = 0
      let targetHits = 0

      for (let i = 0; i < 10; i++) {
        const result = runner.runGame(undefined, mode)
        results.push(result)

        const okMix = result.status === 'completed' // constraints satisfied
        const finalRejections = result.finalRejections ?? 0
        const hitTarget = finalRejections <= Conf.TARGET_REJECTIONS

        if (okMix) {
          successes++
          if (!hitTarget) {
            console.warn(`[benchmark] hit mix but missed target: rej=${finalRejections} (> ${Conf.TARGET_REJECTIONS})`)
          } else {
            targetHits++
          }
          console.log(`Game ${i + 1}: SUCCESS - ${finalRejections} rejections`)
        } else {
          console.log(`Game ${i + 1}: FAILED`)
        }
      }

      const successfulGames = results.filter((r) => r.status === 'completed')
      const avgRejections =
        successfulGames.length > 0
          ? successfulGames.reduce((sum, r) => sum + r.finalRejections, 0) / successfulGames.length
          : 0

      console.log(`\nBenchmark Results:`)
      console.log(`  Success Rate (mix met): ${((successes / 10) * 100).toFixed(0)}%`)
      console.log(`  Average Rejections (successful only): ${avgRejections.toFixed(0)}`)
      console.log(`  Target Hits: ${targetHits}/${successes} (<= ${Conf.TARGET_REJECTIONS})`)
      break
    }

    case 'validate': {
      const samples = await Disk.getFilePathsInDir('data/samples/*')
      samples.forEach(async (sample) => {
        await Bun.$`bun run neural test ${sample} --mode=bouncer`
      })
      await Bun.$`bun run src/neural-net/runner benchmark --mode=bouncer`
      break
    }
    case 'sanity': {
      const out1 = await Bun.$`bun run src/neural-net/runner test "" --mode=score`.text()
      const out2 = await Bun.$`bun run src/neural-net/runner test "" --mode=bouncer`.text()
      const out3 = await Bun.$`bun run src/neural-net/runner test "" --mode=hybrid`.text()
      console.log(out1)
      console.log(out2)
      console.log(out3)
      return
    }
    case 'diagnose':
      runner.diagnose(100_000)
      break
    default: {
      console.log('Neural Network Bouncer Runner')
      console.log('\nCommands:')
      console.log(
        '  train [epochs] [episodes] [--assistGain=2] [--oracleRelabelFrac=0.35] [--elitePercentile=0.2] [--resume]'
      )
      console.log('  test [datafile] [--mode=score|bouncer|hybrid]    - Run a single test game')
      console.log('  resume [epochs] [episodes]                       - Continue training from saved weights')
      console.log('  benchmark [--mode=score|bouncer|hybrid]          - Run 10 games and show statistics')
      console.log('  validate                                         - Run test on on all samples in data/smaples/*')
      console.log('  diagnose                                         - Greedy feasibility check over a sampled pool')
      console.log('  curriculum                                       - train data on samples files in directory')
      console.log('  sanity                                           - run tests in all three modes (alias)')
    }
  }
}

/**
 *  ## Neural Entry Points
 *
 *  @note all the commands above use this file as the entry point.
 */
if (import.meta.main) {
  main().catch(console.error)
}
