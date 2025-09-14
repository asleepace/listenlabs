/** @file runner.ts */

import type { Game, GameStatusRunning, PersonAttributesScenario2, BerghainBouncer, ScenarioAttributes } from '../types'

import { NeuralNetBouncer } from './neural-net-bouncer'
import { SelfPlayTrainer } from './training'
import { initializeScoring } from './scoring'
import * as fs from 'fs'
import * as path from 'path'
import { createBerghainNet, NeuralNet } from './neural-net'

const FEATURE_SIZE = 17 // encoder feature size for scenario 2
const MAX_ADMISSIONS = 1_000
const MAX_REJECTIONS = 20_000
const TARGET_REJECTIONS = 5_000 // explicit, though scoring has a default

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

function parseFlags(args: string[]) {
  const flags: Record<string, string> = {}
  for (const a of args) {
    if (a.startsWith('--')) {
      const i = a.indexOf('=')
      if (i > 2) flags[a.slice(2, i)] = a.slice(i + 1)
      else flags[a.slice(2)] = 'true'
    }
  }
  return flags
}

export class NeuralNetBouncerRunner {
  private bouncer: NeuralNetBouncer | null = null
  private game: Game | null = null
  private weightsPath: string
  private logPath: string

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
      } catch {}
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
    const resumeFlag = flags.resume === 'true' // also supported via separate "resume" command

    const trainer = new SelfPlayTrainer(this.game, {
      episodes: episodesPerEpoch,
      batchSize: 32,
      learningRate: 0.001,
      explorationStart: 0.9,
      explorationEnd: 0.2,
      explorationDecay: 0.97,
      successThreshold: 5000,
      elitePercentile,
      assistGain,
      oracleRelabelFrac,
    })

    if (resumeFlag && fs.existsSync(this.weightsPath)) {
      try {
        const weights = JSON.parse(fs.readFileSync(this.weightsPath, 'utf-8'))
        trainer.loadWeights(weights)
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

    console.log('\n=== Testing Trained Network ===\n')
    const testResults = trainer.test(100)
    console.log('Test Results (100 episodes):')
    console.log(`  Success Rate: ${(testResults.successRate * 100).toFixed(1)}%`)
    console.log(`  Average Rejections: ${testResults.avgRejections.toFixed(0)}`)
    console.log(`  Best Performance: ${testResults.minRejections} rejections`)
    console.log(`  Worst Performance: ${testResults.maxRejections} rejections`)

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

    this.bouncer = new NeuralNetBouncer(this.game, { explorationRate: 0, baseThreshold: 0.5 })
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
        baseThreshold: 0.35,
        minThreshold: 0.25,
        maxThreshold: 0.7,
        urgencyFactor: 2.0,
      })

      // Prefer static loader; fall back to instance methods if present.
      let net: NeuralNet
      try {
        net = NeuralNet.fromJSON(weights)
      } catch {
        net = createBerghainNet(FEATURE_SIZE)
        ;(net as any).fromJSON?.(weights) || (net as any).loadJSON?.(weights) || (net as any).load?.(weights)
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
    const remaining = Math.max(0, MAX_ADMISSIONS - admitted)
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
    while (chosen.length < MAX_ADMISSIONS && pool.length > 0) {
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

  runGame(sampleData?: any[], mode: 'score' | 'bouncer' | 'hybrid' = 'score'): any {
    if (!this.game) throw new Error('Game not initialized')

    let admitted = 0
    let rejected = 0
    let personIndex = 0

    const getData = sampleData ? () => sampleData[personIndex++] : () => this.generatePerson(personIndex++)

    const scoring = initializeScoring(this.game, {
      maxRejections: MAX_REJECTIONS,
      maxAdmissions: MAX_ADMISSIONS,
      targetRejections: TARGET_REJECTIONS,
    })

    while (scoring.inProgress() && (!sampleData || personIndex < sampleData.length)) {
      const personData = getData()

      // normalize
      const attributes: PersonAttributesScenario2 = {} as any
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        attributes[key as keyof PersonAttributesScenario2] = false
      }
      if (Array.isArray(personData)) {
        for (const a of personData) if (a in attributes) attributes[a as keyof PersonAttributesScenario2] = true
      } else if (personData && typeof personData === 'object') {
        for (const [k, v] of Object.entries(personData))
          if (k in attributes) attributes[k as keyof PersonAttributesScenario2] = !!v
      }

      const guest = attributes as ScenarioAttributes
      const quotasCompleted = scoring.isFinishedWithQuotas()

      // --- pick decision source ---
      let admit: boolean = quotasCompleted

      if (quotasCompleted && admitted < 1000) {
        // Seats left and quotas are complete -> admit everyone to finish with zero extra rejections
        admit = true
      } else if (mode === 'bouncer') {
        if (!this.bouncer) throw new Error('Bouncer not initialized')
        const status: GameStatusRunning<PersonAttributesScenario2> = {
          status: 'running',
          admittedCount: admitted,
          rejectedCount: rejected,
          nextPerson: { personIndex, attributes },
        }
        admit = this.bouncer.admit(status)
      } else if (mode === 'hybrid') {
        if (!this.bouncer) throw new Error('Bouncer not initialized')
        const status: GameStatusRunning<PersonAttributesScenario2> = {
          status: 'running',
          admittedCount: admitted,
          rejectedCount: rejected,
          nextPerson: { personIndex, attributes },
        }

        // --- FINISHER: if any quota needs just 1 head, only admit if guest hits it ---
        const critical = scoring
          .quotas() // unmet only
          .map((q) => ({ q, need: q.needed() }))
          .sort((a, b) => a.need - b.need)[0]

        if (critical && critical.need <= 1) {
          admit = !!guest[critical.q.attribute]
        } else {
          // regular hybrid gating
          const nnAdmit = this.bouncer.admit(status)
          const policyAdmit = scoring.shouldAdmit(guest, 1.0, 0.5)
          const quotasOutstanding = scoring.quotas().length > 0
          if (quotasOutstanding) {
            admit = policyAdmit && nnAdmit
            // Optional soft guard: block low-urgency slip-throughs
            if (!policyAdmit && admit) {
              const minimalTheta = 1.0 + 0.5 * scoring.seatScarcity()
              if (scoring.guestScore(guest) < minimalTheta) admit = false
            }
          } else {
            admit = nnAdmit || policyAdmit
          }
        }
      } else {
        // score-only
        admit = scoring.shouldAdmit(guest, 1.0, 0.5)
      }

      if (admit) admitted++
      else rejected++

      scoring.update({ guest, admit })
      if (admitted >= MAX_ADMISSIONS || rejected >= MAX_REJECTIONS) break
    }

    // summarize
    const constraints = this.game.constraints.map((c) => {
      const q = scoring.get(c.attribute)
      const current = q?.count ?? 0
      return { attribute: c.attribute, current, required: c.minCount, satisfied: current >= c.minCount }
    })
    const allMet = constraints.every((c) => c.satisfied)

    if (admitted >= MAX_ADMISSIONS && allMet) {
      return { status: 'completed', finalRejections: rejected, constraints }
    }
    const hitRejectionCap = rejected >= MAX_REJECTIONS
    return {
      status: 'failed',
      reason: hitRejectionCap ? 'Too many rejections' : 'Constraints not satisfied',
      finalRejections: rejected,
      constraints,
    }
  }

  /** Sample a synthetic person as a list of attribute names set to true. */
  private generatePerson(index: number): string[] {
    if (!this.game) throw new Error('Game not initialized')

    const attributes: string[] = []
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    // base samples
    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < clamp(freq, 0, 1)
    }

    // apply correlations (positive -> boost, negative -> reduce)
    for (const [attr1, correlations] of Object.entries(stats.correlations)) {
      if (samples[attr1]) {
        for (const [attr2, corr] of Object.entries(correlations)) {
          if (attr1 !== attr2 && Math.abs(corr) > 0.3) {
            const base = stats.relativeFrequencies[attr2] ?? 0
            const adjustedProb = clamp(base * (1 + corr * 0.5), 0, 1)
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
      console.log(`  Final Rejections: ${result.finalRejections}`)
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
      for (let i = 0; i < 10; i++) {
        const result = runner.runGame(undefined, mode)
        results.push(result)
        if (result.status === 'completed') {
          successes++
          console.log(`Game ${i + 1}: SUCCESS - ${result.finalRejections} rejections`)
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
      console.log(`  Success Rate: ${((successes / 10) * 100).toFixed(0)}%`)
      console.log(`  Average Rejections (successful only): ${avgRejections.toFixed(0)}`)
      break
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
      console.log('  diagnose                                         - Greedy feasibility check over a sampled pool')
    }
  }
}

if (require.main === module) {
  main().catch(console.error)
}
