/** @file runner.ts */

import type {
  Game,
  GameStatusRunning,
  GameStatusCompleted,
  GameStatusFailed,
  PersonAttributesScenario2,
  BerghainBouncer,
  ScenarioAttributes,
} from '../types'

import { NeuralNetBouncer } from './neural-net-bouncer'
import { SelfPlayTrainer } from './training'
import { initializeScoring } from './scoring' // adjust path
import * as fs from 'fs'
import * as path from 'path'

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
    if (!log?.runs) {
      log.runs = []
    }
    console.log(log.runs)
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

  // ---------- load / diagnose / runGame (unchanged except for small guards) ----------
  load(): boolean {
    if (!fs.existsSync(this.weightsPath)) {
      console.log('No saved weights found. Please train first.')
      return false
    }
    try {
      const weights = JSON.parse(fs.readFileSync(this.weightsPath, 'utf-8'))
      this.game = this.initializeGame()
      this.bouncer = new NeuralNetBouncer(this.game, {
        explorationRate: 0,
        baseThreshold: 0.35,
        minThreshold: 0.25,
        maxThreshold: 0.7,
        urgencyFactor: 2.0,
      })
      this.bouncer.loadWeights(weights)
      console.log('Weights loaded successfully!')
      return true
    } catch (error) {
      console.error('Error loading weights:', error)
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
    const remaining = Math.max(0, 1000 - admitted)
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
    while (chosen.length < 1000 && pool.length > 0) {
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

  runGame(sampleData?: any[]): any {
    if (!this.game) throw new Error('Game not initialized')
    // We won't rely on this.bouncer’s internal tracker since we decide via scoring.
    // (You can still load it for other commands.)

    let admitted = 0
    let rejected = 0
    let personIndex = 0

    const getData = sampleData ? () => sampleData[personIndex++] : () => this.generatePerson(personIndex++)

    // Seat/line model for decisions
    const scoring = initializeScoring(this.game, {
      maxRejections: 20_000,
      maxAdmissions: 1_000,
      targetRejections: 5_000,
      weights: {
        // .. override here
      },
    })

    while (scoring.inProgress() && (!sampleData || personIndex < sampleData.length)) {
      const personData = getData()

      // --- normalize into a full attributes object of booleans ---
      const attributes: PersonAttributesScenario2 = {} as any
      // start all attrs as false
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        attributes[key as keyof PersonAttributesScenario2] = false
      }
      if (Array.isArray(personData)) {
        for (const a of personData) {
          if (a in attributes) attributes[a as keyof PersonAttributesScenario2] = true
        }
      } else if (personData && typeof personData === 'object') {
        for (const [k, v] of Object.entries(personData)) {
          if (k in attributes) attributes[k as keyof PersonAttributesScenario2] = !!v
        }
      }

      // --- decision (your rule-of-thumb combo) ---
      const guest = attributes as ScenarioAttributes
      const admit = scoring.shouldAdmit(guest, /*baseTheta=*/ 1.0, /*baseFrac=*/ 0.5)

      // keep global counters for logging and termination
      if (admit) admitted++
      else rejected++

      // keep the scorer in sync (it maintains per-quota counts)
      scoring.update({ guest, admit })

      // optional early exit: if all quotas met and room is full, we can stop
      if (admitted >= 1000 || scoring.isFinishedWithQuotas()) break
    }

    // --- summarize constraints from scoring’s quota counts ---
    const constraints = this.game.constraints.map((c) => {
      const q = scoring.get(c.attribute)
      const current = q?.count ?? 0
      return { attribute: c.attribute, current, required: c.minCount, satisfied: current >= c.minCount }
    })
    const allMet = constraints.every((c) => c.satisfied)

    const totalScore = scoring.getLossScore()
    console.log(`  Score (diagnostic): ${totalScore.toFixed(2)}`)

    // Decide final status
    if (admitted >= 1000 && allMet) {
      const result = {
        status: 'completed',
        finalRejections: rejected,
        constraints,
      }
      return result
    }

    const hitRejectionCap = rejected >= 20000
    const result = {
      status: 'failed',
      reason: hitRejectionCap ? 'Too many rejections' : 'Constraints not satisfied',
      finalRejections: rejected,
      constraints,
    }
    return result
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

    // pack true attrs
    for (const [attr, hasAttr] of Object.entries(samples)) if (hasAttr) attributes.push(attr)
    return attributes
  }
}

export async function main() {
  const runner = new NeuralNetBouncerRunner('./bouncer-data', '2')

  const args = process.argv.slice(2)
  const command = args[0] || 'help'

  switch (command) {
    case 'train': {
      const epochs = parseInt(args[1] || '20', 10)
      const episodes = parseInt(args[2] || '50', 10)
      const flags = parseFlags(args.slice(3))
      await runner.train(epochs, episodes, flags)
      break
    }
    case 'resume': {
      // <-- NEW
      const epochs = parseInt(args[1] || '10', 10)
      const episodes = parseInt(args[2] || '50', 10)
      const flags = parseFlags(args.slice(3))
      await runner.resume(epochs, episodes, flags)
      break
    }
    case 'test': {
      if (!runner.load()) {
        console.log('Please train the model first: bun run src/neural-net/runner train 20')
        break
      }
      let sampleData: any[] | null = null
      if (args[1]) {
        try {
          const data = fs.readFileSync(args[1], 'utf-8')
          sampleData = JSON.parse(data)
          console.log(`Loaded ${sampleData?.length} samples from ${args[1]}`)
        } catch (error) {
          console.error('Error loading sample data:', (error as Error).message)
        }
      }
      console.log('\n=== Running Test Game ===\n')
      const result = runner.runGame(sampleData || undefined)
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
      const results: any[] = []
      let successes = 0
      for (let i = 0; i < 10; i++) {
        const result = runner.runGame()
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
      console.log('  resume [epochs] [episodes]  - Continue training from saved weights')
      console.log('  test [datafile]             - Run a single test game')
      console.log('  benchmark                   - Run 10 games and show statistics')
      console.log('  diagnose                    - Greedy feasibility check over a sampled pool')
    }
  }
}

if (require.main === module) {
  main().catch(console.error)
}
