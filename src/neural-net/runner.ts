/** @file runner.ts */

import type {
  Game,
  GameStatusRunning,
  GameStatusCompleted,
  GameStatusFailed,
  PersonAttributesScenario2,
  BerghainBouncer,
} from '../types'

import { NeuralNetBouncer } from './neural-net-bouncer'
import { SelfPlayTrainer } from './training'
import * as fs from 'fs'
import * as path from 'path'

export class NeuralNetBouncerRunner {
  private bouncer: NeuralNetBouncer | null = null
  private game: Game | null = null
  private weightsPath: string
  private logPath: string

  constructor(private dataDir: string = '../training', private scenario: '2' = '2') {
    if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true })
    this.weightsPath = path.join(dataDir, `weights-scenario-${scenario}.json`)
    this.logPath = path.join(dataDir, `training-log-${scenario}.json`)
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

  async train(epochs: number = 20, episodesPerEpoch: number = 50): Promise<void> {
    console.log('=== Neural Network Bouncer Training ===\n')
    this.game = this.initializeGame()

    const trainer = new SelfPlayTrainer(this.game, {
      episodes: episodesPerEpoch,
      batchSize: 32,
      learningRate: 0.001,
      explorationStart: 0.9,
      explorationEnd: 0.2,
      explorationDecay: 0.97, // per-epoch
      successThreshold: 5000,
      elitePercentile: 0.2,
    })

    await trainer.train(epochs)

    console.log('\n=== Testing Trained Network ===\n')
    const testResults = trainer.test(100)
    console.log('Test Results (100 episodes):')
    console.log(`  Success Rate: ${(testResults.successRate * 100).toFixed(1)}%`)
    console.log(`  Average Rejections: ${testResults.avgRejections.toFixed(0)}`)
    console.log(`  Best Performance: ${testResults.minRejections} rejections`)
    console.log(`  Worst Performance: ${testResults.maxRejections} rejections`)

    const weights = trainer.getBestWeights()
    fs.writeFileSync(this.weightsPath, JSON.stringify(weights, null, 2))
    console.log(`\nWeights saved to: ${this.weightsPath}`)

    const stats = {
      timestamp: new Date().toISOString(),
      epochs,
      episodesPerEpoch,
      testResults,
      trainingStats: trainer.getStats(),
    }
    fs.writeFileSync(this.logPath, JSON.stringify(stats, null, 2))
    console.log(`Training log saved to: ${this.logPath}`)

    this.bouncer = new NeuralNetBouncer(this.game, { explorationRate: 0, baseThreshold: 0.5 })
    this.bouncer.setNetwork(trainer.getNetwork())
  }

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

  getBouncer(): BerghainBouncer | null {
    return this.bouncer
  }

  runGame(sampleData?: any[]): any {
    if (!this.bouncer || !this.game) throw new Error('Bouncer not initialized. Call train() or load() first.')

    this.bouncer.reset()

    let admitted = 0
    let rejected = 0
    let personIndex = 0

    const getData = sampleData ? () => sampleData[personIndex++] : () => this.generatePerson(personIndex++)

    while (admitted < 1000 && rejected < 20000 && (!sampleData || personIndex < sampleData.length)) {
      const personData = getData()

      const attributes: PersonAttributesScenario2 = {} as any
      for (const attr of personData) attributes[attr as keyof PersonAttributesScenario2] = true
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        if (!(key in attributes)) attributes[key as keyof PersonAttributesScenario2] = false
      }

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: { personIndex, attributes },
      }

      const admit = this.bouncer.admit(status)
      if (admit) admitted++
      else rejected++

      if (admitted === 1000) {
        const progress = this.bouncer.getProgress()
        const satisfied = progress.constraints.every((c: any) => c.satisfied)
        if (satisfied) {
          const finalStatus: GameStatusCompleted = { status: 'completed', rejectedCount: rejected, nextPerson: null }
          return this.bouncer.getOutput(finalStatus)
        } else {
          const finalStatus: GameStatusFailed = {
            status: 'failed',
            reason: 'Constraints not satisfied',
            nextPerson: null,
          }
          return this.bouncer.getOutput(finalStatus)
        }
      }
    }

    const finalStatus: GameStatusFailed = { status: 'failed', reason: 'Too many rejections', nextPerson: null }
    return this.bouncer.getOutput(finalStatus)
  }

  private generatePerson(index: number): string[] {
    if (!this.game) throw new Error('Game not initialized')

    const attributes: string[] = []
    const stats = this.game.attributeStatistics
    const samples: Record<string, boolean> = {}

    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) samples[attr] = Math.random() < freq

    for (const [attr1, correlations] of Object.entries(stats.correlations)) {
      if (samples[attr1]) {
        for (const [attr2, corr] of Object.entries(correlations)) {
          if (attr1 !== attr2 && Math.abs(corr) > 0.3) {
            const adjustedProb = stats.relativeFrequencies[attr2] * (1 + corr * 0.5)
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

  const args = process.argv.slice(2)
  const command = args[0] || 'help'

  switch (command) {
    case 'train': {
      const epochs = parseInt(args[1] || '20')
      const episodes = parseInt(args[2] || '50')
      await runner.train(epochs, episodes)
      break
    }
    case 'test': {
      if (!runner.load()) {
        console.log('Please train the model first: npm run bouncer train')
        break
      }
      let sampleData = null
      if (args[1]) {
        try {
          const data = fs.readFileSync(args[1], 'utf-8')
          sampleData = JSON.parse(data)
          console.log(`Loaded ${sampleData.length} samples from ${args[1]}`)
        } catch (error) {
          console.error('Error loading sample data:', error)
        }
      }
      console.log('\n=== Running Test Game ===\n')
      const result = runner.runGame(sampleData)
      console.log('Game Result:')
      console.log(`  Status: ${result.status}`)
      console.log(`  Final Rejections: ${result.finalRejections}`)
      console.log(`  Constraints:`)
      result.constraints.forEach((c: any) => {
        const ok = c.satisfied ? '✓' : '✗'
        console.log(`    ${ok} ${c.attribute}: ${c.current}/${c.required}`)
      })
      break
    }
    case 'benchmark': {
      if (!runner.load()) {
        console.log('Please train the model first: npm run bouncer train')
        break
      }
      console.log('\n=== Running Benchmark (10 games) ===\n')
      const results = []
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
      console.log(`  Average Rejections: ${avgRejections.toFixed(0)}`)
      break
    }
    default: {
      console.log('Neural Network Bouncer Runner')
      console.log('\nCommands:')
      console.log('  train [epochs] [episodes]  - Train a new neural network')
      console.log('  test [datafile]            - Run a single test game')
      console.log('  benchmark                  - Run 10 games and show statistics')
    }
  }
}

if (require.main === module) {
  main().catch(console.error)
}
