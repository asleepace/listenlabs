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
import { StateEncoder } from './state-encoder'
import { SelfPlayTrainer } from './training'
import * as fs from 'fs'
import * as path from 'path'

/**
 * Main runner class for the Neural Network Bouncer
 */
export class NeuralNetBouncerRunner {
  private bouncer: NeuralNetBouncer | null = null
  private game: Game | null = null
  private weightsPath: string
  private logPath: string

  constructor(
    private dataDir: string = '../training',
    private scenario: '2' = '2' // We're focusing on scenario 2
  ) {
    // Ensure data directory exists
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true })
    }

    this.weightsPath = path.join(dataDir, `weights-scenario-${scenario}.json`)
    this.logPath = path.join(dataDir, `training-log-${scenario}.json`)
  }

  /**
   * Initialize the game configuration for scenario 2
   */
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
          techno_lover: {
            techno_lover: 1,
            well_connected: -0.4696,
            creative: 0.0946,
            berlin_local: -0.6549,
          },
          well_connected: {
            techno_lover: -0.4696,
            well_connected: 1,
            creative: 0.142,
            berlin_local: 0.5724,
          },
          creative: {
            techno_lover: 0.0946,
            well_connected: 0.142,
            creative: 1,
            berlin_local: 0.1445,
          },
          berlin_local: {
            techno_lover: -0.6549,
            well_connected: 0.5724,
            creative: 0.1445,
            berlin_local: 1,
          },
        },
      },
    }
  }

  /**
   * Train a new neural network bouncer
   */
  async train(epochs: number = 20, episodesPerEpoch: number = 50): Promise<void> {
    console.log('=== Neural Network Bouncer Training ===\n')

    this.game = this.initializeGame()

    // Create trainer with configuration
    const trainer = new SelfPlayTrainer(this.game, {
      episodes: episodesPerEpoch,
      batchSize: 32,
      learningRate: 0.001,
      explorationStart: 0.3,
      explorationEnd: 0.05,
      explorationDecay: 0.995,
      successThreshold: 5000,
      elitePercentile: 0.2,
    })

    // Train the network
    await trainer.train(epochs)

    // Test the trained network
    console.log('\n=== Testing Trained Network ===\n')
    const testResults = trainer.test(100)

    console.log('Test Results (100 episodes):')
    console.log(`  Success Rate: ${(testResults.successRate * 100).toFixed(1)}%`)
    console.log(`  Average Rejections: ${testResults.avgRejections.toFixed(0)}`)
    console.log(`  Best Performance: ${testResults.minRejections} rejections`)
    console.log(`  Worst Performance: ${testResults.maxRejections} rejections`)

    // Save weights
    const weights = trainer.getBestWeights()
    fs.writeFileSync(this.weightsPath, JSON.stringify(weights, null, 2))
    console.log(`\nWeights saved to: ${this.weightsPath}`)

    // Save training stats
    const stats = {
      timestamp: new Date().toISOString(),
      epochs,
      episodesPerEpoch,
      testResults,
      trainingStats: trainer.getStats(),
    }
    fs.writeFileSync(this.logPath, JSON.stringify(stats, null, 2))
    console.log(`Training log saved to: ${this.logPath}`)

    // Create the bouncer with trained weights
    this.bouncer = new NeuralNetBouncer(this.game)
    this.bouncer.setNetwork(trainer.getNetwork())
  }

  /**
   * Load existing weights and create a bouncer
   */
  load(): boolean {
    if (!fs.existsSync(this.weightsPath)) {
      console.log('No saved weights found. Please train first.')
      return false
    }

    try {
      const weights = JSON.parse(fs.readFileSync(this.weightsPath, 'utf-8'))
      this.game = this.initializeGame()

      const expected = new StateEncoder(this.game).getFeatureSize()
      if (weights.layers?.[0]?.weightsShape?.[0] !== expected) {
        throw new Error(
          `Weights expect input ${weights.layers[0].weightsShape[0]} but encoder produces ${expected}. Retrain or update encoder/model.`
        )
      }
      this.bouncer = new NeuralNetBouncer(this.game, {
        explorationRate: 0, // No exploration when using trained model
        baseThreshold: 0.5,
        minThreshold: 0.3,
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

  /**
   * Get the bouncer instance for use in the game runner
   */
  getBouncer(): BerghainBouncer | null {
    return this.bouncer
  }

  /**
   * Run a single game and return results
   */
  runGame(sampleData?: any[]): any {
    if (!this.bouncer || !this.game) {
      throw new Error('Bouncer not initialized. Call train() or load() first.')
    }

    // Reset bouncer state
    this.bouncer.reset()

    let admitted = 0
    let rejected = 0
    let personIndex = 0

    // Use sample data if provided, otherwise generate
    const getData = sampleData ? () => sampleData[personIndex++] : () => this.generatePerson(personIndex++)

    while (admitted < 1000 && rejected < 20000 && (!sampleData || personIndex < sampleData.length)) {
      const personData = getData()

      // Convert to proper format
      const attributes: PersonAttributesScenario2 = {} as any
      for (const attr of personData) {
        attributes[attr as keyof PersonAttributesScenario2] = true
      }
      // Set false for missing attributes
      for (const key of Object.keys(this.game.attributeStatistics.relativeFrequencies)) {
        if (!(key in attributes)) {
          attributes[key as keyof PersonAttributesScenario2] = false
        }
      }

      const status: GameStatusRunning<PersonAttributesScenario2> = {
        status: 'running',
        admittedCount: admitted,
        rejectedCount: rejected,
        nextPerson: {
          personIndex,
          attributes,
        },
      }

      const admit = this.bouncer.admit(status)

      if (admit) {
        admitted++
      } else {
        rejected++
      }

      // Check if we've reached capacity
      if (admitted === 1000) {
        const progress = this.bouncer.getProgress()
        const satisfied = progress.constraints.every((c: any) => c.satisfied)

        if (satisfied) {
          const finalStatus: GameStatusCompleted = {
            status: 'completed',
            rejectedCount: rejected,
            nextPerson: null,
          }
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

    // Too many rejections
    const finalStatus: GameStatusFailed = {
      status: 'failed',
      reason: 'Too many rejections',
      nextPerson: null,
    }
    return this.bouncer.getOutput(finalStatus)
  }

  /**
   * Generate a synthetic person based on game statistics
   */
  private generatePerson(index: number): string[] {
    if (!this.game) throw new Error('Game not initialized')

    const attributes: string[] = []
    const stats = this.game.attributeStatistics

    // Sample based on frequencies and correlations
    const samples: Record<string, boolean> = {}

    for (const [attr, freq] of Object.entries(stats.relativeFrequencies)) {
      samples[attr] = Math.random() < freq
    }

    // Apply correlations
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

    // Convert to array format
    for (const [attr, hasAttr] of Object.entries(samples)) {
      if (hasAttr) {
        attributes.push(attr)
      }
    }

    return attributes
  }
}

/**
 * Main execution function
 */
export async function main() {
  const runner = new NeuralNetBouncerRunner('./bouncer-data', '2')

  // Parse command line arguments
  const args = process.argv.slice(2)
  const command = args[0] || 'help'

  switch (command) {
    case 'train':
      const epochs = parseInt(args[1] || '20')
      const episodes = parseInt(args[2] || '50')
      await runner.train(epochs, episodes)
      break

    case 'test':
      if (!runner.load()) {
        console.log('Please train the model first: npm run bouncer train')
        break
      }

      // Load sample data if provided
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
        const status = c.satisfied ? '✓' : '✗'
        console.log(`    ${status} ${c.attribute}: ${c.current}/${c.required}`)
      })
      break

    case 'benchmark':
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

    default:
      console.log('Neural Network Bouncer Runner')
      console.log('\nCommands:')
      console.log('  train [epochs] [episodes]  - Train a new neural network')
      console.log('  test [datafile]            - Run a single test game')
      console.log('  benchmark                  - Run 10 games and show statistics')
      console.log('\nExample:')
      console.log('  npm run bouncer train 20 50')
      console.log('  npm run bouncer test sample-data.json')
      console.log('  npm run bouncer benchmark')
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error)
}
