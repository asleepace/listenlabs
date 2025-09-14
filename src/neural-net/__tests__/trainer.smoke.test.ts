import { describe, it, expect } from 'bun:test'
import { SelfPlayTrainer } from '../../neural-net/training'

const game = {
  gameId: 'scenario-2-test',
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
} as any

describe('SelfPlayTrainer – smoke', () => {
  it('runs a very tiny train loop', async () => {
    const trainer = new SelfPlayTrainer(game, {
      episodes: 4, // tiny
      batchSize: 8,
      learningRate: 0.0005,
      explorationStart: 0.5,
      explorationEnd: 0.2,
      explorationDecay: 0.99,
      elitePercentile: 0.2,
      assistGain: 2.5,
      oracleRelabelFrac: 0.5,
      successThreshold: 5000,
    })
    await trainer.train(1) // 1 epoch
    const res = trainer.test(5)
    expect(Number.isFinite(res.avgAdmissions)).toBe(true)
    expect(Number.isFinite(res.avgRejections)).toBe(true)
    expect(res.minRejections).toBeGreaterThanOrEqual(0)
  }, 20_000)
})
