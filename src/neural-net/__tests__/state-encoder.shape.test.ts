import { describe, it, expect } from 'bun:test'
import { StateEncoder } from '../../neural-net/state-encoder'

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
}

describe('StateEncoder – feature size & encode', () => {
  it('feature size is 17 and encode returns 17 numbers', () => {
    const enc = new StateEncoder(game as any)
    expect(enc.getFeatureSize()).toBe(17)

    const status = {
      status: 'running',
      admittedCount: 0,
      rejectedCount: 0,
      nextPerson: {
        personIndex: 1,
        attributes: { techno_lover: false, well_connected: false, creative: false, berlin_local: false },
      },
    } as any

    const counts = { techno_lover: 0, well_connected: 0, creative: 0, berlin_local: 0 }
    const vec = enc.encode(status, counts)
    expect(vec.length).toBe(17)
    expect(vec.every(Number.isFinite)).toBe(true)
  })
})
