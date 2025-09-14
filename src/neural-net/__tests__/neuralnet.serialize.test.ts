import { describe, it, expect } from 'bun:test'
import { NeuralNet, createBerghainNet } from '../../neural-net/neural-net'

describe('NeuralNet – toJSON/fromJSON roundtrip', () => {
  it('preserves weights and outputs', () => {
    const net1 = createBerghainNet(17)
    const x = Array(17)
      .fill(0)
      .map((_, i) => (i % 3 === 0 ? 0.5 : -0.1))
    const y1 = net1.forward(x)[0]

    const json = net1.toJSON()
    const net2 = NeuralNet.fromJSON(json)
    const y2 = net2.forward(x)[0]

    expect(Math.abs(y1 - y2)).toBeLessThan(1e-7)
  })
})
