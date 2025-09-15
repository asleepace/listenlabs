// src/neural-net/__tests__/neuralnet.or.test.ts
import { describe, it, expect } from 'bun:test'
import { NeuralNet } from '../../neural-net/neural-net'

describe('NeuralNet – tiny OR learning', () => {
  it('learns OR and crosses the 0.5 decision boundary', () => {
    // Single-layer logistic regression is enough for OR
    const net = new NeuralNet(0.05, 0.00001)
    net.addLayer(2, 1, 'sigmoid', 'xavier')

    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]
    const y = [0, 1, 1, 1]

    // train up to N steps, early stop when it’s clearly learned
    let ok = false
    for (let e = 0; e < 1200; e++) {
      net.trainBatch(X, y, 1)
      if (e % 50 === 0) {
        const p = X.map((x) => net.forward(x)[0])
        if (p[0] < 0.5 && p[1] > 0.5 && p[2] > 0.5 && p[3] > 0.5) {
          ok = true
          break
        }
      }
    }
    const p = X.map((x) => net.forward(x)[0])
    expect(p[0]).toBeLessThan(0.5)
    expect(p[1]).toBeGreaterThan(0.5)
    expect(p[2]).toBeGreaterThan(0.5)
    expect(p[3]).toBeGreaterThan(0.5)
    expect(ok).toBe(true)
  })

  it('no NaNs after fwd/bwd', () => {
    const net = new NeuralNet(0.001, 0.0001)
    net.addLayer(3, 5, 'relu', 'he')
    net.addLayer(5, 1, 'sigmoid', 'xavier')
    const x = [0.1, -0.2, 0.3]
    const out = net.forward(x)
    expect(out.length).toBe(1)
    expect(Number.isFinite(out[0])).toBe(true)
    net.backward([1], out)
    const out2 = net.forward(x)
    expect(Number.isFinite(out2[0])).toBe(true)
  })
})
