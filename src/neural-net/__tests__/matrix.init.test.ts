import { describe, it, expect } from 'bun:test'
import { Matrix } from '../../neural-net/matrix'

function stats(arr: number[]) {
  const n = arr.length
  const mean = arr.reduce((s, x) => s + x, 0) / n
  const var_ = arr.reduce((s, x) => s + (x - mean) * (x - mean), 0) / Math.max(1, n - 1)
  return { mean, std: Math.sqrt(var_) }
}

describe('Matrix – initializers', () => {
  it('xavier is finite-ish with near-zero mean', () => {
    const m = Matrix.xavier(64, 32)
    const { mean, std } = stats(m.toArray())
    expect(Number.isFinite(mean)).toBe(true)
    expect(Number.isFinite(std)).toBe(true)
    expect(Math.abs(mean)).toBeLessThan(0.1)
  })

  it('he std roughly matches sqrt(2/fan_in)', () => {
    const fanIn = 64
    const m = Matrix.he(fanIn, 32)
    const { mean, std } = stats(m.toArray())
    const target = Math.sqrt(2 / fanIn)
    // Loose bounds due to randomness
    expect(Math.abs(mean)).toBeLessThan(0.15)
    expect(std).toBeGreaterThan(0.4 * target)
    expect(std).toBeLessThan(2.5 * target)
  })
})
