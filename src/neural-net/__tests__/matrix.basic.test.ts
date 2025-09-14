import { describe, it, expect } from 'bun:test'
import { Matrix } from '../../neural-net/matrix'

describe('Matrix – basics', () => {
  it('add/subtract/hadamard/scale', () => {
    const a = Matrix.from2D([
      [1, 2],
      [3, 4],
    ])
    const b = Matrix.from2D([
      [5, 6],
      [7, 8],
    ])
    expect(a.add(b).toArray()).toEqual([6, 8, 10, 12])
    expect(b.subtract(a).toArray()).toEqual([4, 4, 4, 4])
    expect(a.hadamard(b).toArray()).toEqual([5, 12, 21, 32])
    expect(a.scale(2).toArray()).toEqual([2, 4, 6, 8])
  })

  it('dot & transpose', () => {
    const a = Matrix.from2D([
      [1, 2, 3],
      [4, 5, 6],
    ]) // 2x3
    const b = Matrix.from2D([
      [7, 8],
      [9, 10],
      [11, 12],
    ]) // 3x2
    const c = a.dot(b) // 2x2
    expect(c.rows).toBe(2)
    expect(c.cols).toBe(2)
    expect(c.toArray().map((n) => +n.toFixed(5))).toEqual([58, 64, 139, 154])
    const t = a.transpose()
    expect(t.rows).toBe(3)
    expect(t.cols).toBe(2)
    expect(t.get(1, 0)).toBe(2)
    expect(t.get(2, 1)).toBe(6)
  })

  it('zeros/ones/identity', () => {
    const z = Matrix.zeros(2, 3)
    expect(z.toArray()).toEqual([0, 0, 0, 0, 0, 0])
    const o = Matrix.ones(2, 2)
    expect(o.toArray()).toEqual([1, 1, 1, 1])
    const I = Matrix.identity(3)
    expect(I.toArray()).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1])
  })
})
