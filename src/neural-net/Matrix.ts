/** @file matrix.ts */

export class Matrix {
  readonly rows: number
  readonly cols: number
  readonly data: Float32Array

  constructor(rows: number, cols: number, data?: number[] | Float32Array) {
    this.rows = rows
    this.cols = cols

    if (data) {
      if (data.length !== rows * cols) {
        throw new Error(`Data length ${data.length} doesn't match dimensions ${rows}x${cols}`)
      }
      this.data = new Float32Array(data)
    } else {
      this.data = new Float32Array(rows * cols)
    }
  }

  get(row: number, col: number): number {
    return this.data[row * this.cols + col]
  }

  set(row: number, col: number, value: number): void {
    this.data[row * this.cols + col] = value
  }

  dot(other: Matrix): Matrix {
    if (this.cols !== other.rows) {
      console.log({ ...this, other })
      throw new Error(`Cannot multiply ${this.rows}x${this.cols} with ${other.rows}x${other.cols}`)
    }

    const result = new Matrix(this.rows, other.cols)

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0
        for (let k = 0; k < this.cols; k++) {
          sum += this.get(i, k) * other.get(k, j)
        }
        result.set(i, j, sum)
      }
    }

    return result
  }

  add(other: Matrix | number): Matrix {
    const result = new Matrix(this.rows, this.cols)

    if (typeof other === 'number') {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] + other
    } else {
      if (this.rows !== other.rows || this.cols !== other.cols)
        throw new Error('Matrices must have same dimensions for addition')
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] + other.data[i]
    }

    return result
  }

  subtract(other: Matrix | number): Matrix {
    const result = new Matrix(this.rows, this.cols)

    if (typeof other === 'number') {
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] - other
    } else {
      if (this.rows !== other.rows || this.cols !== other.cols)
        throw new Error('Matrices must have same dimensions for subtraction')
      for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] - other.data[i]
    }

    return result
  }

  hadamard(other: Matrix): Matrix {
    if (this.rows !== other.rows || this.cols !== other.cols)
      throw new Error('Matrices must have same dimensions for Hadamard product')
    const result = new Matrix(this.rows, this.cols)
    for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] * other.data[i]
    return result
  }

  scale(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols)
    for (let i = 0; i < this.data.length; i++) result.data[i] = this.data[i] * scalar
    return result
  }

  transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows)
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) result.set(j, i, this.get(i, j))
    return result
  }

  map(fn: (value: number, index: number) => number): Matrix {
    const result = new Matrix(this.rows, this.cols)
    for (let i = 0; i < this.data.length; i++) result.data[i] = fn(this.data[i], i)
    return result
  }

  sum(): number {
    let sum = 0
    for (let i = 0; i < this.data.length; i++) sum += this.data[i]
    return sum
  }

  max(): number {
    return Math.max(...this.data)
  }

  min(): number {
    return Math.min(...this.data)
  }

  copy(): Matrix {
    return new Matrix(this.rows, this.cols, this.data)
  }

  toArray(): number[] {
    return Array.from(this.data)
  }

  toString(): string {
    let str = ''
    for (let i = 0; i < this.rows; i++) {
      const row = []
      for (let j = 0; j < this.cols; j++) row.push(this.get(i, j).toFixed(4))
      str += `[${row.join(', ')}]\n`
    }
    return str
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols)
  }

  static ones(rows: number, cols: number): Matrix {
    const m = new Matrix(rows, cols)
    m.data.fill(1)
    return m
  }

  static random(rows: number, cols: number, min = -1, max = 1): Matrix {
    const m = new Matrix(rows, cols)
    const range = max - min
    for (let i = 0; i < m.data.length; i++) m.data[i] = Math.random() * range + min
    return m
  }

  static xavier(rows: number, cols: number): Matrix {
    const limit = Math.sqrt(6 / (rows + cols))
    return Matrix.random(rows, cols, -limit, limit)
  }

  static he(rows: number, cols: number): Matrix {
    const std = Math.sqrt(2 / rows)
    const m = new Matrix(rows, cols)
    for (let i = 0; i < m.data.length; i++) {
      const u1 = Math.random()
      const u2 = Math.random()
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
      m.data[i] = z0 * std
    }
    return m
  }

  static identity(size: number): Matrix {
    const m = new Matrix(size, size)
    for (let i = 0; i < size; i++) m.set(i, i, 1)
    return m
  }

  static fromArray(arr: number[]): Matrix {
    return new Matrix(arr.length, 1, arr)
  }

  static from2D(arr: number[][]): Matrix {
    const rows = arr.length
    const cols = arr[0].length
    const data = new Float32Array(rows * cols)
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) data[i * cols + j] = arr[i][j]
    return new Matrix(rows, cols, data)
  }
}

export const activations = {
  relu: (x: number) => Math.max(0, x),
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  tanh: (x: number) => Math.tanh(x),
  leakyRelu: (x: number, alpha = 0.01) => (x > 0 ? x : alpha * x),
}

export const gradients = {
  relu: (x: number) => (x > 0 ? 1 : 0),
  sigmoid: (y: number) => y * (1 - y),
  tanh: (y: number) => 1 - y * y,
  leakyRelu: (x: number, alpha = 0.01) => (x > 0 ? 1 : alpha),
}

export function applyActivation(m: Matrix, activation: (x: number) => number): Matrix {
  return m.map(activation)
}
