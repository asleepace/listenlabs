/** @file neural-net.ts */

import { Matrix, activations, gradients } from './matrix'

export interface Layer {
  weights: Matrix
  bias: Matrix
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear'
}

export class NeuralNet {
  /** checks if the given object is a neural net instance. */
  static isNeuralNet(obj: unknown): obj is NeuralNet {
    if (!obj) return false
    return obj instanceof NeuralNet
  }

  private layers: Layer[] = []
  private layerOutputs: Matrix[] = []
  private layerInputs: Matrix[] = []

  private learningRate: number
  private l2Lambda: number

  constructor(learningRate = 0.001, l2Lambda = 0.0001) {
    this.learningRate = learningRate
    this.l2Lambda = l2Lambda
  }

  addLayer(
    inputSize: number,
    outputSize: number,
    activation: Layer['activation'] = 'relu',
    initMethod: 'he' | 'xavier' | 'random' = 'he'
  ): void {
    let weights: Matrix
    switch (initMethod) {
      case 'he':
        weights = Matrix.he(inputSize, outputSize)
        break
      case 'xavier':
        weights = Matrix.xavier(inputSize, outputSize)
        break
      case 'random':
        weights = Matrix.random(inputSize, outputSize, -0.5, 0.5)
        break
    }
    const bias = Matrix.zeros(1, outputSize)

    this.layers.push({ weights, bias, activation })
  }

  /** Forward pass. Returns an array (usually length 1 for a sigmoid head). */
  forward(input: number[] | Matrix): number[] {
    let current = input instanceof Matrix ? input : Matrix.fromArray(input).transpose()
    this.layerInputs = [current.copy()]
    this.layerOutputs = []

    for (let li = 0; li < this.layers.length; li++) {
      const layer = this.layers[li]
      const z = current.dot(layer.weights).add(layer.bias)

      let output: Matrix
      switch (layer.activation) {
        case 'relu':
          output = z.map(activations.relu)
          break
        case 'sigmoid':
          output = z.map(activations.sigmoid)
          break
        case 'tanh':
          output = z.map(activations.tanh)
          break
        case 'linear':
          output = z
          break
      }

      this.layerOutputs.push(output.copy())
      if (li < this.layers.length - 1) {
        this.layerInputs.push(output.copy())
      }
      current = output
    }

    return current.toArray()
  }

  /** Alias for readability in callers that expect `infer(...)`. */
  infer(input: number[] | Matrix): number[] {
    return this.forward(input)
  }

  backward(target: number[] | number, _predicted?: number[]): void {
    if (!this.layerOutputs.length) {
      throw new Error('Must call forward() before backward()')
    }

    const targetMatrix = typeof target === 'number' ? new Matrix(1, 1, [target]) : Matrix.fromArray(target).transpose()
    const outputLayer = this.layers[this.layers.length - 1]
    const output = this.layerOutputs[this.layerOutputs.length - 1]

    // MSE loss gradient dL/dy
    let delta = output.subtract(targetMatrix).scale(2 / output.cols)

    // Output activation gradient
    if (outputLayer.activation === 'sigmoid') {
      delta = delta.hadamard(output.map((y) => gradients.sigmoid(y)))
    } else if (outputLayer.activation === 'tanh') {
      delta = delta.hadamard(output.map((y) => gradients.tanh(y)))
    }
    // (relu for the output is handled in the loop below if present)

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const layerInput = this.layerInputs[i]

      const weightGrad = layerInput.transpose().dot(delta)
      const biasGrad = delta.copy()

      // L2 regularization
      const weightUpdate = weightGrad.add(layer.weights.scale(this.l2Lambda)).scale(this.learningRate)

      layer.weights = layer.weights.subtract(weightUpdate)
      layer.bias = layer.bias.subtract(biasGrad.scale(this.learningRate))

      if (i > 0) {
        delta = delta.dot(layer.weights.transpose())

        const prevLayer = this.layers[i - 1]
        const prevOutput = this.layerOutputs[i - 1]

        switch (prevLayer.activation) {
          case 'relu':
            delta = delta.hadamard(prevOutput.map((x) => gradients.relu(x)))
            break
          case 'sigmoid':
            delta = delta.hadamard(prevOutput.map((y) => gradients.sigmoid(y)))
            break
          case 'tanh':
            delta = delta.hadamard(prevOutput.map((y) => gradients.tanh(y)))
            break
          case 'linear':
            // no-op
            break
        }
      }
    }
  }

  trainBatch(inputs: number[][], targets: number[] | number[][], epochs = 1): number {
    let totalLoss = 0

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0

      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i]

        let target: number[]
        if (Array.isArray(targets[0])) {
          target = (targets as number[][])[i]
        } else {
          target = [(targets as number[])[i]]
        }

        const output = this.forward(input)

        const loss =
          output.reduce((sum, val, idx) => {
            const diff = val - target[idx]
            return sum + diff * diff
          }, 0) / output.length

        epochLoss += loss
        this.backward(target)
      }

      totalLoss = epochLoss / inputs.length
    }

    return totalLoss
  }

  /** Convenience for thresholding callers. */
  predict(input: number[]): { value: number; confidence: number } {
    const output = this.forward(input)
    const value = output[0]
    const confidence = Math.min(1, Math.max(0, Math.abs(value - 0.5) * 2))
    return { value, confidence }
  }

  /** Serialize weights, biases, activations and training hyperparams. */
  toJSON(): any {
    return {
      __kind: 'NeuralNet',
      version: 1,
      layers: this.layers.map((layer) => ({
        weights: Array.from(layer.weights.data),
        weightsShape: [layer.weights.rows, layer.weights.cols],
        bias: Array.from(layer.bias.data),
        biasShape: [layer.bias.rows, layer.bias.cols],
        activation: layer.activation,
      })),
      learningRate: this.learningRate,
      l2Lambda: this.l2Lambda,
    }
  }

  /** Static constructor (kept for backwards compatibility). */
  static fromJSON(json: any): NeuralNet {
    const net = new NeuralNet(json.learningRate ?? 0.001, json.l2Lambda ?? 0.0001)
    net.fromJSON(json)
    return net
  }

  /**
   * NEW: Instance deserializer so code that already holds a net can load into it.
   * Used by trainer/runner via n.fromJSON(...) / n.loadJSON(...) / n.load(...).
   */
  fromJSON(json: any): void {
    if (!json || !Array.isArray(json.layers)) {
      throw new Error('Invalid network JSON: missing layers')
    }
    this.learningRate = json.learningRate ?? this.learningRate
    this.l2Lambda = json.l2Lambda ?? this.l2Lambda

    this.layers = json.layers.map((layerData: any) => {
      const [wr, wc] = layerData.weightsShape
      const [br, bc] = layerData.biasShape
      return {
        weights: new Matrix(wr, wc, layerData.weights),
        bias: new Matrix(br, bc, layerData.bias),
        activation: layerData.activation as Layer['activation'],
      }
    })
    // clear caches
    this.layerInputs = []
    this.layerOutputs = []
  }

  /** Aliases so different call sites (“loadJSON”, “load”) work too. */
  loadJSON(json: any): void {
    this.fromJSON(json)
  }
  load(json: any): void {
    this.fromJSON(json)
  }

  // --- small utilities ---
  getLearningRate(): number {
    return this.learningRate
  }
  setLearningRate(rate: number): void {
    this.learningRate = rate
  }
  getParameterCount(): number {
    return this.layers.reduce((sum, layer) => sum + layer.weights.data.length + layer.bias.data.length, 0)
  }
}

// Clean, input-size-first topology.
export function createBerghainNet(inputSize: number = 17): NeuralNet {
  const net = new NeuralNet(0.001, 0.0001)
  net.addLayer(inputSize, 32, 'relu', 'he')
  net.addLayer(32, 16, 'relu', 'he')
  net.addLayer(16, 1, 'sigmoid', 'xavier')
  return net
}
