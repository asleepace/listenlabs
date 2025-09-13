/** @file neural-net.ts */

import { Matrix, activations, gradients } from './matrix'

export interface Layer {
  weights: Matrix
  bias: Matrix
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear'
}

export class NeuralNet {
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
      default:
        weights = Matrix.random(inputSize, outputSize, -0.5, 0.5)
    }
    const bias = Matrix.zeros(1, outputSize)
    this.layers.push({ weights, bias, activation })
  }

  forward(input: number[] | Matrix): number[] {
    let current = input instanceof Matrix ? input : Matrix.fromArray(input).transpose()

    // Guard: shape must match first layer
    if (this.layers.length > 0 && current.cols !== this.layers[0].weights.rows) {
      throw new Error(
        `Shape mismatch: input has ${current.cols} features, layer expects ${this.layers[0].weights.rows}`
      )
    }

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
        default:
          output = z
      }

      this.layerOutputs.push(output.copy())
      if (li < this.layers.length - 1) this.layerInputs.push(output.copy())
      current = output
    }

    return current.toArray()
  }

  backward(target: number[] | number): void {
    if (!this.layerOutputs.length) throw new Error('Must call forward() before backward()')

    const targetMatrix = typeof target === 'number' ? new Matrix(1, 1, [target]) : Matrix.fromArray(target).transpose()
    const outputLayer = this.layers[this.layers.length - 1]
    const output = this.layerOutputs[this.layerOutputs.length - 1]

    let delta = output.subtract(targetMatrix).scale(2 / output.cols)

    if (outputLayer.activation === 'sigmoid') {
      delta = delta.hadamard(output.map((y) => gradients.sigmoid(y)))
    } else if (outputLayer.activation === 'tanh') {
      delta = delta.hadamard(output.map((y) => gradients.tanh(y)))
    }

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const layerInput = this.layerInputs[i]

      const weightGrad = layerInput.transpose().dot(delta)
      const biasGrad = delta.copy()

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
        const target: number[] = Array.isArray(targets[0]) ? (targets as number[][])[i] : [(targets as number[])[i]]

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

  predict(input: number[]): { value: number; confidence: number } {
    const output = this.forward(input)
    const value = output[0]
    const confidence = Math.abs(value - 0.5) * 2
    return { value, confidence }
  }

  toJSON(): any {
    return {
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

  static fromJSON(json: any): NeuralNet {
    const net = new NeuralNet(json.learningRate, json.l2Lambda)
    net.layers = json.layers.map((ld: any) => ({
      weights: new Matrix(ld.weightsShape[0], ld.weightsShape[1], ld.weights),
      bias: new Matrix(ld.biasShape[0], ld.biasShape[1], ld.bias),
      activation: ld.activation,
    }))
    return net
  }

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

// Creates a net matching an encoder-provided input size
export function createBerghainNet(inputSize: number = 17): NeuralNet {
  const net = new NeuralNet(0.001, 0.0001)
  net.addLayer(inputSize, 32, 'relu', 'he') // first hidden
  net.addLayer(32, 16, 'relu', 'he') // second hidden
  net.addLayer(16, 1, 'sigmoid', 'xavier') // output
  return net
}
