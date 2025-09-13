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

  // Learning rate and regularization
  private learningRate: number
  private l2Lambda: number

  constructor(learningRate = 0.001, l2Lambda = 0.0001) {
    this.learningRate = learningRate
    this.l2Lambda = l2Lambda
  }

  // Add a layer to the network
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

    this.layers.push({
      weights,
      bias,
      activation,
    })
  }

  // Forward propagation
  forward(input: number[] | Matrix): number[] {
    let current = input instanceof Matrix ? input : Matrix.fromArray(input).transpose()

    // Store for backprop
    this.layerInputs = [current.copy()]
    this.layerOutputs = []

    for (const layer of this.layers) {
      if (current.cols !== layer.weights.rows) {
        throw new Error(`Shape mismatch: input has ${current.cols} features, but layer expects ${layer.weights.rows}.`)
      }

      // Linear transformation: z = x * W + b
      const z = current.dot(layer.weights).add(layer.bias)

      // Apply activation
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
      if (this.layers.indexOf(layer) < this.layers.length - 1) {
        this.layerInputs.push(output.copy())
      }

      current = output
    }

    return current.toArray()
  }

  // Backward propagation
  backward(target: number[] | number, predicted?: number[]): void {
    if (!this.layerOutputs.length) {
      throw new Error('Must call forward() before backward()')
    }

    // Convert target to matrix
    const targetMatrix = typeof target === 'number' ? new Matrix(1, 1, [target]) : Matrix.fromArray(target).transpose()

    // Start with output layer error
    const outputLayer = this.layers[this.layers.length - 1]
    const output = this.layerOutputs[this.layerOutputs.length - 1]

    // Calculate initial gradient (assuming MSE loss)
    let delta = output.subtract(targetMatrix).scale(2 / output.cols)

    // Apply output activation gradient
    if (outputLayer.activation === 'sigmoid') {
      delta = delta.hadamard(output.map((y) => gradients.sigmoid(y)))
    } else if (outputLayer.activation === 'tanh') {
      delta = delta.hadamard(output.map((y) => gradients.tanh(y)))
    }

    // Backpropagate through layers
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const layerInput = this.layerInputs[i]

      // Calculate gradients
      const weightGrad = layerInput.transpose().dot(delta)
      const biasGrad = delta.copy()

      // Update weights with L2 regularization
      const weightUpdate = weightGrad.add(layer.weights.scale(this.l2Lambda)).scale(this.learningRate)

      layer.weights = layer.weights.subtract(weightUpdate)
      layer.bias = layer.bias.subtract(biasGrad.scale(this.learningRate))

      // Propagate error to previous layer
      if (i > 0) {
        delta = delta.dot(layer.weights.transpose())

        // Apply activation gradient of previous layer
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

  // Train on a batch of examples
  trainBatch(inputs: number[][], targets: number[] | number[][], epochs = 1): number {
    let totalLoss = 0

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0

      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i]

        // Handle both single values and arrays for targets
        let target: number[]
        if (Array.isArray(targets[0])) {
          // targets is number[][]
          target = (targets as number[][])[i]
        } else {
          // targets is number[]
          target = [(targets as number[])[i]]
        }

        // Forward pass
        const output = this.forward(input)

        // Calculate loss (MSE)
        const loss =
          output.reduce((sum, val, idx) => {
            const diff = val - target[idx]
            return sum + diff * diff
          }, 0) / output.length

        epochLoss += loss

        // Backward pass
        this.backward(target)
      }

      totalLoss = epochLoss / inputs.length
    }

    return totalLoss
  }

  // Predict with confidence (for binary classification)
  predict(input: number[]): { value: number; confidence: number } {
    const output = this.forward(input)
    const value = output[0]

    // For sigmoid output, confidence is how far from 0.5
    const confidence = Math.abs(value - 0.5) * 2

    return { value, confidence }
  }

  // Save weights to JSON
  toJSON(): any {
    return {
      inputSize: this.layers[0].weights.rows,
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

  // Load weights from JSON
  static fromJSON(json: any): NeuralNet {
    const net = new NeuralNet(json.learningRate, json.l2Lambda)

    net.layers = json.layers.map((layerData: any) => ({
      weights: new Matrix(layerData.weightsShape[0], layerData.weightsShape[1], layerData.weights),
      bias: new Matrix(layerData.biasShape[0], layerData.biasShape[1], layerData.bias),
      activation: layerData.activation,
    }))

    return net
  }

  // Get current learning rate
  getLearningRate(): number {
    return this.learningRate
  }

  // Update learning rate (for decay)
  setLearningRate(rate: number): void {
    this.learningRate = rate
  }

  // Calculate total number of parameters
  getParameterCount(): number {
    return this.layers.reduce((sum, layer) => {
      return sum + layer.weights.data.length + layer.bias.data.length
    }, 0)
  }
}

// Example usage for Berghain problem
export function createBerghainNet(inputSize: number = 17): NeuralNet {
  // Input layer: 17 features
  // - 4 person attributes
  // - 4 constraint satisfaction ratios
  // - 4 constraint pressure scores
  // - 3 global features
  // - 1 alignment + 1 correlation
  const net = new NeuralNet(0.001, 0.0001)
  net.addLayer(inputSize, 24, 'relu', 'he')
  net.addLayer(24, 12, 'relu', 'he')
  net.addLayer(12, 1, 'sigmoid', 'xavier')
  return net
}
