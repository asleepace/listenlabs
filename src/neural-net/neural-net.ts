/** @file neural-net.ts */

import { Matrix, activations, gradients } from './matrix'

export interface Layer {
  weights: Matrix
  bias: Matrix
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear'
}

function safeMap(m: Matrix, cap = 1e6): Matrix {
  return m.map((v) => (Number.isFinite(v) ? Math.max(-cap, Math.min(cap, v)) : 0))
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

  // ↓ default LR lowered for stability
  constructor(learningRate = 0.0003, l2Lambda = 0.0001) {
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

    const CLIP_ACT = (v: number) => Math.max(-60, Math.min(60, v)) // guard exp()/tanh() blowups

    for (let li = 0; li < this.layers.length; li++) {
      const layer = this.layers[li]
      if (!layer) {
        throw new Error(`NeuralNet: Missing layer (${li}) in forward.`)
      }

      const z = current.dot(layer.weights).add(layer.bias)

      let output: Matrix
      switch (layer.activation) {
        case 'relu':
          // ReLU is safe without clipping
          output = z.map(activations.relu)
          break
        case 'sigmoid':
          // clip pre-activations before exp
          output = z.map((v) => activations.sigmoid(CLIP_ACT(v)))
          break
        case 'tanh':
          // clip pre-activations before tanh
          output = z.map((v) => activations.tanh(CLIP_ACT(v)))
          break
        case 'linear':
          output = z
          break
      }

      output = safeMap(output)

      this.layerOutputs.push(output.copy())
      if (li < this.layers.length - 1) {
        this.layerInputs.push(output.copy())
      }
      current = output
    }

    const out = current.toArray()
    for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) out[i] = 0.5
    return out
  }

  /** Alias for readability in callers that expect `infer(...)`. */
  infer(input: number[] | Matrix): number[] {
    return this.forward(input)
  }

  backward(target: number[] | number, _predicted?: number[]): void {
    if (!this.layerOutputs.length) {
      throw new Error('NeuralNet: Must call forward() before backward()')
    }

    const targetArray = typeof target === 'number' ? [target] : (target as number[])
    // Label smoothing: 0→0.02, 1→0.98
    const ySmooth = targetArray.map((t) => t * 0.96 + 0.02)
    const targetMatrix = Matrix.fromArray(ySmooth).transpose()
    const outputLayer = this.layers[this.layers.length - 1]
    const output = this.layerOutputs[this.layerOutputs.length - 1]

    // Loss gradient:
    // For sigmoid output, use BCE-style gradient: dL/dz = (ŷ - y_smooth)
    let delta: Matrix
    if (outputLayer.activation === 'sigmoid') {
      delta = output.subtract(targetMatrix)
    } else {
      // Fallback for non-sigmoid heads
      delta = output.subtract(targetMatrix).scale(2 / output.cols)
      if (outputLayer.activation === 'tanh') {
        delta = delta.hadamard(output.map((y) => gradients.tanh(y)))
      }
    }
    // Small clip on delta to stop rare spikes
    const CLIP = 5
    delta = delta.map((v) => Math.max(-CLIP, Math.min(CLIP, v)))

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      const layerInput = this.layerInputs[i]

      if (!layer || !layerInput) {
        throw new Error(`NeuralNet: Missing layer (${i}) in backward.`)
      }

      // snapshot weights BEFORE updating (for correct chain rule)
      const Wprev = layer.weights.copy()

      let weightGrad = layerInput.transpose().dot(delta)
      let biasGrad = delta.copy()

      // Gradient-norm clipping (per-layer)
      const gradNorm =
        Math.sqrt(weightGrad.data.reduce((s, v) => s + v * v, 0) + biasGrad.data.reduce((s, v) => s + v * v, 0)) + 1e-12
      const MAX_NORM = 10
      const scale = Math.min(1, MAX_NORM / gradNorm)
      if (scale < 1) {
        weightGrad = weightGrad.scale(scale)
        biasGrad = biasGrad.scale(scale)
      }

      // L2 regularization
      const weightUpdate = weightGrad.add(layer.weights.scale(this.l2Lambda)).scale(this.learningRate)

      layer.weights = layer.weights.subtract(weightUpdate)
      layer.bias = layer.bias.subtract(biasGrad.scale(this.learningRate))

      // Sanitize weights/bias (no NaN/Inf)
      for (let k = 0; k < layer.weights.data.length; k++) {
        const v = layer.weights.data[k]
        layer.weights.data[k] = Number.isFinite(v) ? Math.max(Math.min(v, 1e6), -1e6) : 0
      }
      for (let k = 0; k < layer.bias.data.length; k++) {
        const v = layer.bias.data[k]
        layer.bias.data[k] = Number.isFinite(v) ? Math.max(Math.min(v, 1e6), -1e6) : 0
      }

      if (i > 0) {
        // propagate with pre-update weights
        delta = delta.dot(Wprev.transpose())

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

        // BCE loss with the same smoothed label we use in backward
        const y = target[0] * 0.96 + 0.02
        const yhat = output[0]
        const eps = 1e-7
        const yhatClamped = Math.min(1 - eps, Math.max(eps, yhat))
        const loss = -(y * Math.log(yhatClamped) + (1 - y) * Math.log(1 - yhatClamped))

        epochLoss += loss
        this.backward([target[0]])
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
    const net = new NeuralNet(json.learningRate ?? 0.0003, json.l2Lambda ?? 0.0001)
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
  const net = new NeuralNet(0.0003, 0.00001)
  net.addLayer(inputSize, 64, 'relu', 'he')
  net.addLayer(64, 32, 'relu', 'he')
  net.addLayer(32, 1, 'sigmoid', 'xavier')
  return net
}
