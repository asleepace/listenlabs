// src/main.ts (where you currently wire BanditBouncer)
import * as fs from 'fs/promises'
import * as path from 'path'
import { fileURLToPath } from 'url'
import { NeuralNetBouncer } from './neural-net-bouncer'
import { NeuralNet } from './neural-net'
import type { BerghainBouncer, GameState, ScenarioAttributes } from '../types'
import { initializeScoring } from './scoring'
import { Conf } from './config'
import { getAttributes } from './util'
import { Disk } from '../utils/disk'

// ESM __dirname shim
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const trainingData: string[][] = []

enum Weights {
  scenario2BestAverage = `../../bouncer-data/weights-s2.best-4407avg.json`,
  scenario2Normal = `../../bouncer-data/weights-scenario-2.json`,
}

export async function initializeNeuralNetwork(initialState: GameState): Promise<BerghainBouncer> {
  const weightsPath = path.resolve(__dirname, Weights.scenario2Normal)

  console.log('[neural-net] weights:', weightsPath)

  // build bouncer (no exploration for prod)
  const bouncer = new NeuralNetBouncer(initialState.game, {
    explorationRate: 0,
    baseThreshold: 0.28,
    minThreshold: 0.18,
    maxThreshold: 0.6,
    urgencyFactor: 1.5, // a little less tightening
    optimism: 0.8, // bigger slack before gates kick in
  })

  // load weights (fallback to fresh net if missing)
  const raw = await fs.readFile(weightsPath, 'utf-8')
  const json = JSON.parse(raw)
  const net = NeuralNet.fromJSON(json)
  bouncer.setNetwork(net)
  console.log(`[bouncer] Loaded weights: ${weightsPath}`)

  const scoring = initializeScoring(initialState.game, {
    maxAdmissions: Conf.MAX_ADMISSIONS,
    maxRejections: Conf.MAX_REJECTIONS,
  })

  let lastPerson: ScenarioAttributes = {} as ScenarioAttributes
  let lastAdmit = false
  let isSampleOnly = false // allow game to progress to 10,000

  const outputFile = `data/random-sample-${+new Date()}.json`

  const saveRandomSample = () => {
    Disk.saveJsonFile(outputFile, trainingData).catch(console.warn)
  }

  return {
    admit(next) {
      lastPerson = next.nextPerson.attributes
      if (isSampleOnly) return false
      const admit = bouncer.admit(next, scoring.getCounts())
      scoring.update({ guest: next.nextPerson.attributes, admit })
      lastAdmit = admit
      return admit
    },
    getProgress() {
      const attributes = getAttributes(lastPerson)
      trainingData.push(attributes)

      if (trainingData.length % 100 === 0) saveRandomSample()

      return {
        currentIndex: trainingData.length,
        decision: lastAdmit,
        attributes,
        admitted: scoring.admitted,
        rejected: scoring.rejected,
        quotas: scoring.quotas().map((quota) => ({
          attribute: quota.attribute,
          relative: (quota.relativeProgress(scoring.getPeopleLeftInLine()) * 100).toFixed(2) + '%',
          progress: (quota.progress() * 100).toFixed(2) + '%',
          needed: quota.needed(),
        })),
      }
    },
    getOutput(lastStatus) {
      saveRandomSample()

      return {
        ...initialState,
        status: lastStatus.status,
        summary: scoring.getSummary(),
      }
    },
  }
}
