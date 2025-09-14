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

// ESM __dirname shim
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export async function initializeNeuralNetwork(initialState: GameState): Promise<BerghainBouncer> {
  const weightsPath = path.resolve(__dirname, `../bouncer-data/weights-s2.best-4407avg.json`)

  console.log('[neural-net] weights:', weightsPath)

  // build bouncer (no exploration for prod)
  const bouncer = new NeuralNetBouncer(initialState.game, {
    explorationRate: 0,
    baseThreshold: 0.32, // same as training
    minThreshold: 0.22,
    maxThreshold: 0.62,
    urgencyFactor: 2.0,
  })

  // load weights (fallback to fresh net if missing)
  try {
    const raw = await fs.readFile(weightsPath, 'utf-8')
    const json = JSON.parse(raw)
    const net = NeuralNet.fromJSON(json)
    bouncer.setNetwork(net)
    console.log(`[bouncer] Loaded weights: ${weightsPath}`)
  } catch (err) {
    console.warn(`[bouncer] No weights at ${weightsPath} — using fresh net.`, (err as Error).message)
    const net = new NeuralNet(0.0003, 0.00001)
    // build same topology as training (input size is inferred by StateEncoder at runtime)
    // you can keep a tiny helper if you prefer: createBerghainNet(...)
    // net.addLayer(<featureSize>, 64, 'relu', 'he')... <-- not needed if you will immediately load weights later
    bouncer.setNetwork(net)
  }

  const scoring = initializeScoring(initialState.game, {
    maxAdmissions: Conf.MAX_ADMISSIONS,
    maxRejections: Conf.MAX_REJECTIONS,
  })

  let lastPerson: ScenarioAttributes = {} as ScenarioAttributes
  let lastAdmit = false

  return {
    admit(next) {
      lastPerson = next.nextPerson.attributes
      const admit = bouncer.admit(next, scoring.getCounts())
      scoring.update({ guest: next.nextPerson.attributes, admit })
      lastAdmit = admit
      return admit
    },
    getProgress() {
      const attributes = getAttributes(lastPerson).join(',')
      return {
        decision: `[${attributes}]=${lastAdmit}`,
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
      return {
        ...initialState,
        status: lastStatus.status,
        summary: scoring.getSummary(),
      }
    },
  }
}
