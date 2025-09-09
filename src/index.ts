import { Berghain, type ListenLabsConfig } from './berghain'
import { getCliArgs } from './cli/get-cli-args'
import { HansBouncer } from './hans-bouncer'

console.log('+'.repeat(64))

/**
 *  Configure listen labs default settings.
 */
const settings: ListenLabsConfig = {
  uniqueId: 'c9d5d12b-66e6-4d03-909b-5825fac1043b',
  baseUrl: 'https://berghain.challenges.listenlabs.ai/',
  scenario: '3',
}

/**
 *  Set game configuration and allow overriding via the command line,
 *
 */
const configuration = getCliArgs()

console.log('[game] settings:', configuration)

const scenario = configuration.SCENARIO

if (!scenario) {
  throw new Error(`Missing scenario: "${scenario}"`)
}

/**
 *  Initialize the game with the desired scenario and then pass
 *  in the bouncer you would like to use. Then either create a
 *  new game or load one from disk.
 */
await Berghain.initialize({
  scenario,
})
  .withBouncer(async (initialState) => {
    const bouncer = new HansBouncer(initialState, {
      MAX_CAPACITY: 1000,
      TOTAL_PEOPLE: 10_000,
      TARGET_RANGE: 0,
    })
    await bouncer.initializeLearningData()
    return bouncer
  })
  .startNewGame()
  .catch(console.warn)
  .finally(async () => {
    console.log('===============================================')
    console.log(`>>  ${configuration.MESSAGE || 'finished!'}`)
    console.log('===============================================')
  })
