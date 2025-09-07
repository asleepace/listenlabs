import { Bouncer } from './src/bouncer'
import { Berghain, type ListenLabsConfig } from './src/berghain'

/**
 *  Configure listen labs default settings.
 */
const settings: ListenLabsConfig = {
  uniqueId: 'c9d5d12b-66e6-4d03-909b-5825fac1043b',
  baseUrl: 'https://berghain.challenges.listenlabs.ai/',
  scenario: '3',
}

/**
 *  Process command line arguments.
 */
Bun.argv.slice(2).forEach((current, index, array) => {
  const nextItem = array.at(index + 1)
  if (current === '--scenario' || (current === '-s' && nextItem)) {
    settings.scenario = String(Number(nextItem)) as '1' | '2' | '3'
  }
})

console.log('[game] settings:', settings)

/**
 *  Initialize the game with the desired scenario and then pass
 *  in the bouncer you would like to use. Then either create a
 *  new game or load one from disk.
 */
await Berghain.initialize({ scenario: settings.scenario })
  .withBouncer(
    Bouncer.intialize({
      TARGET_RANGE: 4000,
      BASE_THRESHOLD: 0.49,
      MIN_RAW_SCORES: 5,
    })
  )
  .startNewGame()
  .catch(console.warn)
