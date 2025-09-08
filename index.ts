import { Bouncer, CONFIG } from './src/bouncer'
import { Berghain, type ListenLabsConfig } from './src/berghain'

/**
 *  Configure listen labs default settings.
 */
const settings: ListenLabsConfig = {
  uniqueId: 'c9d5d12b-66e6-4d03-909b-5825fac1043b',
  baseUrl: 'https://berghain.challenges.listenlabs.ai/',
  scenario: '3',
}

const config = {
  message: '',
  ...CONFIG,
}

/**
 *  Process command line arguments.
 */
Bun.argv.slice(2).forEach((current, index, array) => {
  const nextItem = array.at(index + 1)

  switch (current) {
    case '--scenario':
    case '-s': {
      settings.scenario = String(Number(nextItem)) as '1' | '2' | '3'
      break
    }

    case '--target':
    case '-t': {
      config.TARGET_RANGE = Number(nextItem)
      break
    }

    case '--base':
    case '-b': {
      config.BASE_THRESHOLD = Number(nextItem)
      break
    }

    case '--message':
    case '-m': {
      // useful for tracking different expiraments
      config.message = String(nextItem)
    }
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
  .finally(() => {
    console.log('====================== ✉️ ======================')
    console.log(`[game] message: "${config.message}"`)
    console.log('====================== - ======================')
  })
