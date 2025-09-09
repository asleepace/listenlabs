import { Bouncer } from './bouncer'
import { Berghain, type ListenLabsConfig } from './core/berghain'
import { getCliArgs } from './cli/get-cli-args'

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

/**
 *  Initialize the game with the desired scenario and then pass
 *  in the bouncer you would like to use. Then either create a
 *  new game or load one from disk.
 */
await Berghain.initialize({
  scenario: configuration.SCENARIO || settings.scenario,
})
  .withBouncer(Bouncer.intialize(configuration))
  .startNewGame()
  .catch(console.warn)
  .finally(() => {
    console.log('===============================================')
    console.log(`>>  ${configuration.MESSAGE || 'finished!'}`)
    console.log('===============================================')
  })
