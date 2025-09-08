import { parseArgs } from 'util'
import { BASE_CONFIG, type GameConfig } from '../conf/game-config'

const getTypeThenCast = (keyString: string, value: string | boolean) => {
  const key = keyString.toUpperCase() as keyof GameConfig
  if (!(key in BASE_CONFIG))
    throw new Error('Invalid command line argument: ' + key)
  if (typeof BASE_CONFIG[key] === 'number') {
    return {
      [key]: Number(value),
    }
  }
  if (typeof BASE_CONFIG[key] === 'string') {
    return {
      [key]: String(value),
    }
  }
  if (typeof BASE_CONFIG[key] === 'boolean') {
    return {
      [key]:
        typeof value === 'boolean' ? value : value === 'true' ? true : false,
    }
  } else {
    throw new Error('Invalid command line argument type: ' + key + typeof value)
  }
}

/**
 *  Get command line arguments to override configurations.
 */
export function getCliArgs(baseConfig: GameConfig = BASE_CONFIG) {
  /**
   *  Allow overriding specific options via the cli.
   */
  const options = Object.entries(baseConfig).reduce((params, [name, value]) => {
    return {
      ...params,
      [name.toLowerCase()]: {
        type: typeof value === 'boolean' ? 'boolean' : 'string',
      },
    }
  }, {})

  const { values } = parseArgs({
    args: Bun.argv,
    options: {
      ...options,
      target: {
        short: 't',
        type: 'string',
      },
      message: {
        short: 'm',
        type: 'string',
      },
      scenario: {
        short: 's',
        type: 'string',
      },
    },
    strict: true,
    allowPositionals: true,
  })

  return Object.entries(values).reduce((output, [key, value]) => {
    return {
      ...output,
      ...getTypeThenCast(key as any, value),
    }
  }, BASE_CONFIG) as GameConfig
}
