import { Glob } from 'bun'
import type { GameState } from '../types'

/**
 *  ## Disk
 *
 *  Shared utilities for reading and saving files to disk.
 */
export namespace Disk {
  /**
   * Returns an array of all files in the specified directory.
   */
  export async function getFilePathsInDir(
    pattern = 'data/*'
  ): Promise<string[]> {
    console.log(`[disk] reading files from directory: "${pattern}"`)
    const filePaths: string[] = []
    const glob = new Glob(pattern)
    for await (const filePath of glob.scan({ absolute: true })) {
      filePaths.push(filePath)
    }
    console.log('[disk] found files:', filePaths)
    return filePaths
  }

  /**
   * Load all files in the specified directory.
   */
  export async function getJsonDataFromFiles<T>(): Promise<T[]> {
    const dataDir = 'data'
    const filePaths = await getFilePathsInDir(`${dataDir}/*.json`)
    const filePromises = filePaths.map(async (path) => {
      try {
        const json = await Bun.file(path).json()
        return json as T
      } catch (e) {
        console.warn(`[disk] failed to load file: "${path}"`, e)
        return undefined
      }
    })
    const files = await Promise.all(filePromises)
    return files.filter((file): file is Awaited<T> => file !== undefined)
  }

  /**
   *  Save the game state to a file.
   */
  export async function saveGameState<T>(gameState: GameState<T>) {
    const path = `data/scenario-${gameState.scenario}-${gameState.game.gameId}.json`
    const file = Bun.file(path)
    await file.write(JSON.stringify(gameState, null, 2))
    return path
  }
}
