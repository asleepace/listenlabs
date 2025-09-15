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
   * @default 'data/*''
   */
  export async function getFilePathsInDir(pattern = 'data/*'): Promise<string[]> {
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

  /**
   *  Simple help for saving JSON.
   */
  export async function saveJsonFile<T extends {}>(filePath: string, jsonData: T) {
    try {
      const file = Bun.file(filePath)
      await file.write(JSON.stringify(jsonData, null, 2))
      return filePath
    } catch (e) {
      console.warn('[disk] failed to save data:')
      console.log(jsonData)
      return undefined
    }
  }

  /**
   *  Simple help for saving JSON.
   */
  export async function getJsonFile<T extends {}>(filePath: string) {
    const file = Bun.file(filePath)
    const data = await file.json()
    if (!data) throw new Error(`Disk: failed to load file "${filePath}"`)
    return data as T
  }
}
