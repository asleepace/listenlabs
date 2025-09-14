/** @file util.ts */

import type { ScenarioAttributes } from '../types'

export type Range = readonly [lowerBound: number, upperBound: number]

/**
 * Clamps a value between a range of values.
 *
 * ```ts
 * clamp(threshold, [0.1, 1.0])
 * ```
 */
export const clamp = (value: number, range: Range): number => {
  let [low, high] = range
  if (low > high) [low, high] = [high, low]
  if (Number.isNaN(value)) return value
  return value < low ? low : value > high ? high : value
}

/**
 *  Sums an array of values or returns 0 if empty.
 */
export const sum = (vals: number[]): number => vals.reduce((a, b) => a + b, 0) || 0

/**
 *  Computes the average of an array of values or returns 0 if emtpy.
 */
export const average = (values: number[]): number => {
  if (!values.length) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

/**
 * Returns the difference between two values.
 */
export const diff = (v1: number, v2: number) => {
  return (v1 - v2) / 2
}

/**
 * Returns an array of attributes on guest which are true.
 */
export const getAttributes = (guest: ScenarioAttributes): string[] => {
  return Object.entries(guest)
    .filter(([_, value]) => value)
    .map(([key]) => key)
}

/**
 * Returns a number with specified amout of percision.
 */
export const toFixed = (x: number, fractionDigits = 0): number => {
  return +x.toFixed(fractionDigits)
}

/**
 * Helper which will parse any flags passed via the command line.
 */
export const parseFlags = (args: string[]): Record<string, string> => {
  const flags: Record<string, string> = {}
  for (const a of args) {
    if (a.startsWith('--')) {
      const i = a.indexOf('=')
      if (i > 2) flags[a.slice(2, i)] = a.slice(i + 1)
      else flags[a.slice(2)] = 'true'
    }
  }
  return flags
}
