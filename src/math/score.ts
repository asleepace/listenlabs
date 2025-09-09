import * as Calculation from './score-calculation'
import * as Defaltion from './score-deflation'

export namespace Score {
  // export scoring presets
  export const PRESETS = Calculation.SCORE_PRESETS

  // calculating score deflation factor (all of these do the same thing)
  export const getScoreDeflationFactorCombined =
    Defaltion.getScoreDeflationFactorCombined
  export const getAdaptiveDeflationFactor = Defaltion.getAdaptiveDeflationFactor
  export const getExponentialDeflationFactor =
    Defaltion.getExponentialDeflationFactor
  export const getPowerLawDeflationFactor = Defaltion.getPowerLawDeflationFactor
  export const getScoreDeflationFactor = Defaltion.getScoreDeflationFactor

  // calculating admission scores
  export const calculateAdmissionScore = Calculation.calculateAdmissionScore
  export const calculateEndgameScore = Calculation.calculateEndgameScore
  export const getScoreAnalysis = Calculation.getScoreAnalysis
}
