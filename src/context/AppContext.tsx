import React, { createContext, useContext, useState, useEffect } from "react"
import { checkBackendHealth, PrivacyProcessResponse } from "../services/api"

export interface DatasetState {
  file: File | null
  name: string
  size: string
  records: number
  columns: number
  targetColumn: string
  missingValues: number
  previewRows: Array<Record<string, string>>
  status: "idle" | "uploading" | "ready" | "error"
  uploadProgress: number
  errorMessage?: string
}

export interface PrivacyConfigState {
  epsilon: number
  delta: number
  sensitivity: number
  mechanism: string
  mechanismType: "plain_laplace" | "laplace_clipped" | "non_private"
  classifierType: "logistic_regression" | "random_forest"
  noiseScale: string
}

export interface ProcessingState {
  status: "idle" | "processing" | "completed" | "error"
  currentStep: number
  progressPct: number
  error: string | null
}

interface AppContextType {
  authenticated: boolean
  login: () => void
  logout: () => void
  dataset: DatasetState
  setDataset: React.Dispatch<React.SetStateAction<DatasetState>>
  privacyConfig: PrivacyConfigState
  setPrivacyConfig: React.Dispatch<React.SetStateAction<PrivacyConfigState>>
  processingState: ProcessingState
  setProcessingState: React.Dispatch<React.SetStateAction<ProcessingState>>
  experimentResults: PrivacyProcessResponse | null
  setExperimentResults: React.Dispatch<React.SetStateAction<PrivacyProcessResponse | null>>
  lastRunResult: PrivacyProcessResponse | null
  setLastRunResult: React.Dispatch<React.SetStateAction<PrivacyProcessResponse | null>>
  backendStatus: { online: boolean; message: string }
  checkBackend: () => Promise<void>
}

const defaultDataset: DatasetState = {
  file: null,
  name: "CreditCard.csv",
  size: "143.8 MB",
  records: 284807,
  columns: 31,
  targetColumn: "Class",
  missingValues: 0,
  previewRows: [
    {
      Time: "0",
      V1: "-1.360",
      V2: "-0.073",
      V3: "2.536",
      Amount: "149.62",
      Class: "0",
    },
    {
      Time: "0",
      V1: "1.192",
      V2: "0.266",
      V3: "0.167",
      Amount: "2.69",
      Class: "0",
    },
    {
      Time: "1",
      V1: "-1.358",
      V2: "-1.340",
      V3: "1.773",
      Amount: "378.66",
      Class: "0",
    },
    {
      Time: "1",
      V1: "-0.967",
      V2: "-0.185",
      V3: "1.793",
      Amount: "123.50",
      Class: "0",
    },
    {
      Time: "2",
      V1: "-1.158",
      V2: "0.878",
      V3: "1.549",
      Amount: "69.99",
      Class: "1",
    },
  ],
  status: "ready",
  uploadProgress: 100,
}

const defaultPrivacyConfig: PrivacyConfigState = {
  epsilon: 1.0,
  delta: 0.00001,
  sensitivity: 1.0,
  mechanism: "laplace_clipped",
  mechanismType: "laplace_clipped",
  classifierType: "logistic_regression",
  noiseScale: "1.0000",
}

const defaultProcessing: ProcessingState = {
  status: "completed",
  currentStep: 12,
  progressPct: 100,
  error: null,
}

const defaultCurve = [
  {
    epsilon: 0.1,
    accuracy: 0.891,
    precision: 0.872,
    recall: 0.801,
    f1: 0.823,
    rocAuc: 0.934,
    miaRisk: 0.511,
    reconRisk: 0.08,
  },
  {
    epsilon: 0.5,
    accuracy: 0.933,
    precision: 0.912,
    recall: 0.843,
    f1: 0.871,
    rocAuc: 0.962,
    miaRisk: 0.534,
    reconRisk: 0.12,
  },
  {
    epsilon: 1.0,
    accuracy: 0.948,
    precision: 0.932,
    recall: 0.857,
    f1: 0.893,
    rocAuc: 0.974,
    miaRisk: 0.552,
    reconRisk: 0.16,
  },
  {
    epsilon: 2.0,
    accuracy: 0.957,
    precision: 0.941,
    recall: 0.869,
    f1: 0.908,
    rocAuc: 0.981,
    miaRisk: 0.579,
    reconRisk: 0.22,
  },
  {
    epsilon: 5.0,
    accuracy: 0.963,
    precision: 0.948,
    recall: 0.879,
    f1: 0.917,
    rocAuc: 0.986,
    miaRisk: 0.621,
    reconRisk: 0.31,
  },
  {
    epsilon: 10.0,
    accuracy: 0.967,
    precision: 0.953,
    recall: 0.887,
    f1: 0.923,
    rocAuc: 0.989,
    miaRisk: 0.668,
    reconRisk: 0.42,
  },
]

const defaultResults: PrivacyProcessResponse = {
  success: true,
  experimentId: "EXP-DEFAULT",
  epsilon: 1.0,
  delta: 0.00001,
  sensitivity: 1.0,
  mechanism: "Laplace",
  noiseScale: 1.0,
  metrics: {
    accuracy: 0.948,
    precision: 0.932,
    recall: 0.857,
    f1: 0.893,
    rocAuc: 0.974,
    miaRisk: 0.552,
    reconRisk: 0.16,
    confusionMatrix: {
      tn: 56839,
      fp: 25,
      fn: 96,
      tp: 813,
    },
  },
  tradeoffCurve: defaultCurve,
  pareto_curve: defaultCurve,
  confusion_matrix: [
    [56839, 25],
    [96, 813],
  ],
  processed_rows: 57773,
  mia_risk: {
    without_dp: 0.721,
    with_dp: 0.552,
  },
  reconstruction_risk: {
    without_dp: 0.012,
    with_dp: 0.16,
  },
  mechanism_type: "laplace_clipped",
  classifier_type: "logistic_regression",
  relative_utility_loss: 4.8,
  baselines: {
    non_private: {
      accuracy: 0.968,
      precision: 0.952,
      recall: 0.895,
      f1: 0.923,
      rocAuc: 0.989,
      description:
        "Standard training on unperturbed clean data (epsilon = inf)",
    },
    encrypted_non_private: {
      accuracy: 0.968,
      precision: 0.952,
      recall: 0.895,
      f1: 0.923,
      rocAuc: 0.989,
      encryption: "AES-256 / Fernet verified",
      description:
        "Encrypted at rest with transient RAM decryption, without Differential Privacy (epsilon = inf)",
    },
  },
  empirical_attacks: {
    mia: {
      attack_accuracy: 0.552,
      attacker_advantage: 0.084,
      attack_auc: 0.562,
    },
    reconstruction: {
      nmse: 0.16,
      baseline_nmse: 0.012,
    },
  },
  model_comparison: {
    logistic_regression: {
      accuracy: 0.948,
      precision: 0.932,
      recall: 0.857,
      f1: 0.893,
      rocAuc: 0.974,
    },
    random_forest: {
      accuracy: 0.959,
      precision: 0.946,
      recall: 0.871,
      f1: 0.907,
      rocAuc: 0.981,
    },
  },
  espinosa_benchmark: [
    {
      epsilon: 0.1,
      privacyguard_rul: 11.8,
      espinosa_rul: 14.8,
      delta_advantage: 3.0,
      domain: "Banking Fraud (Espinosa et al., 2026)",
    },
    {
      epsilon: 1.0,
      privacyguard_rul: 4.8,
      espinosa_rul: 6.2,
      delta_advantage: 1.4,
      domain: "Banking Fraud (Espinosa et al., 2026)",
    },
    {
      epsilon: 5.0,
      privacyguard_rul: 1.9,
      espinosa_rul: 2.4,
      delta_advantage: 0.5,
      domain: "Banking Fraud (Espinosa et al., 2026)",
    },
    {
      epsilon: 10.0,
      privacyguard_rul: 0.7,
      espinosa_rul: 0.9,
      delta_advantage: 0.2,
      domain: "Banking Fraud (Espinosa et al., 2026)",
    },
  ],
  isSimulated: true,
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [authenticated, setAuthenticated] = useState<boolean>(true)
  const [dataset, setDataset] = useState<DatasetState>(defaultDataset)
  const [privacyConfig, setPrivacyConfig] =
    useState<PrivacyConfigState>(defaultPrivacyConfig)
  const [processingState, setProcessingState] =
    useState<ProcessingState>(defaultProcessing)
  const [experimentResults, setExperimentResultsState] =
    useState<PrivacyProcessResponse | null>(defaultResults)
  const [lastRunResult, setLastRunResultState] =
    useState<PrivacyProcessResponse | null>(defaultResults)

  const setExperimentResults: React.Dispatch<React.SetStateAction<PrivacyProcessResponse | null>> =
    (action) => {
      setExperimentResultsState((prev) => {
        const next = typeof action === "function" ? action(prev) : action
        setLastRunResultState(next)
        return next
      })
    }

  const setLastRunResult: React.Dispatch<React.SetStateAction<PrivacyProcessResponse | null>> =
    (action) => {
      setLastRunResultState((prev) => {
        const next = typeof action === "function" ? action(prev) : action
        setExperimentResultsState(next)
        return next
      })
    }

  const [backendStatus, setBackendStatus] = useState<{
    online: boolean
    message: string
  }>({
    online: false,
    message: "Checking backend...",
  })

  const checkBackend = async () => {
    const status = await checkBackendHealth()
    setBackendStatus(status)
  }

  useEffect(() => {
    checkBackend()
  }, [])

  const login = () => setAuthenticated(true)
  const logout = () => setAuthenticated(false)

  return (
    <AppContext.Provider
      value={{
        authenticated,
        login,
        logout,
        dataset,
        setDataset,
        privacyConfig,
        setPrivacyConfig,
        processingState,
        setProcessingState,
        experimentResults,
        setExperimentResults,
        lastRunResult,
        setLastRunResult,
        backendStatus,
        checkBackend,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) {
    throw new Error("useApp must be used within an AppProvider")
  }
  return ctx
}
