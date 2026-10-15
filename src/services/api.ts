/**
 * api.ts — Centralized API service for PrivacyGuard backend (FastAPI).
 * Default endpoint: http://localhost:8000/api
 *
 * Scientific correctness: NO simulation fallbacks.
 * If the backend is unavailable, functions throw an error.
 * The UI must display "--" or "Experiment unavailable", NOT simulated data.
 */

// ─── Request/Response Types ────────────────────────────────────────────────

export interface UploadResponse {
  success: boolean
  message: string
  datasetId: string
  filename: string
  filesize: string
  records: number
  columns: number
  columnNames: string[]
  targetColumn: string
  missingValues: number
  previewRows: Record<string, string>[]
}

export interface PrivacyProcessParams {
  datasetId?: string
  epsilon: number
  delta?: number
  sensitivity: number
  mechanism?: string
  mechanism_type?: string
  classifier_type?: string
  seed?: number
  n_shadow?: number
}

export interface EpsilonAccounting {
  epsilon_record: number
  epsilon_feature: number
  feature_count: number
  composition_method: string
}

export interface MIAMetrics {
  attack_accuracy: number | null
  attack_auc: number | null
  attacker_advantage: number | null
  tpr_at_1fpr: number | null
  tpr_at_01fpr: number | null
  n_members_evaluated?: number
  n_nonmembers_evaluated?: number
  error?: string
}

export interface ParetoPoint {
  epsilon: number
  epsilon_feature?: number | null
  // Utility
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1?: number | null
  rocAuc?: number | null
  prAuc?: number | null
  fraud_f1?: number | null
  fraud_recall?: number | null
  // MIA (from real shadow attack)
  miaAccuracy?: number | null
  miaAuc?: number | null
  attackerAdvantage?: number | null
  tprAt1Fpr?: number | null
  // Legacy
  miaRisk?: number | null
  reconRisk?: number | null
}

export interface UtilityMetrics {
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1?: number | null
  rocAuc?: number | null
  prAuc?: number | null
  fraud_precision?: number | null
  fraud_recall?: number | null
  fraud_f1?: number | null
  fraud_support?: number | null
  confusionMatrix?: {
    tn: number
    fp: number
    fn: number
    tp: number
  } | null
  // Legacy (kept for compat)
  miaRisk?: number | null
  reconRisk?: number | null
}

export interface BaselineMetric {
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1?: number | null
  rocAuc?: number | null
  prAuc?: number | null
  description?: string
}

export interface PrivacyProcessResponse {
  success: boolean
  experimentId: string
  is_simulated: boolean

  // DP config
  epsilon: number
  delta: number
  sensitivity: number
  mechanism: string
  mechanism_type?: string | null
  mechanism_label?: string | null
  classifier_type?: string | null
  noiseScale: number

  // Epsilon accounting
  epsilon_accounting?: EpsilonAccounting | null

  // Dataset split
  split_seed?: number | null
  train_size?: number | null
  test_size?: number | null
  fraud_count_train?: number | null
  fraud_count_test?: number | null
  feature_list?: string[] | null

  // Metrics
  metrics: UtilityMetrics

  // MIA
  mia_result?: MIAMetrics | null
  mia_per_class?: Record<string, unknown> | null

  // Tradeoff curve
  tradeoffCurve: ParetoPoint[]
  pareto_curve?: ParetoPoint[] | null

  // Baselines
  baselines?: Record<string, BaselineMetric> | null

  // Other
  model_comparison?: Record<string, BaselineMetric> | null
  confusion_matrix?: number[][] | null
  processed_rows?: number | null
  relative_utility_loss?: number | null

  // Legacy
  mia_risk?: { without_dp: number; with_dp: number } | null
  reconstruction_risk?: { without_dp: number; with_dp: number } | null
  empirical_attacks?: Record<string, unknown> | null
}

// ─── Experiment History ────────────────────────────────────────────────────

export interface ExperimentRecord {
  id: string
  dataset: string
  epsilon: number
  epsilon_record?: number | null
  epsilon_feature?: number | null
  feature_count?: number | null
  composition_method?: string | null
  delta?: number | null
  mechanism: string
  classifier_type?: string | null
  date: string
  timestamp?: string
  // Utility (formatted as "xx.x%")
  accuracy?: string | null
  precision?: string | null
  recall?: string | null
  f1?: string | null
  rocAuc?: string | null
  prAuc?: string | null
  // Fraud class
  fraud_f1?: string | null
  fraud_recall?: string | null
  // MIA
  mia_accuracy?: string | null
  mia_auc?: string | null
  attacker_advantage?: string | null
  tpr_at_1fpr?: string | null
  tpr_at_01fpr?: string | null
  // Status
  privacyRisk?: "LOW" | "MEDIUM" | "HIGH" | null
  status: string
  hasDataset: boolean
  hasModel: boolean
  is_simulated: boolean
}

export interface ExperimentRunRequest {
  mode?: "fast" | "full"
  epsilon_grid?: number[]
  seeds?: number[]
  mechanisms?: string[]
  classifier_types?: string[]
  n_shadow?: number
}

// ─── API Base URL ──────────────────────────────────────────────────────────

let activeApiBase = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api"

export function getApiBase(): string {
  return activeApiBase
}

export class BackendOfflineError extends Error {
  constructor() {
    super("Backend is offline. Start the FastAPI server: cd backend && uvicorn main:app --reload")
    this.name = "BackendOfflineError"
  }
}

// ─── Health Check ──────────────────────────────────────────────────────────

export async function checkBackendHealth(): Promise<{
  online: boolean
  message: string
}> {
  const candidates = [
    activeApiBase,
    "http://127.0.0.1:8000/api",
    "http://localhost:8000/api",
  ]
  const uniqueCandidates = Array.from(new Set(candidates))

  for (const url of uniqueCandidates) {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 2000)
      const res = await fetch(`${url}/health`, {
        method: "GET",
        signal: controller.signal,
      })
      clearTimeout(timeoutId)
      if (res.ok) {
        activeApiBase = url
        return {
          online: true,
          message: `Backend connected (${new URL(url).host})`,
        }
      }
    } catch {
      // Try next candidate
    }
  }

  return {
    online: false,
    message: "Backend offline — start the server: cd backend && uvicorn main:app --reload",
  }
}

// ─── Dataset Upload ────────────────────────────────────────────────────────

export async function uploadDatasetApi(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<UploadResponse> {
  if (onProgress) onProgress(20)

  const formData = new FormData()
  formData.append("file", file)

  if (onProgress) onProgress(50)

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 10000)

  try {
    const res = await fetch(`${activeApiBase}/upload`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (res.ok) {
      const data = await res.json()
      if (onProgress) onProgress(100)
      return data as UploadResponse
    }

    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(`Upload failed: ${err.detail || res.statusText}`)
  } catch (e: unknown) {
    clearTimeout(timeoutId)
    if ((e as Error).name === "AbortError") {
      throw new BackendOfflineError()
    }
    if (e instanceof BackendOfflineError) throw e
    // If the backend is unreachable (TypeError: Failed to fetch)
    const msg = (e as Error).message || ""
    if (msg.includes("Failed to fetch") || msg.includes("fetch")) {
      throw new BackendOfflineError()
    }
    throw e
  }
}

// ─── Process Privacy ───────────────────────────────────────────────────────

export async function processPrivacyApi(
  params: PrivacyProcessParams,
): Promise<PrivacyProcessResponse> {
  const controller = new AbortController()
  // Long timeout — experiments can take up to 20 minutes for Random Forest
  const timeoutId = setTimeout(() => controller.abort(), 1200000) // 20 minutes

  try {
    const res = await fetch(`${activeApiBase}/process-privacy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (res.ok) {
      const data = await res.json()
      return data as PrivacyProcessResponse
    }

    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(`Experiment failed: ${err.detail || res.statusText}`)
  } catch (e: unknown) {
    clearTimeout(timeoutId)
    if ((e as Error).name === "AbortError") {
      throw new Error("Experiment timed out (>5 minutes). Try FAST mode or fewer shadow models.")
    }
    const msg = (e as Error).message || ""
    if (msg.includes("Failed to fetch") || msg.includes("fetch")) {
      throw new BackendOfflineError()
    }
    throw e
  }
}

// ─── Batch Experiment Runner ───────────────────────────────────────────────

export async function runExperimentBatchApi(
  req: ExperimentRunRequest,
): Promise<{ job_id: string; message: string; config: Record<string, unknown> }> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 5000)

  try {
    const res = await fetch(`${activeApiBase}/run-experiment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (res.ok) {
      return await res.json()
    }
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(`Batch run failed: ${err.detail || res.statusText}`)
  } catch (e: unknown) {
    clearTimeout(timeoutId)
    const msg = (e as Error).message || ""
    if (msg.includes("Failed to fetch") || msg.includes("fetch")) {
      throw new BackendOfflineError()
    }
    throw e
  }
}

// ─── Experiment History ────────────────────────────────────────────────────

export async function getExperimentsApi(): Promise<ExperimentRecord[]> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 5000)

  try {
    const res = await fetch(`${activeApiBase}/experiments`, {
      method: "GET",
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (res.ok) {
      return (await res.json()) as ExperimentRecord[]
    }
    throw new Error(`Failed to fetch experiments: ${res.statusText}`)
  } catch (e: unknown) {
    clearTimeout(timeoutId)
    const msg = (e as Error).message || ""
    if (msg.includes("Failed to fetch") || msg.includes("fetch") || (e as Error).name === "AbortError") {
      throw new BackendOfflineError()
    }
    throw e
  }
}

// ─── Experiment Results ────────────────────────────────────────────────────

export async function getExperimentResultsApi(
  experimentId: string,
): Promise<PrivacyProcessResponse> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 5000)

  try {
    const res = await fetch(`${activeApiBase}/results/${experimentId}`, {
      method: "GET",
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (res.ok) {
      return (await res.json()) as PrivacyProcessResponse
    }
    if (res.status === 404) {
      throw new Error(`Experiment ${experimentId} not found. Run it first.`)
    }
    throw new Error(`Failed to fetch results: ${res.statusText}`)
  } catch (e: unknown) {
    clearTimeout(timeoutId)
    const msg = (e as Error).message || ""
    if (msg.includes("Failed to fetch") || (e as Error).name === "AbortError") {
      throw new BackendOfflineError()
    }
    throw e
  }
}

// ─── Download URLs ─────────────────────────────────────────────────────────

export function downloadDatasetUrl(experimentId: string): string {
  return `${activeApiBase}/download-dataset/${experimentId}`
}

export function downloadModelUrl(experimentId: string): string {
  return `${activeApiBase}/download-model/${experimentId}`
}

export function downloadFile(url: string, filename: string): void {
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// ─── Formatting Helpers ────────────────────────────────────────────────────

/** Format a metric value as percentage, or "--" if null/undefined. */
export function fmtPct(val: number | null | undefined): string {
  if (val === null || val === undefined) return "--"
  return `${(val * 100).toFixed(1)}%`
}

/** Format a metric value with given decimal places, or "--" if null/undefined. */
export function fmtNum(val: number | null | undefined, decimals = 4): string {
  if (val === null || val === undefined) return "--"
  return val.toFixed(decimals)
}

/** Format epsilon with label. */
export function fmtEpsilon(val: number | null | undefined): string {
  if (val === null || val === undefined) return "--"
  if (val === Infinity || val > 1e10) return "∞ (non-private)"
  return val.toFixed(4)
}
