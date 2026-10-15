import React, { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import {
  Eye,
  GitCompare,
  Trash2,
  Filter,
  Download,
  FileJson,
  RefreshCw,
  Loader2,
} from "lucide-react"
import {
  getExperimentsApi,
  downloadDatasetUrl,
  downloadModelUrl,
  downloadFile,
  type ExperimentRecord,
} from "../services/api"

const riskColors: Record<string, string> = {
  LOW: "text-emerald-700 bg-emerald-50 border-emerald-200",
  MEDIUM: "text-amber-700 bg-amber-50 border-amber-200",
  HIGH: "text-red-700 bg-red-50 border-red-200",
}

const statusColors: Record<string, string> = {
  Completed: "text-emerald-700 bg-emerald-50",
  Running: "text-blue-700 bg-blue-50",
  Failed: "text-red-700 bg-red-50",
}

const formatTimestamp = (isoString?: string): string => {
  if (!isoString) return "N/A"

  // Append 'Z' if the backend returned a naive string without UTC indicator
  const cleanIso = isoString.includes(" ")
    ? isoString.replace(" ", "T")
    : isoString
  const normalizedIso =
    cleanIso.endsWith("Z") || cleanIso.includes("+") ? cleanIso : `${cleanIso}Z`

  const date = new Date(normalizedIso)
  if (isNaN(date.getTime())) return isoString // Fallback if parsing fails

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  })
}

export default function ExperimentHistoryPage() {
  const navigate = useNavigate()
  const [filterStatus, setFilterStatus] = useState("All")
  const [rows, setRows] = useState<ExperimentRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [isLive, setIsLive] = useState(false)
  const [downloadingId, setDownloadingId] = useState<string | null>(null)

  const loadExperiments = async () => {
    setLoading(true)
    try {
      const data = await getExperimentsApi()
      setRows(data)
      // If any row has a timestamp from the DB it's a live result
      setIsLive(data.some((d) => !!d.timestamp))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadExperiments()
  }, [])

  const statuses = ["All", "Completed", "Running", "Failed"]
  const filtered =
    filterStatus === "All"
      ? rows
      : rows.filter((r) => r.status === filterStatus)

  const handleDownloadDataset = async (id: string) => {
    setDownloadingId(id)
    try {
      downloadFile(downloadDatasetUrl(id), `${id}_perturbed_dataset.csv`)
    } finally {
      setTimeout(() => setDownloadingId(null), 1200)
    }
  }

  const handleDownloadModel = async (id: string) => {
    setDownloadingId(id)
    try {
      downloadFile(downloadModelUrl(id), `${id}_model_weights.json`)
    } finally {
      setTimeout(() => setDownloadingId(null), 1200)
    }
  }

  return (
    <section
      aria-labelledby="history-heading"
      className="space-y-6 max-w-7xl mx-auto"
    >
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1
            id="history-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Differential Privacy Experiment History
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Audit trail and comparative ledger of all privacy-preserving
            training executions
            {isLive && (
              <span className="ml-2 inline-flex items-center gap-1 text-[10px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE — SQLite
              </span>
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={loadExperiments}
          disabled={loading}
          className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 border border-slate-200 bg-white hover:bg-slate-50 px-3 py-1.5 rounded-lg shadow-xs transition-all disabled:opacity-60"
        >
          {loading ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <RefreshCw size={13} />
          )}
          Refresh
        </button>
      </header>

      {/* Filters Toolbar */}
      <article className="flex flex-wrap items-center justify-between gap-3 bg-white rounded-2xl border border-slate-200 p-4 shadow-xs">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 font-semibold">
            <Filter size={14} className="text-slate-400" />
            <span>Filter Status:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {statuses.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setFilterStatus(s)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all ${
                  filterStatus === s
                    ? "bg-blue-600 text-white border-blue-600 shadow-2xs"
                    : "border-slate-200 text-slate-600 hover:border-slate-300 bg-slate-50"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          {filtered.length} of {rows.length} records
        </span>
      </article>

      {/* History Ledger Table */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        {loading && rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3 text-slate-400">
            <Loader2 size={28} className="animate-spin text-blue-500" />
            <span className="text-sm font-medium">
              Loading experiment history...
            </span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 text-left font-semibold uppercase tracking-wider">
                  {[
                    "ID",
                    "Dataset",
                    "Budget (ε)",
                    "Mechanism",
                    "Timestamp",
                    "Accuracy",
                    "F1-Score",
                    "Risk Rating",
                    "Status",
                    "Actions",
                  ].map((h) => (
                    <th key={h} className="px-4 py-3 whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {filtered.map((experiment) => (
                  <tr
                    key={experiment.id}
                    className="hover:bg-slate-50/60 transition-colors"
                  >
                    <td className="px-4 py-3 font-bold text-blue-700">
                      {experiment.id}
                    </td>
                    <td className="px-4 py-3 text-slate-800 font-sans">
                      {experiment.dataset}
                    </td>
                    <td className="px-4 py-3 font-bold text-[#0F172A]">
                      ε = {experiment.epsilon}
                    </td>
                    <td className="px-4 py-3 text-slate-600 font-sans">
                      {experiment.mechanism || "Laplace"}
                    </td>
                    <td className="px-4 py-3 text-slate-500">
                      {formatTimestamp(experiment.timestamp)}
                    </td>
                    <td className="px-4 py-3 font-bold text-[#0F172A]">
                      {experiment.accuracy}
                    </td>
                    <td className="px-4 py-3 font-bold text-[#0F172A]">
                      {experiment.f1}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`font-sans text-[10px] font-bold px-2.5 py-0.5 rounded-full border ${riskColors[experiment.privacyRisk]}`}
                      >
                        {experiment.privacyRisk}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`font-sans text-[10px] font-semibold px-2 py-0.5 rounded-full ${statusColors[experiment.status]}`}
                      >
                        {experiment.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => navigate("/model-analysis")}
                          className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="Inspect detailed analysis"
                          aria-label="Inspect detailed analysis"
                        >
                          <Eye size={13} />
                        </button>
                        <button
                          onClick={() => navigate("/privacy-utility")}
                          className="p-1.5 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors"
                          title="Compare in explorer"
                          aria-label="Compare in explorer"
                        >
                          <GitCompare size={13} />
                        </button>
                        <button
                          onClick={() => handleDownloadDataset(experiment.id)}
                          disabled={downloadingId === experiment.id}
                          className="p-1.5 text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors disabled:opacity-50"
                          title="Download perturbed dataset (CSV)"
                          aria-label="Download perturbed dataset (CSV)"
                        >
                          {downloadingId === experiment.id ? (
                            <Loader2 size={13} className="animate-spin" />
                          ) : (
                            <Download size={13} />
                          )}
                        </button>
                        <button
                          onClick={() => handleDownloadModel(experiment.id)}
                          disabled={downloadingId === experiment.id}
                          className="p-1.5 text-slate-400 hover:text-purple-600 hover:bg-purple-50 rounded-lg transition-colors disabled:opacity-50"
                          title="Export model weights (JSON)"
                          aria-label="Export model weights (JSON)"
                        >
                          <FileJson size={13} />
                        </button>
                        <button
                          onClick={() =>
                            setRows((r) =>
                              r.filter((x) => x.id !== experiment.id),
                            )
                          }
                          className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Remove from view"
                          aria-label="Remove from view"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 && (
              <div className="text-center py-12 text-slate-400 text-sm font-medium">
                No experiments match the selected filter.
              </div>
            )}
          </div>
        )}
      </article>
    </section>
  )
}
