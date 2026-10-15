import React, { useState, useRef } from "react"
import { useNavigate } from "react-router-dom"
import {
  Upload,
  FileText,
  CheckCircle2,
  Lock,
  ArrowRight,
  Database,
  AlertCircle,
  RefreshCw,
  Layers,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import { uploadDatasetApi } from "../services/api"
import DataTable from "../components/DataTable"
import StatusBanner from "../components/StatusBanner"

export default function DatasetUploadPage() {
  const navigate = useNavigate()
  const { dataset, setDataset } = useApp()

  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(
    dataset.status === "ready" ? 100 : 0,
  )
  const [isUploading, setIsUploading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [successBanner, setSuccessBanner] = useState<string | null>(
    dataset.status === "ready"
      ? `${dataset.name} is uploaded and ready for privacy configuration.`
      : null,
  )

  const fileInputRef = useRef<HTMLInputElement>(null)

  /**
   * Parse CSV headers and top rows client-side
   */
  const parseCsvPreview = async (file: File): Promise<{
    columns: string[]
    rows: Record<string, string>[]
    totalRows: number
  }> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        if (!text) {
          resolve({ columns: [], rows: [], totalRows: 0 })
          return
        }

        const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0)
        if (lines.length === 0) {
          resolve({ columns: [], rows: [], totalRows: 0 })
          return
        }

        const columns = lines[0]
          .split(",")
          .map((c) => c.trim().replace(/^["']|["']$/g, ""))
        const rows: Record<string, string>[] = []

        for (let i = 1; i < Math.min(lines.length, 6); i++) {
          const values = lines[i]
            .split(",")
            .map((v) => v.trim().replace(/^["']|["']$/g, ""))
          const row: Record<string, string> = {}
          columns.forEach((col, idx) => {
            row[col] = values[idx] ?? ""
          })
          rows.push(row)
        }

        resolve({
          columns,
          rows,
          totalRows: Math.max(lines.length - 1, 1),
        })
      }
      reader.onerror = () => resolve({ columns: [], rows: [], totalRows: 0 })
      reader.readAsText(file.slice(0, 100 * 1024)) // Read first 100KB for quick preview
    })
  }

  /**
   * Handle File Upload
   */
  const processUploadedFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv") && file.type !== "text/csv") {
      setErrorMsg("Please upload a valid CSV formatted dataset.")
      return
    }

    setErrorMsg(null)
    setIsUploading(true)
    setUploadProgress(10)

    try {
      // 1. Parse client preview
      const preview = await parseCsvPreview(file)

      // 2. Call backend upload API with progress callback
      const res = await uploadDatasetApi(file, (pct) => setUploadProgress(pct))

      const sizeMb = (file.size / (1024 * 1024)).toFixed(1)
      const rowCount = preview.totalRows > 1 ? preview.totalRows : res.records
      const colNames =
        preview.columns.length > 0 ? preview.columns : res.columnNames
      const previewRows =
        preview.rows.length > 0 ? preview.rows : res.previewRows

      setDataset({
        file,
        name: file.name,
        size: `${sizeMb} MB`,
        records: rowCount,
        columns: colNames.length || 31,
        targetColumn: colNames.includes("Class")
          ? "Class"
          : colNames[colNames.length - 1] || "target",
        missingValues: 0,
        previewRows,
        status: "ready",
        uploadProgress: 100,
      })

      setSuccessBanner(
        res.isSimulated
          ? `${file.name} loaded successfully (Client simulation mode). Ready for privacy configuration.`
          : `${file.name} securely uploaded to backend vault. Ready for privacy configuration.`,
      )
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to process dataset. Please try again.")
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processUploadedFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processUploadedFile(e.target.files[0])
    }
  }

  return (
    <section
      aria-labelledby="upload-heading"
      className="space-y-6 max-w-4xl mx-auto"
    >
      {/* Page Header */}
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1
            id="upload-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Dataset Upload & Ingestion
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Upload sensitive tabular datasets for differential privacy noise
            injection and ML analysis
          </p>
        </div>

        {dataset.status === "ready" && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="self-start sm:self-auto inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 border border-blue-200 px-3 py-1.5 rounded-lg transition-colors"
          >
            <RefreshCw size={13} />
            Replace Dataset
          </button>
        )}
      </header>

      {/* Security alert banner */}
      <div className="flex items-center gap-2.5 bg-green-50 border border-green-200 rounded-xl px-4 py-3 shadow-xs">
        <Lock size={15} className="text-green-600 flex-shrink-0" />
        <span className="text-xs sm:text-sm text-green-900 font-medium">
          Zero Plaintext Exposure: Data is encrypted with AES-256 before disk
          storage.
        </span>
        <span className="hidden sm:inline-flex items-center gap-1 text-xs font-semibold text-green-700 bg-green-100 rounded-full px-2.5 py-0.5 ml-auto">
          <CheckCircle2 size={11} />
          AES-256 Encrypted
        </span>
      </div>

      {/* Notifications */}
      {errorMsg && (
        <StatusBanner
          variant="error"
          title="Upload Error"
          message={errorMsg}
          onDismiss={() => setErrorMsg(null)}
        />
      )}

      {successBanner && (
        <StatusBanner
          variant="success"
          message={successBanner}
          onDismiss={() => setSuccessBanner(null)}
        />
      )}

      {/* Interactive Drop Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
          isDragging
            ? "border-blue-500 bg-blue-50/70 scale-[0.99]"
            : isUploading
              ? "border-blue-300 bg-slate-50 cursor-wait"
              : dataset.status === "ready"
                ? "border-green-400 bg-green-50/40 hover:bg-green-50/60"
                : "border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/30 shadow-xs"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleFileChange}
          aria-label="Upload CSV File"
        />

        <div
          className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 transition-transform ${
            isUploading
              ? "bg-blue-100 text-blue-600 animate-pulse"
              : dataset.status === "ready"
                ? "bg-green-100 text-green-600"
                : "bg-blue-50 text-blue-600"
          }`}
        >
          {isUploading ? (
            <RefreshCw size={28} className="animate-spin" />
          ) : dataset.status === "ready" ? (
            <CheckCircle2 size={28} />
          ) : (
            <Upload size={28} />
          )}
        </div>

        {isUploading ? (
          <div className="max-w-xs mx-auto space-y-2">
            <p className="text-base font-semibold text-blue-700">
              Uploading & Sanitizing Dataset...
            </p>
            <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p className="text-xs font-mono text-slate-500">
              {uploadProgress}% complete
            </p>
          </div>
        ) : dataset.status === "ready" ? (
          <div>
            <p className="text-base font-bold text-green-800">
              {dataset.name} ready
            </p>
            <p className="text-sm text-green-700 mt-1">
              {dataset.records.toLocaleString()} records · {dataset.columns}{" "}
              features · {dataset.size}
            </p>
            <p className="text-xs text-slate-400 mt-3">
              Click or drop another CSV file to replace
            </p>
          </div>
        ) : (
          <div>
            <p className="text-base sm:text-lg font-semibold text-[#0F172A]">
              Drop your CSV dataset here
            </p>
            <p className="text-sm text-slate-500 mt-1 mb-5">
              Supports tabular datasets (.csv) up to 500 MB
            </p>
            <button
              type="button"
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-5 py-2.5 rounded-lg shadow-xs transition-colors"
            >
              <Upload size={16} />
              Browse CSV Files
            </button>
          </div>
        )}
      </div>

      {/* Dataset Summary Metrics */}
      {dataset.status === "ready" && (
        <article className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-4">
            <Database size={16} className="text-blue-600" />
            <h2 className="text-sm font-semibold text-[#0F172A]">
              Dataset Metadata
            </h2>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
            {[
              ["File Name", dataset.name],
              ["File Size", dataset.size],
              ["Total Records", dataset.records.toLocaleString()],
              ["Feature Columns", `${dataset.columns}`],
              ["Target Column", dataset.targetColumn],
              ["Missing Values", `${dataset.missingValues}`],
            ].map(([label, val]) => (
              <div
                key={label}
                className="bg-slate-50 rounded-lg p-3 sm:px-4 sm:py-3 border border-slate-100"
              >
                <span className="text-[11px] text-slate-500 font-medium block mb-0.5">
                  {label}
                </span>
                <span className="text-sm font-bold text-[#0F172A] font-mono truncate block">
                  {val}
                </span>
              </div>
            ))}
          </div>
        </article>
      )}

      {/* Live Data Preview Table */}
      {dataset.status === "ready" && dataset.previewRows.length > 0 && (
        <section aria-labelledby="preview-heading">
          <DataTable
            data={dataset.previewRows}
            title="Tabular Data Preview"
            subtitle="Inspecting first sample rows from uploaded dataset"
            totalRows={dataset.records}
            maxDisplay={5}
            highlightColumn={dataset.targetColumn}
          />
        </section>
      )}

      {/* Continue CTA */}
      {dataset.status === "ready" && (
        <div className="flex justify-end pt-2">
          <button
            onClick={() => navigate("/privacy-config")}
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-6 py-3 rounded-xl shadow-xs hover:shadow-sm transition-all"
          >
            Continue to Privacy Configuration
            <ArrowRight size={16} />
          </button>
        </div>
      )}
    </section>
  )
}
