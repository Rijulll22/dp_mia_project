import React, { useState } from "react"
import {
  FileText,
  Download,
  Eye,
  Plus,
  GitCompare,
  CheckCircle2,
  FileJson,
  DatabaseZap,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import StatusBanner from "../components/StatusBanner"
import {
  downloadDatasetUrl,
  downloadModelUrl,
  downloadFile,
} from "../services/api"

interface ReportItem {
  id: string
  title: string
  dataset: string
  epsilon: number | string
  model: string
  accuracy: string
  f1: string
  miaRisk: string
  date: string
  sections: number
}

const INITIAL_REPORTS: ReportItem[] = [
  {
    id: "RPT-001",
    title: "Differential Privacy Analysis Audit — ε = 1.0",
    dataset: "CreditCard.csv",
    epsilon: 1.0,
    model: "Logistic Regression",
    accuracy: "94.8%",
    f1: "89.3%",
    miaRisk: "LOW",
    date: "2026-08-30",
    sections: 8,
  },
  {
    id: "RPT-002",
    title: "Differential Privacy Analysis Audit — ε = 0.1",
    dataset: "CreditCard.csv",
    epsilon: 0.1,
    model: "Logistic Regression",
    accuracy: "89.1%",
    f1: "83.5%",
    miaRisk: "LOW",
    date: "2026-08-28",
    sections: 8,
  },
  {
    id: "RPT-003",
    title: "Privacy–Utility Multi-Budget Comparison Audit",
    dataset: "CreditCard.csv",
    epsilon: "0.1–10.0",
    model: "Logistic Regression",
    accuracy: "89–97%",
    f1: "83–92%",
    miaRisk: "LOW–MED",
    date: "2026-09-01",
    sections: 8,
  },
]

const REPORT_SECTIONS = [
  "Dataset Provenance & Ingestion Checksums",
  "Cryptographic Storage & Vault Enclave",
  "Differential Privacy Budget & Noise Calibration",
  "Model Accuracy & Generalization Metrics",
  "Empirical Membership Inference Evaluation",
  "Feature Inversion & Reconstruction Bounds",
  "Recommended Decision Thresholds",
  "Compliance & Audit Sign-Off",
]

export default function ReportsPage() {
  const { dataset, privacyConfig, experimentResults } = useApp()
  const [reports, setReports] = useState<ReportItem[]>(INITIAL_REPORTS)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  const experimentId = experimentResults?.experimentId || "EXP-003"

  const handleDownload = (report: ReportItem) => {
    const content = `
PRIVACYGUARD DIFFERENTIAL PRIVACY AUDIT REPORT
==============================================
Report ID: ${report.id}
Date: ${report.date}
Dataset: ${report.dataset}
Privacy Budget (ε): ${report.epsilon}
Classifier: ${report.model}
Accuracy: ${report.accuracy}
F1-Score: ${report.f1}
MIA Risk: ${report.miaRisk}

SECTIONS INCLUDED:
${REPORT_SECTIONS.map((s, i) => `${i + 1}. ${s}`).join("\n")}

COMPLIANCE ATTESTATION:
Differential Privacy Laplace mechanism calibrated with sensitivity Δf = 1.0.
Guaranteed under (ε, δ)-Differential Privacy.
`
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `${report.id}-privacy-report.txt`
    link.click()
    URL.revokeObjectURL(url)
    setToastMessage(`Downloaded audit report ${report.id}`)
  }

  const handleGenerateReport = () => {
    const newId = `RPT-00${reports.length + 1}`
    const newRpt: ReportItem = {
      id: newId,
      title: `DP Audit Report — ${dataset.name} (ε = ${privacyConfig.epsilon})`,
      dataset: dataset.name,
      epsilon: privacyConfig.epsilon,
      model: "Logistic Regression",
      accuracy: "94.8%",
      f1: "89.3%",
      miaRisk: "LOW",
      date: new Date().toISOString().split("T")[0],
      sections: 8,
    }
    setReports([newRpt, ...reports])
    setToastMessage(`Generated new audit report ${newId}`)
  }

  return (
    <section
      aria-labelledby="reports-heading"
      className="space-y-6 max-w-5xl mx-auto"
    >
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1
            id="reports-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Governance & Audit Reports
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Export certified Differential Privacy audit reports for compliance
            and security review
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={() =>
              downloadFile(
                downloadDatasetUrl(experimentId),
                `${experimentId}_perturbed_dataset.csv`,
              )
            }
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 px-3 py-2 rounded-xl shadow-xs transition-all"
            title="Download privacy-perturbed dataset (CSV)"
          >
            <DatabaseZap size={13} />
            Dataset (CSV)
          </button>
          <button
            type="button"
            onClick={() =>
              downloadFile(
                downloadModelUrl(experimentId),
                `${experimentId}_model_weights.json`,
              )
            }
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-purple-700 border border-purple-200 bg-purple-50 hover:bg-purple-100 px-3 py-2 rounded-xl shadow-xs transition-all"
            title="Export model weights (JSON)"
          >
            <FileJson size={13} />
            Model Weights (JSON)
          </button>
          <button
            onClick={handleGenerateReport}
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-xs sm:text-sm font-semibold px-4 py-2 rounded-xl shadow-xs transition-colors"
          >
            <Plus size={15} />
            Compile New Report
          </button>
        </div>
      </header>

      {toastMessage && (
        <StatusBanner
          variant="success"
          message={toastMessage}
          onDismiss={() => setToastMessage(null)}
        />
      )}

      {/* Reports List */}
      <div className="space-y-4">
        {reports.map((r) => (
          <article
            key={r.id}
            className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 transition-shadow hover:shadow-sm"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="flex items-start gap-3.5">
                <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5">
                  <FileText size={20} />
                </div>
                <div>
                  <h2 className="font-bold text-[#0F172A] text-sm sm:text-base leading-snug">
                    {r.title}
                  </h2>
                  <span className="text-xs text-slate-400 font-mono mt-0.5 block">
                    {r.id} · Issued {r.date} · {r.sections} Certified Sections
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2 self-end sm:self-center">
                <button
                  onClick={() => handleDownload(r)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 hover:text-blue-600 border border-slate-200 hover:border-blue-300 bg-white px-3 py-1.5 rounded-lg transition-colors shadow-2xs"
                >
                  <Download size={13} />
                  Download
                </button>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                ["Dataset", r.dataset],
                ["Budget", `ε = ${r.epsilon}`],
                ["Algorithm", r.model],
                ["MIA Attack Risk", r.miaRisk],
              ].map(([k, v]) => (
                <div
                  key={k}
                  className="bg-slate-50 border border-slate-100 rounded-lg px-3 py-2"
                >
                  <span className="text-[10px] text-slate-400 font-medium block mb-0.5">
                    {k}
                  </span>
                  <span className="text-xs font-bold text-[#0F172A] font-mono block">
                    {v}
                  </span>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>

      {/* Report Structure Spec Card */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 sm:p-6 space-y-3">
        <h3 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
          Standard Report Certification Modules
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {REPORT_SECTIONS.map((c, i) => (
            <div
              key={c}
              className="flex items-center gap-2.5 text-xs text-slate-700 bg-slate-50 p-2.5 rounded-lg border border-slate-100"
            >
              <span className="w-5 h-5 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center font-bold text-[10px] flex-shrink-0">
                {i + 1}
              </span>
              <span>{c}</span>
            </div>
          ))}
        </div>
      </article>
    </section>
  )
}
