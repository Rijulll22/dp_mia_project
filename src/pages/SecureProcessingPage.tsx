import React, { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import {
  CheckCircle2,
  Clock,
  Lock,
  Shield,
  Database,
  Cpu,
  ArrowRight,
  RefreshCw,
  AlertTriangle,
  Play,
  Activity,
  Layers,
  Check,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import { processPrivacyApi } from "../services/api"
import StatusBanner from "../components/StatusBanner"

const PIPELINE_STEPS = [
  {
    label: "Dataset Ingestion & Integrity Check",
    desc: "SHA-256 checksum verified against tampered records",
  },
  {
    label: "AES-256 Symmetric Encryption",
    desc: "Encrypted at rest before persistent storage",
  },
  {
    label: "Secure Vault Storage",
    desc: "Ciphertext persisted to isolated enclave storage",
  },
  {
    label: "Transient RAM Decryption",
    desc: "Decrypted solely within ephemeral runtime memory",
  },
  {
    label: "Data Cleaning & Validation",
    desc: "Outlier detection, null imputation, type validation",
  },
  {
    label: "Min-Max Feature Scaling",
    desc: "All features normalized to [0, 1] bounded range",
  },
  {
    label: "Differential Privacy Noise Injection",
    desc: "Laplace / Gaussian noise applied at calibrated scale",
  },
  {
    label: "Model Training (Privacy-Preserved)",
    desc: "Fraud detection classifier trained on perturbed features",
  },
  {
    label: "Model Utility Evaluation",
    desc: "Cross-validated Accuracy, Precision, Recall, F1, ROC-AUC",
  },
  {
    label: "Privacy Attack Simulation",
    desc: "Membership inference and attribute reconstruction attacks executed",
  },
  {
    label: "Privacy–Utility Frontier Modeling",
    desc: "Multi-budget empirical trade-off calculation",
  },
  {
    label: "Audit Artifacts & Results Generation",
    desc: "Secure signed report generated and persisted",
  },
]

export default function SecureProcessingPage() {
  const navigate = useNavigate()
  const {
    privacyConfig,
    dataset,
    experimentResults,
    setExperimentResults,
    lastRunResult,
    setLastRunResult,
    processingState,
    setProcessingState,
  } = useApp()

  const [currentStep, setCurrentStep] = useState<number>(
    processingState.currentStep || 0,
  )
  const [status, setStatus] =
    useState<"idle" | "processing" | "completed" | "error">(
      processingState.status || "idle",
    )
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [progressPct, setProgressPct] = useState<number>(
    processingState.progressPct || 0,
  )

  // Run the end-to-end processing pipeline
  const runAnalysis = async () => {
    setStatus("processing")
    setErrorMessage(null)
    setCurrentStep(0)
    setProgressPct(5)

    try {
      // Step-by-step visual animation through the pipeline
      for (let step = 0; step < PIPELINE_STEPS.length; step++) {
        setCurrentStep(step)
        const targetPct = Math.min(
          95,
          Math.round(((step + 1) / PIPELINE_STEPS.length) * 100),
        )
        setProgressPct(targetPct)
        await new Promise((resolve) => setTimeout(resolve, 380))
      }

      // Call the API endpoint
      const result = await processPrivacyApi({
        datasetId: dataset.name,
        epsilon: privacyConfig.epsilon,
        delta: privacyConfig.delta,
        sensitivity: privacyConfig.sensitivity,
        mechanism: privacyConfig.mechanism,
        mechanism_type: privacyConfig.mechanismType,
        classifier_type: privacyConfig.classifierType,
      })

      setExperimentResults(result)
      setLastRunResult(result)
      setCurrentStep(PIPELINE_STEPS.length)
      setProgressPct(100)
      setStatus("completed")
      setProcessingState({
        status: "completed",
        currentStep: PIPELINE_STEPS.length,
        progressPct: 100,
        error: null,
      })
    } catch (err: any) {
      setStatus("error")
      setErrorMessage(
        err.message || "Error occurred during differential privacy execution.",
      )
      setProcessingState({
        status: "error",
        currentStep,
        progressPct,
        error: err.message,
      })
    }
  }

  const getStepState = (index: number) => {
    if (status === "completed" || index < currentStep) return "done"
    if (status === "processing" && index === currentStep) return "active"
    if (status === "error" && index === currentStep) return "error"
    return "pending"
  }

  return (
    <section
      aria-labelledby="secure-heading"
      className="space-y-6 max-w-5xl mx-auto"
    >
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1
            id="secure-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Secure Processing Pipeline
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Privacy-preserving execution with {privacyConfig.mechanism}{" "}
            mechanism (ε = {privacyConfig.epsilon})
          </p>
        </div>

        <div className="flex items-center gap-3">
          {status === "idle" && (
            <button
              onClick={runAnalysis}
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-5 py-2.5 rounded-xl shadow-xs transition-colors"
            >
              <Play size={16} fill="currentColor" />
              Start Secure Pipeline
            </button>
          )}

          {status === "processing" && (
            <div className="inline-flex items-center gap-2 text-xs font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-4 py-2 rounded-xl">
              <RefreshCw size={14} className="animate-spin text-blue-600" />
              Executing Pipeline...
            </div>
          )}

          {status === "completed" && (
            <button
              onClick={runAnalysis}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 px-3 py-2 rounded-xl transition-colors shadow-2xs"
            >
              <RefreshCw size={13} />
              Re-run Pipeline
            </button>
          )}

          {status === "error" && (
            <button
              onClick={runAnalysis}
              className="inline-flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white text-sm font-semibold px-4 py-2 rounded-xl shadow-xs transition-colors"
            >
              <RefreshCw size={14} />
              Retry Execution
            </button>
          )}
        </div>
      </header>

      {/* Error / Status Notifications */}
      {status === "error" && errorMessage && (
        <StatusBanner
          variant="error"
          title="Pipeline Execution Failed"
          message={errorMessage}
          actionText="Try again"
          onAction={runAnalysis}
        />
      )}

      {/* Progress Bar Container */}
      <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between text-xs font-semibold">
          <span className="text-[#0F172A] flex items-center gap-2">
            <Activity
              size={15}
              className={
                status === "processing"
                  ? "text-blue-600 animate-pulse"
                  : "text-slate-400"
              }
            />
            Pipeline Status: {status.toUpperCase()}
          </span>
          <span className="font-mono text-blue-700">{progressPct}%</span>
        </div>

        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              status === "error"
                ? "bg-red-500"
                : status === "completed"
                  ? "bg-emerald-500"
                  : "bg-blue-600"
            }`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </article>

      {/* Main Grid: Pipeline Steps & System Architecture */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Step Progression List (2 Cols) */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs space-y-3">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
              Execution Trace ({PIPELINE_STEPS.length} Stages)
            </h2>
            <span className="text-xs text-slate-400 font-mono">
              ε={privacyConfig.epsilon} · S={privacyConfig.sensitivity}
            </span>
          </div>

          <div className="space-y-2">
            {PIPELINE_STEPS.map((step, i) => {
              const st = getStepState(i)

              return (
                <div
                  key={step.label}
                  className={`flex items-start gap-3 p-3 rounded-xl transition-all duration-200 ${
                    st === "active"
                      ? "bg-blue-50 border border-blue-200 ring-1 ring-blue-300 shadow-2xs"
                      : st === "done"
                        ? "bg-slate-50/70 border border-slate-100"
                        : st === "error"
                          ? "bg-red-50 border border-red-200"
                          : "opacity-40 border border-transparent"
                  }`}
                >
                  <div className="mt-0.5 flex-shrink-0">
                    {st === "done" ? (
                      <div className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
                        <Check size={12} />
                      </div>
                    ) : st === "active" ? (
                      <RefreshCw
                        size={18}
                        className="text-blue-600 animate-spin"
                      />
                    ) : st === "error" ? (
                      <AlertTriangle size={18} className="text-red-600" />
                    ) : (
                      <Clock size={16} className="text-slate-300" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div
                      className={`text-xs sm:text-sm font-semibold ${
                        st === "done"
                          ? "text-slate-800"
                          : st === "active"
                            ? "text-blue-700"
                            : st === "error"
                              ? "text-red-700"
                              : "text-slate-400"
                      }`}
                    >
                      {i + 1}. {step.label}
                    </div>
                    <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">
                      {step.desc}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Completion Callout */}
          {status === "completed" && (
            <div className="mt-6 p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex flex-wrap items-center justify-between gap-3 shadow-2xs">
              <div className="flex items-center gap-2.5">
                <CheckCircle2 size={20} className="text-emerald-600" />
                <div>
                  <span className="font-bold text-emerald-900 block text-sm">
                    Differential Privacy Processing Complete
                  </span>
                  <span className="text-xs text-emerald-700">
                    Model trained & privacy attack risks evaluated
                  </span>
                </div>
              </div>

              <button
                onClick={() => navigate("/model-analysis")}
                className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs sm:text-sm font-semibold px-4 py-2 rounded-lg shadow-xs transition-colors"
              >
                Inspect Model Analysis
                <ArrowRight size={15} />
              </button>
            </div>
          )}
        </div>

        {/* Security & Enclave Info (1 Col) */}
        <div className="space-y-6">
          {/* Security Guarantees */}
          <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
              Enclave Security Status
            </h2>

            <div className="space-y-3">
              {[
                {
                  icon: Lock,
                  label: "Data at Rest",
                  value: "Encrypted",
                  color: "emerald",
                },
                {
                  icon: Database,
                  label: "Persistent Plaintext",
                  value: "None",
                  color: "emerald",
                },
                {
                  icon: Cpu,
                  label: "Processing Enclave",
                  value: "Volatile RAM",
                  color: "blue",
                },
                {
                  icon: Shield,
                  label: "Differential Privacy",
                  value: "Laplace Active",
                  color: "blue",
                },
              ].map(({ icon: Icon, label, value, color }) => (
                <div
                  key={label}
                  className="flex items-center justify-between text-xs py-1 border-b border-slate-100 last:border-0"
                >
                  <div className="flex items-center gap-2">
                    <Icon
                      size={14}
                      className={
                        color === "emerald"
                          ? "text-emerald-600"
                          : "text-blue-600"
                      }
                    />
                    <span className="text-slate-600">{label}</span>
                  </div>
                  <span
                    className={`font-semibold px-2 py-0.5 rounded-full ${
                      color === "emerald"
                        ? "text-emerald-700 bg-emerald-50"
                        : "text-blue-700 bg-blue-50"
                    }`}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </div>

            <div className="pt-2 border-t border-slate-100 flex justify-between items-center text-xs">
              <span className="text-slate-500">Configured Budget (ε):</span>
              <span className="font-mono font-bold text-blue-700">
                {privacyConfig.epsilon}
              </span>
            </div>
          </article>

          {/* Architecture Visual Stack */}
          <article className="bg-[#0F172A] text-white rounded-2xl p-5 shadow-xs space-y-3">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
              Zero-Trust Architecture
            </span>

            <div className="space-y-1.5 text-xs font-mono">
              {[
                ["Frontend Application", "text-blue-400 bg-blue-900/30"],
                ["FastAPI Gateway", "text-slate-300 bg-slate-800"],
                ["AES-256 Vault", "text-emerald-400 bg-emerald-900/30"],
                ["Transient RAM Buffer", "text-amber-400 bg-amber-900/30"],
                ["Laplace Noise Injector", "text-blue-400 bg-blue-900/30"],
                ["ML Training Enclave", "text-violet-400 bg-violet-900/30"],
                [
                  "Privacy Decision Support",
                  "text-emerald-400 bg-emerald-900/30",
                ],
              ].map(([layer, style]) => (
                <div
                  key={layer}
                  className={`px-3 py-1.5 rounded-lg font-medium border border-white/5 ${style}`}
                >
                  {layer}
                </div>
              ))}
            </div>
          </article>
        </div>
      </div>
    </section>
  )
}
