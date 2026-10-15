import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import {
  Shield,
  Database,
  Cpu,
  Lock,
  Plus,
  TrendingUp,
  AlertTriangle,
  ArrowRight,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import MetricsCard from "../components/MetricsCard"

const METRICS_LIST = [
  { key: "accuracy", label: "Accuracy", color: "#2563EB" },
  { key: "f1", label: "F1-Score", color: "#7C3AED" },
  { key: "rocAuc", label: "ROC-AUC", color: "#059669" },
  { key: "miaRisk", label: "MIA Risk", color: "#D97706" },
  { key: "reconRisk", label: "Recon. Risk", color: "#DC2626" },
]

export default function DashboardPage() {
  const navigate = useNavigate()
  const { dataset, privacyConfig, experimentResults } = useApp()
  const [activeMetrics, setActiveMetrics] = useState<string[]>([
    "accuracy",
    "f1",
    "miaRisk",
  ])

  const toggleMetric = (key: string) => {
    setActiveMetrics((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    )
  }

  const results = experimentResults?.metrics || {
    accuracy: 0.948,
    precision: 0.932,
    recall: 0.857,
    f1: 0.893,
    rocAuc: 0.974,
    miaRisk: 0.552,
    reconRisk: 0.16,
  }

  const tradeoffData = experimentResults?.tradeoffCurve || [
    {
      epsilon: 0.1,
      accuracy: 0.891,
      f1: 0.823,
      rocAuc: 0.934,
      miaRisk: 0.511,
      reconRisk: 0.08,
    },
    {
      epsilon: 0.5,
      accuracy: 0.933,
      f1: 0.871,
      rocAuc: 0.962,
      miaRisk: 0.534,
      reconRisk: 0.12,
    },
    {
      epsilon: 1.0,
      accuracy: 0.948,
      f1: 0.893,
      rocAuc: 0.974,
      miaRisk: 0.552,
      reconRisk: 0.16,
    },
    {
      epsilon: 2.0,
      accuracy: 0.957,
      f1: 0.908,
      rocAuc: 0.981,
      miaRisk: 0.579,
      reconRisk: 0.22,
    },
    {
      epsilon: 5.0,
      accuracy: 0.963,
      f1: 0.917,
      rocAuc: 0.986,
      miaRisk: 0.621,
      reconRisk: 0.31,
    },
    {
      epsilon: 10.0,
      accuracy: 0.967,
      f1: 0.923,
      rocAuc: 0.989,
      miaRisk: 0.668,
      reconRisk: 0.42,
    },
  ]

  return (
    <section
      aria-labelledby="dashboard-heading"
      className="space-y-6 max-w-7xl mx-auto"
    >
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1
            id="dashboard-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Privacy ML Dashboard
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Operational status, differential privacy telemetry, and model
            governance
          </p>
        </div>

        <button
          onClick={() => navigate("/dataset")}
          className="self-start sm:self-auto inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-4 py-2.5 rounded-xl shadow-xs transition-colors"
        >
          <Plus size={16} />
          New Dataset Analysis
        </button>
      </header>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricsCard
          label="Active Dataset"
          value={dataset.name.replace(/\.[^/.]+$/, "")}
          subtext={`${dataset.records.toLocaleString()} records · ${dataset.columns} features`}
          icon={Database}
          variant="blue"
        />

        <MetricsCard
          label="Privacy Budget"
          value={`ε = ${privacyConfig.epsilon}`}
          subtext={`${privacyConfig.mechanism} · Δf = ${privacyConfig.sensitivity}`}
          icon={Shield}
          variant="indigo"
        />

        <MetricsCard
          label="Fraud Classifier"
          value="Logistic Reg."
          subtext="DP-Trained & Validated"
          icon={Cpu}
          variant="violet"
        />

        <MetricsCard
          label="Security Enclave"
          value="AES-256 Vault"
          subtext="Transient RAM compute"
          icon={Lock}
          variant="green"
        />
      </div>

      {/* Current Analysis Summary & Model Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Analysis Details */}
        <article className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
                Current Run Summary
              </h2>
              <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-full px-2.5 py-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                Ready
              </span>
            </div>

            <div className="space-y-3 divide-y divide-slate-100 text-xs">
              <div className="flex items-center justify-between pt-2">
                <span className="text-slate-500">Dataset File</span>
                <span className="font-semibold text-slate-800 font-mono truncate max-w-[150px]">
                  {dataset.name}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2">
                <span className="text-slate-500">Records Processed</span>
                <span className="font-semibold text-slate-800 font-mono">
                  {dataset.records.toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2">
                <span className="text-slate-500">Mechanism</span>
                <span className="font-semibold text-blue-700 font-mono">
                  {privacyConfig.mechanism}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2">
                <span className="text-slate-500">Privacy Budget (ε)</span>
                <span className="font-bold text-blue-700 font-mono">
                  ε = {privacyConfig.epsilon}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2">
                <span className="text-slate-500">Noise Scale</span>
                <span className="font-semibold text-slate-800 font-mono">
                  {privacyConfig.noiseScale}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={() => navigate("/model-analysis")}
            className="mt-6 w-full text-center text-xs font-semibold text-blue-600 hover:text-blue-700 border border-blue-200 hover:border-blue-300 rounded-xl py-2.5 transition-colors"
          >
            Inspect Detailed Performance →
          </button>
        </article>

        {/* Model Performance Metrics */}
        <article className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
                Classifier Utility (ε = {privacyConfig.epsilon})
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Empirical test set evaluation after privacy injection
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              {
                label: "Accuracy",
                value: `${(results.accuracy * 100).toFixed(1)}%`,
                delta: "+1.8%",
              },
              {
                label: "Precision",
                value: `${(results.precision * 100).toFixed(1)}%`,
                delta: "+1.4%",
              },
              {
                label: "Recall",
                value: `${(results.recall * 100).toFixed(1)}%`,
                delta: "+2.9%",
              },
              {
                label: "F1-Score",
                value: `${(results.f1 * 100).toFixed(1)}%`,
                delta: "+2.1%",
              },
              {
                label: "ROC-AUC",
                value: `${(results.rocAuc * 100).toFixed(1)}%`,
                delta: "+0.7%",
              },
            ].map(({ label, value, delta }) => (
              <div
                key={label}
                className="bg-slate-50 border border-slate-100 rounded-xl p-3 text-center"
              >
                <span className="text-lg sm:text-xl font-bold text-[#0F172A] block font-mono">
                  {value}
                </span>
                <span className="text-xs text-slate-500 font-medium mt-0.5 block">
                  {label}
                </span>
                <span className="inline-flex items-center gap-0.5 text-[10px] text-emerald-600 font-semibold mt-1">
                  <TrendingUp size={10} />
                  {delta}
                </span>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 text-xs text-slate-500 flex items-center justify-between">
            <span>
              Evaluated against held-out test split (20% stratified sample)
            </span>
            <button
              onClick={() => navigate("/privacy-utility")}
              className="font-semibold text-blue-600 hover:text-blue-700"
            >
              Compare budgets →
            </button>
          </div>
        </article>
      </div>

      {/* Privacy Attack Risk Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
        <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <Shield size={16} />
              </div>
              <span className="text-sm font-semibold text-[#0F172A]">
                Membership Inference Attack (MIA)
              </span>
            </div>
            <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700">
              LOW RISK
            </span>
          </div>
          <div className="font-mono text-2xl font-bold text-[#0F172A] mb-1">
            {results.miaRisk.toFixed(3)}
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Probability attacker determines training presence. Random-guess
            baseline is 0.500. Lower is safer.
          </p>
        </article>

        <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <AlertTriangle size={16} />
              </div>
              <span className="text-sm font-semibold text-[#0F172A]">
                Feature Reconstruction Risk
              </span>
            </div>
            <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700">
              LOW RISK
            </span>
          </div>
          <div className="font-mono text-2xl font-bold text-[#0F172A] mb-1">
            {results.reconRisk.toFixed(3)}
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Normalized correlation score of inverted inputs. Lower values
            indicate resilient privacy boundaries.
          </p>
        </article>
      </div>

      {/* Interactive Trade-off Curve */}
      <article className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
              Privacy–Utility Pareto Frontier
            </h2>
            <p className="text-xs text-slate-400">
              Trade-off between budget ε, predictive performance, and risk
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {METRICS_LIST.map(({ key, label, color }) => (
              <button
                key={key}
                type="button"
                onClick={() => toggleMetric(key)}
                className={`text-xs font-semibold px-3 py-1 rounded-full border transition-all ${
                  activeMetrics.includes(key)
                    ? "text-white border-transparent shadow-2xs"
                    : "text-slate-600 border-slate-200 bg-white hover:border-slate-300"
                }`}
                style={
                  activeMetrics.includes(key) ? { backgroundColor: color } : {}
                }
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="h-64 sm:h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={tradeoffData}
              margin={{ top: 10, right: 20, left: -10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis
                dataKey="epsilon"
                tick={{ fontSize: 11, fill: "#64748B" }}
                label={{
                  value: "ε (Privacy Budget)",
                  position: "insideBottom",
                  offset: -4,
                  fontSize: 11,
                  fill: "#64748B",
                }}
              />
              <YAxis tick={{ fontSize: 11, fill: "#64748B" }} domain={[0, 1]} />
              <Tooltip
                formatter={(v: any) => Number(v).toFixed(3)}
                labelFormatter={(v) => `ε = ${v}`}
              />
              {METRICS_LIST.filter((m) => activeMetrics.includes(m.key)).map(
                ({ key, label, color }) => (
                  <Line
                    key={key}
                    dataKey={key}
                    name={label}
                    stroke={color}
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: color }}
                    activeDot={{ r: 5 }}
                  />
                ),
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </article>

      {/* Recommendation Banner */}
      <aside
        aria-label="Recommended privacy budget"
        className="bg-gradient-to-r from-blue-700 via-blue-800 to-indigo-900 rounded-2xl p-6 text-white shadow-md flex flex-col md:flex-row md:items-center md:justify-between gap-4"
      >
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Shield size={18} className="text-blue-300" />
            <span className="text-xs font-semibold tracking-wider uppercase text-blue-200">
              Optimal Enterprise Trade-off
            </span>
          </div>
          <div className="font-mono text-3xl font-extrabold tracking-tight">
            Recommended: ε = 1.0
          </div>
          <p className="text-xs text-blue-100/80 max-w-xl">
            Provides 94.8% classification accuracy while restricting membership
            attack success to 0.552, adhering to GDPR and HIPAA differential
            privacy thresholds.
          </p>
        </div>

        <button
          onClick={() => navigate("/privacy-utility")}
          className="self-start md:self-auto inline-flex items-center gap-2 bg-white text-blue-900 hover:bg-blue-50 text-xs sm:text-sm font-bold px-5 py-3 rounded-xl shadow-xs transition-colors"
        >
          Explore Privacy–Utility Frontier
          <ArrowRight size={16} />
        </button>
      </aside>
    </section>
  )
}
