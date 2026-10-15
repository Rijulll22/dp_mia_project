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
  Legend,
} from "recharts"
import { Shield, Play, Check, ArrowRight, TrendingUp } from "lucide-react"
import { useApp } from "../context/AppContext"

const ALL_BUDGET_DATA = [
  {
    epsilon: 0.1,
    accuracy: 0.891,
    precision: 0.872,
    recall: 0.801,
    f1: 0.835,
    rocAuc: 0.934,
    miaRisk: 0.511,
    reconRisk: 0.08,
  },
  {
    epsilon: 0.5,
    accuracy: 0.933,
    precision: 0.912,
    recall: 0.843,
    f1: 0.876,
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
    f1: 0.904,
    rocAuc: 0.981,
    miaRisk: 0.579,
    reconRisk: 0.22,
  },
  {
    epsilon: 5.0,
    accuracy: 0.963,
    precision: 0.948,
    recall: 0.879,
    f1: 0.912,
    rocAuc: 0.986,
    miaRisk: 0.621,
    reconRisk: 0.31,
  },
  {
    epsilon: 10.0,
    accuracy: 0.967,
    precision: 0.953,
    recall: 0.887,
    f1: 0.919,
    rocAuc: 0.989,
    miaRisk: 0.668,
    reconRisk: 0.42,
  },
]

const EPSILON_OPTIONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

const CHART_CONFIGS = [
  { key: "accuracy", label: "Accuracy", color: "#2563EB" },
  { key: "f1", label: "F1-Score", color: "#7C3AED" },
  { key: "miaRisk", label: "MIA Risk", color: "#D97706" },
  { key: "reconRisk", label: "Recon. Risk", color: "#DC2626" },
]

export default function PrivacyUtilityPage() {
  const navigate = useNavigate()
  const { setPrivacyConfig, privacyConfig, experimentResults, lastRunResult } =
    useApp()

  const activeResult = lastRunResult || experimentResults
  const dynamicBudgetData =
    activeResult?.pareto_curve || activeResult?.tradeoffCurve || ALL_BUDGET_DATA

  const [selectedEpsilons, setSelectedEpsilons] = useState<number[]>([
    0.1, 0.5, 1.0, 2.0, 5.0, 10.0,
  ])
  const [activeMetric, setActiveMetric] = useState("accuracy")

  const toggleEpsilon = (eps: number) => {
    setSelectedEpsilons((prev) =>
      prev.includes(eps)
        ? prev.filter((e) => e !== eps)
        : [...prev, eps].sort((a, b) => a - b),
    )
  }

  const filteredData = dynamicBudgetData.filter((d) =>
    selectedEpsilons.includes(d.epsilon),
  )

  const applyBudget = (eps: number) => {
    setPrivacyConfig((prev) => ({
      ...prev,
      epsilon: eps,
      noiseScale: (prev.sensitivity / eps).toFixed(4),
    }))
    navigate("/privacy-config")
  }

  return (
    <section
      aria-labelledby="explorer-heading"
      className="space-y-6 max-w-7xl mx-auto"
    >
      <header>
        <h1
          id="explorer-heading"
          className="text-2xl font-bold text-[#0F172A] tracking-tight"
        >
          Privacy–Utility Frontier Explorer
        </h1>
        <p className="text-slate-500 text-sm mt-0.5">
          Simulate multi-budget Differential Privacy scenarios to identify
          optimal decision thresholds
        </p>
      </header>

      {/* Budget Selector Toolbar */}
      <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-bold text-[#0F172A] uppercase tracking-wider">
              Compare Budgets (ε):
            </span>
            <div className="flex flex-wrap gap-2">
              {EPSILON_OPTIONS.map((e) => {
                const isChecked = selectedEpsilons.includes(e)
                return (
                  <button
                    key={e}
                    type="button"
                    onClick={() => toggleEpsilon(e)}
                    className={`text-xs font-mono font-semibold px-3 py-1.5 rounded-lg border transition-all ${
                      isChecked
                        ? "bg-blue-600 text-white border-blue-600 shadow-2xs"
                        : "bg-slate-50 text-slate-600 border-slate-200 hover:border-slate-300"
                    }`}
                  >
                    ε = {e}
                  </button>
                )
              })}
            </div>
          </div>

          <button
            type="button"
            onClick={() => setSelectedEpsilons([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])}
            className="text-xs font-semibold text-blue-600 hover:underline"
          >
            Select All
          </button>
        </div>
      </article>

      {/* Comparative Multi-Budget Table */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-2 bg-slate-50/50">
          <div>
            <h2 className="text-sm font-semibold text-[#0F172A]">
              Empirical Trade-off Matrix
            </h2>
            <p className="text-xs text-slate-400">
              Comparing classifier metrics and attacker success across budgets
            </p>
          </div>
          <span className="text-xs text-blue-700 font-semibold bg-blue-50 border border-blue-200 rounded-full px-2.5 py-0.5">
            ★ ε = 1.0 Optimal Balance
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 text-left">
                <th className="px-4 py-3 font-bold">ε Budget</th>
                <th className="px-4 py-3 font-semibold">Accuracy</th>
                <th className="px-4 py-3 font-semibold">Precision</th>
                <th className="px-4 py-3 font-semibold">Recall</th>
                <th className="px-4 py-3 font-semibold">F1-Score</th>
                <th className="px-4 py-3 font-semibold">ROC-AUC</th>
                <th className="px-4 py-3 font-semibold">MIA Risk</th>
                <th className="px-4 py-3 font-semibold">Recon. Risk</th>
                <th className="px-4 py-3 font-semibold text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredData.map((row) => {
                const isRec = row.epsilon === 1.0
                const isCurrent =
                  Math.abs(privacyConfig.epsilon - row.epsilon) < 0.05

                return (
                  <tr
                    key={row.epsilon}
                    className={`hover:bg-blue-50/20 transition-colors ${
                      isRec ? "bg-blue-50/40" : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-bold text-blue-700">
                      ε = {row.epsilon}
                      {isRec && (
                        <span className="ml-1.5 text-[9px] font-sans font-bold text-blue-700 bg-blue-100 px-1.5 py-0.5 rounded-sm">
                          REC
                        </span>
                      )}
                      {isCurrent && (
                        <span className="ml-1.5 text-[9px] font-sans font-bold text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded-sm">
                          ACTIVE
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-800 font-bold">
                      {(row.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {((row.precision ?? 0.9) * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {((row.recall ?? 0.85) * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-slate-800 font-bold">
                      {(row.f1 * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {(row.rocAuc * 100).toFixed(1)}%
                    </td>
                    <td
                      className={`px-4 py-3 font-bold ${
                        row.miaRisk > 0.6
                          ? "text-amber-600"
                          : "text-emerald-600"
                      }`}
                    >
                      {row.miaRisk.toFixed(3)}
                    </td>
                    <td
                      className={`px-4 py-3 font-bold ${
                        row.reconRisk > 0.25
                          ? "text-amber-600"
                          : "text-emerald-600"
                      }`}
                    >
                      {row.reconRisk.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => applyBudget(row.epsilon)}
                        className="text-[11px] font-sans font-semibold text-blue-600 hover:text-blue-800 hover:underline"
                      >
                        Apply this ε →
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </article>

      {/* Dual Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metric Focus Chart */}
        <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
              Selected Metric vs. ε Budget
            </h2>
            <div className="flex gap-1">
              {CHART_CONFIGS.map(({ key, label, color }) => (
                <button
                  key={key}
                  onClick={() => setActiveMetric(key)}
                  className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border transition-all ${
                    activeMetric === key
                      ? "text-white border-transparent"
                      : "border-slate-200 text-slate-500 hover:bg-slate-50"
                  }`}
                  style={activeMetric === key ? { backgroundColor: color } : {}}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="h-56 sm:h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={filteredData}
                margin={{ top: 10, right: 15, left: -15, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  dataKey="epsilon"
                  tick={{ fontSize: 11, fill: "#64748B" }}
                  label={{
                    value: "ε Budget",
                    position: "insideBottom",
                    offset: -4,
                    fontSize: 11,
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#64748B" }}
                  domain={[0, 1]}
                />
                <Tooltip labelFormatter={(v) => `ε = ${v}`} />
                {CHART_CONFIGS.filter((c) => c.key === activeMetric).map(
                  ({ key, label, color }) => (
                    <Line
                      key={key}
                      dataKey={key}
                      name={label}
                      stroke={color}
                      strokeWidth={3}
                      dot={{ r: 4, fill: color }}
                      activeDot={{ r: 6 }}
                    />
                  ),
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        {/* Global Multi-Metric Frontier */}
        <article className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs space-y-3">
          <h2 className="text-sm font-semibold text-[#0F172A] uppercase tracking-wider">
            Combined Utility & Attack Risk Frontier
          </h2>

          <div className="h-56 sm:h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={filteredData}
                margin={{ top: 10, right: 15, left: -15, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  dataKey="epsilon"
                  tick={{ fontSize: 11, fill: "#64748B" }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#64748B" }}
                  domain={[0, 1]}
                />
                <Tooltip labelFormatter={(v) => `ε = ${v}`} />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  name="Accuracy"
                  stroke="#2563EB"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="f1"
                  name="F1-Score"
                  stroke="#7C3AED"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="miaRisk"
                  name="MIA Risk (Lower is better)"
                  stroke="#D97706"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>
    </section>
  )
}
