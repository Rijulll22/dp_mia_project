import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  Shield,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Info,
  Check,
  Sliders,
  Database,
  Lock,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import NoiseSlider from "../components/NoiseSlider"

export default function PrivacyConfigPage() {
  const navigate = useNavigate()
  const { privacyConfig, setPrivacyConfig, dataset } = useApp()

  const [epsilon, setEpsilon] = useState<number>(privacyConfig.epsilon)
  const [delta, setDelta] = useState<number>(privacyConfig.delta || 0.00001)
  const [sensitivity, setSensitivity] = useState<number>(
    privacyConfig.sensitivity || 1.0,
  )
  const [mechanismType, setMechanismType] =
    useState<"laplace_clipped" | "plain_laplace" | "non_private">(
      (privacyConfig.mechanismType as "laplace_clipped" | "plain_laplace" | "non_private") || "laplace_clipped",
    )
  const [classifierType, setClassifierType] =
    useState<"logistic_regression" | "random_forest">(
      privacyConfig.classifierType || "logistic_regression",
    )
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)

  // Dynamic noise scale calculation
  const calculatedNoiseScale =
    mechanismType === "non_private"
      ? "0.0000"
      : (sensitivity / epsilon).toFixed(4)

  const getPrivacyRating = (e: number) => {
    if (e <= 0.5)
      return {
        label: "Very High",
        color: "text-emerald-700 bg-emerald-50 border-emerald-200",
      }
    if (e <= 1.0)
      return {
        label: "High",
        color: "text-blue-700 bg-blue-50 border-blue-200",
      }
    if (e <= 3.0)
      return {
        label: "Moderate",
        color: "text-amber-700 bg-amber-50 border-amber-200",
      }
    return { label: "Lower", color: "text-red-700 bg-red-50 border-red-200" }
  }

  const privacyLevel = getPrivacyRating(epsilon)

  const presets = [
    {
      title: "Strong Privacy",
      eps: 0.1,
      sens: 1.0,
      d: 0.00001,
      desc: "Maximum theoretical protection. Highest noise injection.",
    },
    {
      title: "Balanced",
      eps: 1.0,
      sens: 1.0,
      d: 0.00001,
      desc: "Standard enterprise recommendation. Strong utility with provable bounds.",
    },
    {
      title: "High Utility",
      eps: 5.0,
      sens: 1.0,
      d: 0.00001,
      desc: "Minimal perturbation for maximum ML classifier accuracy.",
    },
  ]

  const handleApplyPreset = (p: typeof presets[0]) => {
    setEpsilon(p.eps)
    setSensitivity(p.sens)
    setDelta(p.d)
  }

  const handleContinue = () => {
    setPrivacyConfig({
      epsilon,
      delta,
      sensitivity,
      mechanism: mechanismType,
      mechanismType,
      classifierType,
      noiseScale: calculatedNoiseScale,
    })
    navigate("/secure-processing")
  }

  return (
    <section
      aria-labelledby="config-heading"
      className="space-y-6 max-w-4xl mx-auto"
    >
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1
            id="config-heading"
            className="text-2xl font-bold text-[#0F172A] tracking-tight"
          >
            Differential Privacy Configuration
          </h1>
          <p className="text-slate-500 text-sm mt-0.5">
            Calibrate mathematical noise parameters for dataset:{" "}
            <span className="font-semibold text-slate-700">{dataset.name}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold px-3 py-1 rounded-full border bg-white text-slate-700 border-slate-200 shadow-2xs">
            Sensitivity (Δf): {sensitivity.toFixed(1)}
          </span>
        </div>
      </header>

      {/* Main DP Parameter Card */}
      <article className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
              <Shield size={18} />
            </div>
            <div>
              <h2 className="text-base font-semibold text-[#0F172A]">
                Privacy Budget & Mechanism
              </h2>
              <p className="text-xs text-slate-400">
                Strict $(\epsilon, \delta)$-Differential Privacy Guarantee
              </p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200">
              <span className="text-[11px] font-semibold text-slate-500 px-2">
                Mechanism:
              </span>
              <button
                type="button"
                onClick={() => setMechanismType("laplace_clipped")}
                className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg transition-all ${
                  mechanismType === "laplace_clipped"
                    ? "bg-blue-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title="Laplace perturbation with bounded post-processing clip to [0,1]"
              >
                Laplace + Clipping
              </button>
              <button
                type="button"
                onClick={() => setMechanismType("plain_laplace")}
                className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg transition-all ${
                  mechanismType === "plain_laplace"
                    ? "bg-blue-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title="Unbounded standard Laplace perturbation"
              >
                Plain Laplace
              </button>
              <button
                type="button"
                onClick={() => setMechanismType("non_private")}
                className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg transition-all ${
                  mechanismType === "non_private"
                    ? "bg-red-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title="No differential privacy, raw data"
              >
                Non-Private
              </button>
            </div>

            <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200">
              <span className="text-[11px] font-semibold text-slate-500 px-2">
                Model:
              </span>
              <button
                type="button"
                onClick={() => setClassifierType("logistic_regression")}
                className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg transition-all ${
                  classifierType === "logistic_regression"
                    ? "bg-indigo-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Logistic Regression
              </button>
              <button
                type="button"
                onClick={() => setClassifierType("random_forest")}
                className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg transition-all ${
                  classifierType === "random_forest"
                    ? "bg-indigo-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Random Forest
              </button>
            </div>
          </div>
        </div>

        {/* Epsilon Slider & Numeric Input */}
        <div className="space-y-4">
          <NoiseSlider
            label="Privacy Budget (ε)"
            value={epsilon}
            min={0.1}
            max={10.0}
            step={0.1}
            onChange={(val) => setEpsilon(parseFloat(val.toFixed(2)))}
            badge={`${privacyLevel.label} Privacy`}
            badgeColor={privacyLevel.color}
            minLabel="ε = 0.1 (Max Privacy)"
            maxLabel="ε = 10.0 (High Utility)"
            description="Lower values add more mathematical noise to guarantee robust resistance against membership inference."
          />
        </div>

        {/* Presets Grid */}
        <div>
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-2.5">
            Recommended Presets
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {presets.map((p) => {
              const isSelected = Math.abs(epsilon - p.eps) < 0.05
              return (
                <button
                  key={p.title}
                  type="button"
                  onClick={() => handleApplyPreset(p)}
                  className={`text-left p-3.5 rounded-xl border transition-all text-xs ${
                    isSelected
                      ? "border-blue-500 bg-blue-50/70 shadow-2xs ring-1 ring-blue-400"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/60"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-[#0F172A]">{p.title}</span>
                    {isSelected && (
                      <Check size={14} className="text-blue-600" />
                    )}
                  </div>
                  <div className="font-mono text-blue-700 font-semibold mb-1">
                    ε = {p.eps}
                  </div>
                  <p className="text-[11px] text-slate-500 leading-snug">
                    {p.desc}
                  </p>
                </button>
              )
            })}
          </div>
        </div>

        {/* Dynamic Trade-off Explanations */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="bg-emerald-50/80 border border-emerald-200 rounded-xl p-4">
            <span className="text-xs font-bold text-emerald-800 uppercase tracking-wide block mb-1.5">
              Privacy Guarantee at ε = {epsilon}
            </span>
            <ul className="text-xs text-emerald-900 space-y-1.5">
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>
                Record-level ε budget: {epsilon} (shared across all features)
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>
                Attacker bound: e^ε = {Math.exp(epsilon).toFixed(2)}
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>
                MIA metrics reported from actual experiments only
              </li>
            </ul>
          </div>

          <div className="bg-amber-50/80 border border-amber-200 rounded-xl p-4">
            <span className="text-xs font-bold text-amber-800 uppercase tracking-wide block mb-1.5">
              Noise Calibration
            </span>
            <ul className="text-xs text-amber-900 space-y-1.5">
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
                Per-feature ε (with d features): ε / d
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
                Noise scale (Laplace): sensitivity / (ε / d)
              </li>
              <li className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-600"></span>
                Actual metrics reported from executed experiments only
              </li>
            </ul>
          </div>
        </div>

        {/* Mathematical Parameters Bar */}
        <div className="pt-4 border-t border-slate-100">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <span className="text-xs text-slate-400 block mb-0.5">
                Mechanism & Model
              </span>
              <span className="text-xs font-bold text-[#0F172A] font-mono block">
                {mechanismType === "laplace_clipped"
                  ? "Laplace + Clipping"
                  : mechanismType === "non_private"
                    ? "Non-Private"
                    : "Plain Laplace"}
              </span>
              <span className="text-[10px] text-indigo-600 font-semibold">
                {classifierType === "random_forest"
                  ? "Random Forest"
                  : "Logistic Regression"}
              </span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block mb-0.5">
                Global Sensitivity (Δf)
              </span>
              <span className="text-sm font-bold text-[#0F172A] font-mono">
                {sensitivity.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block mb-0.5">
                Relaxation Delta (δ)
              </span>
              <span className="text-sm font-bold text-[#0F172A] font-mono">
                {mechanismType === "non_private"
                  ? delta.toExponential(0)
                  : "0 (Pure DP)"}
              </span>
            </div>
            <div>
              <span className="text-xs text-slate-400 block mb-0.5">
                Computed Scale (b/σ)
              </span>
              <span className="text-sm font-bold text-blue-700 font-mono">
                {calculatedNoiseScale}
              </span>
            </div>
          </div>
        </div>
      </article>

      {/* Advanced Parameters Section */}
      <article className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-6 py-4 text-sm font-semibold text-[#0F172A] hover:bg-slate-50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Sliders size={16} className="text-slate-500" />
            <span>Advanced Mathematical Parameters (Sensitivity & Delta)</span>
          </div>
          {showAdvanced ? (
            <ChevronUp size={16} className="text-slate-400" />
          ) : (
            <ChevronDown size={16} className="text-slate-400" />
          )}
        </button>

        {showAdvanced && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-100 space-y-5 bg-slate-50/40">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Sensitivity S */}
              <NoiseSlider
                label="Feature Sensitivity (S = Δf)"
                value={sensitivity}
                min={0.1}
                max={5.0}
                step={0.1}
                onChange={(v) => setSensitivity(parseFloat(v.toFixed(2)))}
                minLabel="0.1"
                maxLabel="5.0"
                description="Max change in feature vector after Min-Max normalization [0, 1]."
              />

              {/* Delta Input */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-[#0F172A]">
                    Delta (δ) - Cryptographic Bound
                  </label>
                  <span className="font-mono text-sm font-bold text-blue-700">
                    {delta.toExponential(1)}
                  </span>
                </div>
                <input
                  type="number"
                  step="0.000001"
                  min="0.0000001"
                  max="0.01"
                  value={delta}
                  onChange={(e) =>
                    setDelta(parseFloat(e.target.value) || 0.00001)
                  }
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm font-mono focus:outline-hidden focus:border-blue-500"
                />
                <p className="text-xs text-slate-500">
                  Probability of DP guarantee failure. Recommended: strictly
                  less than 1/N (for 284k records, δ &lt; 3.5e-6).
                </p>
              </div>
            </div>
          </div>
        )}
      </article>

      {/* Navigation CTA */}
      <div className="flex justify-between items-center pt-2">
        <button
          onClick={() => navigate("/dataset")}
          className="text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
        >
          ← Back to Dataset Upload
        </button>

        <button
          onClick={handleContinue}
          className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-6 py-3 rounded-xl shadow-xs hover:shadow-sm transition-all"
        >
          Proceed to Secure Processing
          <ArrowRight size={16} />
        </button>
      </div>
    </section>
  )
}
