import React from "react"
import { LucideIcon, TrendingUp, TrendingDown } from "lucide-react"

interface MetricsCardProps {
  label: string
  value: string | number
  delta?: string
  deltaType?: "positive" | "negative" | "neutral"
  subtext?: string
  icon?: LucideIcon
  variant?: "blue" | "indigo" | "violet" | "green" | "amber" | "red"
  className?: string
}

const variantStyles = {
  blue: { bg: "bg-blue-50", text: "text-blue-600", border: "border-blue-100" },
  indigo: {
    bg: "bg-indigo-50",
    text: "text-indigo-600",
    border: "border-indigo-100",
  },
  violet: {
    bg: "bg-violet-50",
    text: "text-violet-600",
    border: "border-violet-100",
  },
  green: {
    bg: "bg-green-50",
    text: "text-green-600",
    border: "border-green-100",
  },
  amber: {
    bg: "bg-amber-50",
    text: "text-amber-600",
    border: "border-amber-100",
  },
  red: { bg: "bg-red-50", text: "text-red-600", border: "border-red-100" },
}

export default function MetricsCard({
  label,
  value,
  delta,
  deltaType = "positive",
  subtext,
  icon: Icon,
  variant = "blue",
  className = "",
}: MetricsCardProps) {
  const styles = variantStyles[variant]

  return (
    <article
      className={`bg-white rounded-xl border border-slate-200 p-4 sm:p-5 shadow-xs transition-shadow hover:shadow-sm ${className}`}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide truncate">
          {label}
        </span>
        {Icon && (
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${styles.bg} ${styles.text}`}
          >
            <Icon size={16} />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span className="text-xl sm:text-2xl font-bold text-[#0F172A] tracking-tight">
          {value}
        </span>
        {delta && (
          <span
            className={`inline-flex items-center gap-0.5 text-xs font-semibold px-1.5 py-0.5 rounded-sm ${
              deltaType === "positive"
                ? "text-green-700 bg-green-50"
                : deltaType === "negative"
                  ? "text-red-700 bg-red-50"
                  : "text-slate-600 bg-slate-100"
            }`}
          >
            {deltaType === "positive" ? (
              <TrendingUp size={11} className="text-green-600" />
            ) : deltaType === "negative" ? (
              <TrendingDown size={11} className="text-red-600" />
            ) : null}
            {delta}
          </span>
        )}
      </div>

      {subtext && (
        <p className="text-xs text-slate-500 mt-1.5 leading-normal">
          {subtext}
        </p>
      )}
    </article>
  )
}
