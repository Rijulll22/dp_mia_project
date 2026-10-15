import React from "react"

interface NoiseSliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (val: number) => void
  unit?: string
  badge?: string
  badgeColor?: string
  minLabel?: string
  maxLabel?: string
  description?: string
}

export default function NoiseSlider({
  label,
  value,
  min,
  max,
  step = 0.1,
  onChange,
  unit = "",
  badge,
  badgeColor = "text-blue-700 bg-blue-50 border-blue-200",
  minLabel,
  maxLabel,
  description,
}: NoiseSliderProps) {
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = parseFloat(e.target.value)
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, parsed))
      onChange(clamped)
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-sm font-semibold text-[#0F172A]">
            {label}
          </label>
          {badge && (
            <span
              className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${badgeColor}`}
            >
              {badge}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={handleInputChange}
            aria-label={`${label} numeric input`}
            className="w-20 text-right font-mono text-sm font-bold text-blue-700 bg-slate-50 border border-slate-200 rounded-md px-2 py-1 focus:outline-hidden focus:border-blue-500 focus:bg-white transition-colors"
          />
          {unit && (
            <span className="text-xs font-mono text-slate-500">{unit}</span>
          )}
        </div>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={`${label} slider`}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-200 rounded-full appearance-none cursor-pointer accent-blue-600 focus:outline-hidden"
      />

      <div className="flex justify-between text-xs text-slate-400 font-mono">
        <span>{minLabel ?? `${min}`}</span>
        <span>{maxLabel ?? `${max}`}</span>
      </div>

      {description && (
        <p className="text-xs text-slate-500 mt-1">{description}</p>
      )}
    </div>
  )
}
