import React from "react"
import { CheckCircle2, AlertTriangle, AlertCircle, Info, X } from "lucide-react"

export type BannerVariant = "success" | "warning" | "error" | "info"

interface StatusBannerProps {
  variant: BannerVariant
  title?: string
  message: string
  onDismiss?: () => void
  actionText?: string
  onAction?: () => void
  className?: string
}

const variantConfig = {
  success: {
    icon: CheckCircle2,
    bg: "bg-green-50",
    border: "border-green-200",
    text: "text-green-800",
    iconColor: "text-green-600",
  },
  warning: {
    icon: AlertTriangle,
    bg: "bg-amber-50",
    border: "border-amber-200",
    text: "text-amber-800",
    iconColor: "text-amber-600",
  },
  error: {
    icon: AlertCircle,
    bg: "bg-red-50",
    border: "border-red-200",
    text: "text-red-800",
    iconColor: "text-red-600",
  },
  info: {
    icon: Info,
    bg: "bg-blue-50",
    border: "border-blue-200",
    text: "text-blue-800",
    iconColor: "text-blue-600",
  },
}

export default function StatusBanner({
  variant,
  title,
  message,
  onDismiss,
  actionText,
  onAction,
  className = "",
}: StatusBannerProps) {
  const cfg = variantConfig[variant]
  const Icon = cfg.icon

  return (
    <aside
      aria-label="Status notification"
      className={`flex items-start gap-3 p-4 rounded-xl border ${cfg.bg} ${cfg.border} ${cfg.text} ${className}`}
    >
      <Icon size={18} className={`flex-shrink-0 mt-0.5 ${cfg.iconColor}`} />
      <div className="flex-1 text-sm">
        {title && <span className="font-semibold block mb-0.5">{title}</span>}
        <p className="leading-relaxed">{message}</p>
        {actionText && onAction && (
          <button
            onClick={onAction}
            className="mt-2 text-xs font-semibold underline hover:opacity-80 transition-opacity"
          >
            {actionText}
          </button>
        )}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="p-1 hover:bg-black/5 rounded-md transition-colors"
          aria-label="Dismiss notification"
        >
          <X size={15} />
        </button>
      )}
    </aside>
  )
}
