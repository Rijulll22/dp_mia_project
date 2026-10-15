import React from "react"
import { Menu, Shield, Lock, Bell, ChevronDown, RefreshCw } from "lucide-react"
import { useApp } from "../context/AppContext"

interface HeaderProps {
  onOpenMobile: () => void
}

export default function Header({ onOpenMobile }: HeaderProps) {
  const { dataset, privacyConfig, backendStatus, checkBackend } = useApp()

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center px-4 sm:px-6 gap-3 flex-shrink-0 z-10">
      {/* Mobile Menu Toggle */}
      <button
        onClick={onOpenMobile}
        className="lg:hidden p-2 rounded-lg text-slate-600 hover:text-[#0F172A] hover:bg-slate-100 transition-colors"
        aria-label="Open sidebar navigation"
      >
        <Menu size={20} />
      </button>

      {/* Dataset & Project Info */}
      <div className="flex items-center gap-2 min-w-0">
        <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 min-w-0">
          <div className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0"></div>
          <span className="text-xs font-semibold text-[#0F172A] truncate">
            {dataset.name
              ? dataset.name.replace(/\.[^/.]+$/, "")
              : "Credit Card Fraud"}
          </span>
          <span className="hidden sm:inline text-[11px] text-slate-400">
            ({dataset.records.toLocaleString()} rows)
          </span>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-2 sm:gap-3 ml-auto">
        {/* Backend connectivity indicator */}
        <button
          onClick={() => checkBackend()}
          title="Click to check backend connectivity at http://localhost:8000/api"
          className={`hidden md:flex items-center gap-1.5 text-xs font-medium border rounded-md px-2.5 py-1 transition-colors ${
            backendStatus.online
              ? "text-emerald-700 bg-emerald-50 border-emerald-200"
              : "text-slate-600 bg-slate-50 border-slate-200 hover:bg-slate-100"
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              backendStatus.online ? "bg-emerald-500" : "bg-slate-400"
            }`}
          />
          <span>{backendStatus.online ? "API Online" : "Simulation Mode"}</span>
          <RefreshCw
            size={11}
            className="text-slate-400 ml-0.5 hover:rotate-180 transition-transform"
          />
        </button>

        {/* DP status pill */}
        <span className="hidden sm:flex items-center gap-1.5 text-xs font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-md px-2.5 py-1">
          <Shield size={12} className="text-blue-600" />
          <span>DP ε={privacyConfig.epsilon}</span>
        </span>

        {/* Notification Bell */}
        <button
          onClick={() =>
            alert("All privacy checks passing. No security alerts.")
          }
          className="relative p-2 text-slate-500 hover:text-[#0F172A] hover:bg-slate-100 rounded-lg transition-colors"
          aria-label="View notifications"
        >
          <Bell size={18} />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-blue-600 rounded-full border-2 border-white"></span>
        </button>

        {/* User profile */}
        <div className="flex items-center gap-2 pl-1 sm:pl-2 py-1">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-xs">
            DS
          </div>
          <div className="hidden xl:block text-left">
            <span className="text-xs font-semibold text-[#0F172A] block leading-tight">
              Data Steward
            </span>
            <span className="text-[10px] text-slate-400 block leading-tight">
              Security Officer
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
