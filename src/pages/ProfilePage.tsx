import React, { useState } from "react"
import {
  User,
  Lock,
  LogOut,
  Monitor,
  Eye,
  EyeOff,
  Shield,
  CheckCircle2,
} from "lucide-react"
import { useApp } from "../context/AppContext"
import StatusBanner from "../components/StatusBanner"

export default function ProfilePage() {
  const { logout } = useApp()

  const [showPw, setShowPw] = useState(false)
  const [currentPw, setCurrentPw] = useState("")
  const [newPw, setNewPw] = useState("")
  const [saveBanner, setSaveBanner] = useState<string | null>(null)

  const handleUpdatePassword = (e: React.FormEvent) => {
    e.preventDefault()
    if (!currentPw || !newPw) return
    setSaveBanner("Security credentials updated successfully.")
    setCurrentPw("")
    setNewPw("")
  }

  return (
    <section
      aria-labelledby="profile-heading"
      className="space-y-6 max-w-2xl mx-auto"
    >
      <header>
        <h1
          id="profile-heading"
          className="text-2xl font-bold text-[#0F172A] tracking-tight"
        >
          Operator Profile & Security Enclave
        </h1>
        <p className="text-slate-500 text-sm mt-0.5">
          Manage credentials, cryptographic keys, and authenticated session
          state
        </p>
      </header>

      {saveBanner && (
        <StatusBanner
          variant="success"
          message={saveBanner}
          onDismiss={() => setSaveBanner(null)}
        />
      )}

      {/* Profile Card */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-5">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 bg-blue-600 rounded-2xl flex items-center justify-center text-white text-xl font-bold shadow-xs">
            DS
          </div>
          <div>
            <span className="font-bold text-[#0F172A] text-lg block leading-tight">
              Data Steward
            </span>
            <span className="text-slate-500 text-sm">
              steward@enterprise-security.org
            </span>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[10px] font-bold text-blue-700 bg-blue-50 border border-blue-200 px-2.5 py-0.5 rounded-full">
                Security Officer Role
              </span>
              <span className="text-[11px] text-slate-400">
                Enclave ID: #DS-8842
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2">
          {[
            ["Operator Name", "Data Steward"],
            ["Organization", "Enterprise Privacy Labs"],
            ["Differential Privacy Clearance", "Level 4 (Full Budget)"],
            ["Cryptographic Hardware Enclave", "AES-NI Hardware Active"],
          ].map(([k, v]) => (
            <div
              key={k}
              className="bg-slate-50 border border-slate-100 rounded-xl p-3"
            >
              <span className="text-[10px] text-slate-400 font-medium block mb-0.5">
                {k}
              </span>
              <span className="text-xs font-bold text-[#0F172A] block">
                {v}
              </span>
            </div>
          ))}
        </div>
      </article>

      {/* Password Change Card */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
            <Lock size={16} />
          </div>
          <h2 className="text-sm font-semibold text-[#0F172A]">
            Update Enclave Password
          </h2>
        </div>

        <form onSubmit={handleUpdatePassword} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Current Password
            </label>
            <input
              type={showPw ? "text" : "password"}
              value={currentPw}
              onChange={(e) => setCurrentPw(e.target.value)}
              placeholder="••••••••••••"
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              New Secure Password
            </label>
            <div className="relative">
              <input
                type={showPw ? "text" : "password"}
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="••••••••••••"
                className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600"
              >
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2.5 rounded-lg transition-colors"
          >
            Update Credentials
          </button>
        </form>
      </article>

      {/* Logout Action */}
      <article className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 flex items-center justify-between">
        <div>
          <span className="text-sm font-bold text-[#0F172A] block">
            Sign Out of Enclave
          </span>
          <span className="text-xs text-slate-500">
            Terminates encrypted browser memory session
          </span>
        </div>

        <button
          onClick={logout}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 px-4 py-2 rounded-xl transition-colors"
        >
          <LogOut size={14} />
          Sign Out
        </button>
      </article>
    </section>
  )
}
