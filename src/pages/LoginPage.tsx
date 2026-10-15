import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  Shield,
  Lock,
  Database,
  BarChart3,
  CheckCircle,
  Eye,
  EyeOff,
  AlertCircle,
} from "lucide-react"
import { useApp } from "../context/AppContext"

export default function LoginPage() {
  const navigate = useNavigate()
  const { login } = useApp()

  const [email, setEmail] = useState("steward@organization.com")
  const [password, setPassword] = useState("differential-privacy-2026")
  const [showPw, setShowPw] = useState(false)
  const [remember, setRemember] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [showSignUp, setShowSignUp] = useState(false)
  const [signUp, setSignUp] = useState({
    name: "",
    email: "",
    password: "",
    confirm: "",
  })

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()
    if (!email || !password) {
      setError("Please enter your credentials.")
      return
    }
    setError("")
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      login()
      navigate("/")
    }, 600)
  }

  const handleSignUp = (e: React.FormEvent) => {
    e.preventDefault()
    if (signUp.password !== signUp.confirm) {
      setError("Passwords do not match.")
      return
    }
    setError("")
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setShowSignUp(false)
      login()
      navigate("/")
    }, 600)
  }

  return (
    <main className="min-h-screen bg-[#F0F4F8] flex">
      {/* Left branding panel */}
      <aside className="hidden lg:flex w-1/2 bg-[#0F172A] flex-col justify-between p-12 text-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-md">
            <Shield size={20} className="text-white" />
          </div>
          <div>
            <div className="text-white font-bold text-xl tracking-tight">
              PrivacyGuard
            </div>
            <div className="text-blue-300/70 text-xs font-medium">
              For Your Datasets
            </div>
          </div>
        </div>

        <div className="max-w-md">
          <h1 className="text-4xl font-extrabold text-white leading-tight mb-4">
            Secure. Private.
            <br />
            Explainable.
          </h1>
          <p className="text-slate-400 text-sm mb-10 leading-relaxed">
            Enterprise-grade decision support platform for differential privacy,
            provable machine learning, and continuous security auditing.
          </p>

          <div className="space-y-4">
            {[
              {
                icon: Lock,
                label: "Encrypted Data Storage",
                desc: "AES-256 zero-knowledge encryption at rest",
              },
              {
                icon: Shield,
                label: "Differential Privacy",
                desc: "Configurable (ε, δ) mathematical guarantees",
              },
              {
                icon: BarChart3,
                label: "Privacy–Utility Frontier",
                desc: "Evidence-based risk & utility modeling",
              },
            ].map(({ icon: Icon, label, desc }) => (
              <div key={label} className="flex items-start gap-3.5">
                <div className="w-9 h-9 bg-blue-500/15 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 text-blue-400">
                  <Icon size={16} />
                </div>
                <div>
                  <div className="text-white text-sm font-semibold">
                    {label}
                  </div>
                  <div className="text-slate-400 text-xs mt-0.5">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <footer className="text-slate-500 text-xs">
          Protected by AES-256 encrypted memory buffers and mathematical
          Differential Privacy guarantees.
        </footer>
      </aside>

      {/* Right login panel */}
      <section className="flex-1 flex items-center justify-center p-6 sm:p-10">
        <div className="w-full max-w-md">
          {/* Mobile brand header */}
          <div className="flex items-center gap-2.5 mb-8 lg:hidden">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-xs">
              <Shield size={20} className="text-white" />
            </div>
            <div>
              <div className="font-bold text-[#0F172A] text-lg">
                PrivacyGuard
              </div>
              <div className="text-blue-600 text-xs font-medium">
                Differential Privacy Platform
              </div>
            </div>
          </div>

          {!showSignUp ? (
            <article className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-[#0F172A] tracking-tight">
                  Data Steward Sign-in
                </h2>
                <p className="text-slate-500 text-xs mt-1">
                  Authenticate to access your secure analysis enclave
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl p-3 text-xs">
                  <AlertCircle size={15} className="flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <form onSubmit={handleLogin} className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Corporate Email
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white transition-colors"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="block text-xs font-semibold text-slate-700">
                      Enclave Password
                    </label>
                    <a
                      href="#forgot"
                      onClick={(e) => {
                        e.preventDefault()
                        alert("Please contact your security administrator.")
                      }}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      Forgot?
                    </a>
                  </div>
                  <div className="relative">
                    <input
                      type={showPw ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white pr-10 transition-colors"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPw(!showPw)}
                      className="absolute right-3 top-3 text-slate-400 hover:text-slate-600"
                    >
                      {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="remember"
                    checked={remember}
                    onChange={(e) => setRemember(e.target.checked)}
                    className="rounded-sm border-slate-300 text-blue-600"
                  />
                  <label
                    htmlFor="remember"
                    className="text-xs text-slate-600 cursor-pointer"
                  >
                    Remember this device for 30 days
                  </label>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-xl text-sm shadow-xs transition-colors disabled:opacity-50"
                >
                  {loading ? "Authenticating..." : "Sign In to Workspace"}
                </button>
              </form>

              <div className="pt-2 text-center text-xs text-slate-500">
                Need an enclave account?{" "}
                <button
                  type="button"
                  onClick={() => setShowSignUp(true)}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Create Steward Account
                </button>
              </div>
            </article>
          ) : (
            <article className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-[#0F172A] tracking-tight">
                  Create Steward Account
                </h2>
                <p className="text-slate-500 text-xs mt-1">
                  Register as an authorized Differential Privacy Data Steward
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 rounded-xl p-3 text-xs">
                  <AlertCircle size={15} className="flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <form onSubmit={handleSignUp} className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Full Name
                  </label>
                  <input
                    type="text"
                    value={signUp.name}
                    onChange={(e) =>
                      setSignUp({ ...signUp, name: e.target.value })
                    }
                    required
                    placeholder="Data Steward"
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Corporate Email
                  </label>
                  <input
                    type="email"
                    value={signUp.email}
                    onChange={(e) =>
                      setSignUp({ ...signUp, email: e.target.value })
                    }
                    required
                    placeholder="steward@organization.com"
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Password
                  </label>
                  <input
                    type="password"
                    value={signUp.password}
                    onChange={(e) =>
                      setSignUp({ ...signUp, password: e.target.value })
                    }
                    required
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Confirm Password
                  </label>
                  <input
                    type="password"
                    value={signUp.confirm}
                    onChange={(e) =>
                      setSignUp({ ...signUp, confirm: e.target.value })
                    }
                    required
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3.5 py-2.5 text-sm focus:outline-hidden focus:border-blue-500 focus:bg-white"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-xl text-sm shadow-xs transition-colors disabled:opacity-50"
                >
                  {loading ? "Creating Enclave..." : "Register Account"}
                </button>
              </form>

              <div className="pt-2 text-center text-xs text-slate-500">
                Already have credentials?{" "}
                <button
                  type="button"
                  onClick={() => setShowSignUp(false)}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Sign In
                </button>
              </div>
            </article>
          )}
        </div>
      </section>
    </main>
  )
}
