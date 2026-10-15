import React from "react"
import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  Upload,
  Settings2,
  Cpu,
  BarChart3,
  ShieldAlert,
  GitCompare,
  Clock,
  FileText,
  Settings,
  HelpCircle,
  LogOut,
  Shield,
  Lock,
  X,
} from "lucide-react"
import { useApp } from "../context/AppContext"

interface SidebarProps {
  mobileOpen: boolean
  setMobileOpen: (open: boolean) => void
}

const navItems = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/dataset", label: "Dataset Upload", icon: Upload },
  { path: "/privacy-config", label: "Privacy Configuration", icon: Settings2 },
  { path: "/secure-processing", label: "Secure Processing", icon: Cpu },
  { path: "/model-analysis", label: "Model Analysis", icon: BarChart3 },
  { path: "/privacy-analysis", label: "Privacy Analysis", icon: ShieldAlert },
  {
    path: "/privacy-utility",
    label: "Privacy–Utility Explorer",
    icon: GitCompare,
  },
  { path: "/experiment-history", label: "Experiment History", icon: Clock },
  { path: "/reports", label: "Reports", icon: FileText },
]

export default function Sidebar({ mobileOpen, setMobileOpen }: SidebarProps) {
  const { logout } = useApp()

  const handleLinkClick = () => {
    if (mobileOpen) setMobileOpen(false)
  }

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-40 lg:hidden"
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:static top-0 bottom-0 left-0 z-50 w-64 bg-[#0F172A] text-white flex flex-col transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          mobileOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full"
        }`}
      >
        {/* Brand header */}
        <div className="px-5 py-5 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-inner flex-shrink-0">
              <Shield size={18} className="text-white" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight leading-none block">
                PrivacyGuard
              </span>
              <span className="text-[10px] text-blue-300/70 font-medium">
                Privacy-Preserving ML
              </span>
            </div>
          </div>

          <button
            onClick={() => setMobileOpen(false)}
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/5"
            aria-label="Close navigation sidebar"
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              onClick={handleLinkClick}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-blue-600 text-white font-medium shadow-xs"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`
              }
            >
              <Icon size={17} className="flex-shrink-0" />
              <span className="truncate">{label}</span>
            </NavLink>
          ))}

          <div className="pt-3 mt-3 border-t border-white/10 space-y-1">
            <NavLink
              to="/profile"
              onClick={handleLinkClick}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-blue-600 text-white font-medium"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`
              }
            >
              <Settings size={17} className="flex-shrink-0" />
              <span>Settings</span>
            </NavLink>

            <a
              href="#help"
              onClick={(e) => {
                e.preventDefault()
                alert(
                  "PrivacyGuard Differential Privacy Documentation & Decision Support Guide",
                )
              }}
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:text-white hover:bg-white/5 transition-colors"
            >
              <HelpCircle size={17} className="flex-shrink-0" />
              <span>Help & Docs</span>
            </a>

            <button
              onClick={() => {
                logout()
                handleLinkClick()
              }}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-slate-300 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <LogOut size={17} className="flex-shrink-0" />
              <span>Logout</span>
            </button>
          </div>
        </nav>

        {/* Security badge footer */}
        <div className="p-4 border-t border-white/10 bg-slate-900/50">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <Lock size={12} className="text-green-400" />
            <span className="text-green-400 text-xs font-medium">
              Encrypted Session Active
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}
