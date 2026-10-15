import React from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { AppProvider, useApp } from "./context/AppContext"
import Layout from "./components/Layout"
import LoginPage from "./pages/LoginPage"
import DashboardPage from "./pages/DashboardPage"
import DatasetUploadPage from "./pages/DatasetUploadPage"
import PrivacyConfigPage from "./pages/PrivacyConfigPage"
import SecureProcessingPage from "./pages/SecureProcessingPage"
import ModelAnalysisPage from "./pages/ModelAnalysisPage"
import PrivacyAnalysisPage from "./pages/PrivacyAnalysisPage"
import PrivacyUtilityPage from "./pages/PrivacyUtilityPage"
import ExperimentHistoryPage from "./pages/ExperimentHistoryPage"
import ReportsPage from "./pages/ReportsPage"
import ProfilePage from "./pages/ProfilePage"

/**
 * Route protection: requires authenticated session
 */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { authenticated } = useApp()
  if (!authenticated) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

/**
 * Public route protection: redirects logged-in users away from login
 */
function PublicRoute({ children }: { children: React.ReactNode }) {
  const { authenticated } = useApp()
  if (authenticated) {
    return <Navigate to="/" replace />
  }
  return <>{children}</>
}

function AppRoutes() {
  return (
    <Routes>
      <Route
        path="/login"
        element={
          <PublicRoute>
            <LoginPage />
          </PublicRoute>
        }
      />

      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<DashboardPage />} />
        <Route path="dashboard" element={<Navigate to="/" replace />} />
        <Route path="dataset" element={<DatasetUploadPage />} />
        <Route path="privacy-config" element={<PrivacyConfigPage />} />
        <Route path="secure-processing" element={<SecureProcessingPage />} />
        <Route path="model-analysis" element={<ModelAnalysisPage />} />
        <Route path="privacy-analysis" element={<PrivacyAnalysisPage />} />
        <Route path="privacy-utility" element={<PrivacyUtilityPage />} />
        <Route path="experiment-history" element={<ExperimentHistoryPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="profile" element={<ProfilePage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AppProvider>
  )
}
