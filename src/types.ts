export type Page = "login" | "dashboard" | "dataset" | "privacy-config" | "secure-processing" | "model-analysis" | "privacy-analysis" | "privacy-utility" | "experiment-history" | "reports" | "profile"

export interface AppState {
  authenticated: boolean
  currentPage: Page
  setPage: (page: Page) => void
  logout: () => void
}
