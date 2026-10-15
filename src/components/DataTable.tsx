import React from "react"
import { Database } from "lucide-react"

interface DataTableProps {
  data: Array<Record<string, string | number>>
  columns?: string[]
  title?: string
  subtitle?: string
  totalRows?: number
  maxDisplay?: number
  highlightColumn?: string
}

export default function DataTable({
  data,
  columns,
  title = "Data Preview",
  subtitle,
  totalRows,
  maxDisplay = 5,
  highlightColumn = "Class",
}: DataTableProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center">
        <Database size={32} className="mx-auto text-slate-300 mb-2" />
        <p className="text-sm font-semibold text-slate-600">
          No data available
        </p>
        <p className="text-xs text-slate-400 mt-1">
          Upload a CSV dataset to view preview rows
        </p>
      </div>
    )
  }

  const tableColumns =
    columns && columns.length > 0 ? columns : Object.keys(data[0] || {})
  const displayRows = data.slice(0, maxDisplay)

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
      <div className="px-5 py-3.5 border-b border-slate-100 flex flex-wrap items-center justify-between gap-2 bg-slate-50/50">
        <div>
          <span className="text-sm font-semibold text-[#0F172A]">{title}</span>
          {subtitle && (
            <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
          )}
        </div>
        {totalRows && (
          <span className="text-xs text-slate-400 font-mono">
            Showing {displayRows.length} of {totalRows.toLocaleString()} rows ·
            sensitive columns masked
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 text-left">
              <th className="px-4 py-2.5 font-semibold text-slate-400 w-12 text-center">
                #
              </th>
              {tableColumns.map((col) => (
                <th
                  key={col}
                  className="px-4 py-2.5 font-semibold whitespace-nowrap"
                >
                  {col}
                  {col === highlightColumn && (
                    <span className="ml-1 text-[9px] bg-blue-100 text-blue-700 px-1.5 py-0.2 rounded-sm font-sans">
                      target
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {displayRows.map((row, i) => (
              <tr key={i} className="hover:bg-blue-50/20 transition-colors">
                <td className="px-4 py-2 text-slate-400 text-center">
                  {i + 1}
                </td>
                {tableColumns.map((col) => {
                  const val = row[col]
                  const isTarget = col === highlightColumn
                  const isPositive = val === "1" || val === 1

                  return (
                    <td
                      key={col}
                      className={`px-4 py-2 whitespace-nowrap ${
                        isTarget
                          ? isPositive
                            ? "text-red-600 font-bold bg-red-50/40"
                            : "text-green-600 font-medium"
                          : "text-slate-700"
                      }`}
                    >
                      {val !== undefined ? String(val) : "—"}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
