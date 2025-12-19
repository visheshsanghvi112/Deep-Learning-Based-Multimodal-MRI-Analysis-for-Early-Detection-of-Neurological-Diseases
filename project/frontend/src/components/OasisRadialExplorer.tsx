"use client"

import { motion, AnimatePresence } from "framer-motion"
import { RadialFilter } from "./RadialFilter"
import { useRadialFilters } from "@/hooks/useRadialFilters"
import { RotateCcw, X } from "lucide-react"
import "@/styles/radialStyles.css"

// =============================================================================
// OASIS RADIAL EXPLORER
// IDA/LONI-style immersive research data explorer
// =============================================================================
export function OasisRadialExplorer() {
  const {
    loading,
    error,
    filters,
    matchMode,
    setMatchMode,
    toggleFilter,
    resetFilters,
    hasActiveFilters,
    filteredData,
    segments,
    data,
  } = useRadialFilters()

  // Calculate stats
  const normalCount = filteredData.filter((s) => s.cdr === 0).length
  const impairedCount = filteredData.filter((s) => s.cdr !== null && s.cdr > 0).length
  const avgAge = filteredData.length > 0
    ? Math.round(filteredData.reduce((sum, s) => sum + s.age, 0) / filteredData.length)
    : 0
  const avgNwbv = filteredData.length > 0
    ? (filteredData.reduce((sum, s) => sum + s.nwbv, 0) / filteredData.length).toFixed(3)
    : "0"

  // Loading state
  if (loading) {
    return (
      <div className="radial-explorer">
        <div className="loading-container">
          <div className="loading-spinner" />
          <div className="loading-text">Loading OASIS Dataset</div>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="radial-explorer">
        <div className="loading-container">
          <div style={{ color: "#f87171", marginBottom: 10 }}>Error loading data</div>
          <div style={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}>{error}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="radial-explorer">
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "60px 40px" }}>
        
        {/* Header */}
        <div className="radial-header">
          <h1 className="radial-title">OASIS-1 Research Explorer</h1>
          <p className="radial-subtitle">
            {data.length} MRI Scans • Cross-Sectional Cohort • Interactive Filtering
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 20, marginBottom: 50, position: "relative", zIndex: 10 }}>
          <div className="match-toggle">
            <button
              className={matchMode === "any" ? "active" : ""}
              onClick={() => setMatchMode("any")}
            >
              Any
            </button>
            <button
              className={matchMode === "all" ? "active" : ""}
              onClick={() => setMatchMode("all")}
            >
              All
            </button>
          </div>

          <button
            className="reset-button"
            onClick={resetFilters}
            disabled={!hasActiveFilters}
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>

        {/* Radial Filter Ring */}
        <div className="radial-ring">
          <RadialFilter
            label="CDR"
            segments={segments.cdr}
            selectedIds={filters.cdr}
            onToggle={(id) => toggleFilter("cdr", id)}
            size={220}
          />
          <RadialFilter
            label="Age"
            segments={segments.age}
            selectedIds={filters.age}
            onToggle={(id) => toggleFilter("age", id)}
            size={220}
          />
          <RadialFilter
            label="Sex"
            segments={segments.sex}
            selectedIds={filters.sex}
            onToggle={(id) => toggleFilter("sex", id)}
            size={220}
          />
          <RadialFilter
            label="MMSE"
            segments={segments.mmse}
            selectedIds={filters.mmse}
            onToggle={(id) => toggleFilter("mmse", id)}
            size={220}
          />
          <RadialFilter
            label="nWBV"
            segments={segments.nwbv}
            selectedIds={filters.nwbv}
            onToggle={(id) => toggleFilter("nwbv", id)}
            size={220}
          />
        </div>

        {/* Active Filters */}
        <AnimatePresence>
          {hasActiveFilters && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="active-filters"
              style={{ display: "flex", justifyContent: "center", marginTop: 40 }}
            >
              <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginRight: 15, textTransform: "uppercase", letterSpacing: "0.1em" }}>
                Active Filters:
              </span>
              {Object.entries(filters).flatMap(([category, values]) =>
                (values as string[]).map((value: string) => (
                  <motion.button
                    key={`${category}-${value}`}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    className="active-filter-tag"
                    onClick={() => toggleFilter(category as keyof typeof filters, value)}
                  >
                    {value}
                    <X size={12} />
                  </motion.button>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Panel */}
        <div className="results-panel" style={{ maxWidth: 900, margin: "60px auto 0" }}>
          
          {/* Main Count */}
          <div className="results-count">
            <motion.div
              key={filteredData.length}
              initial={{ scale: 1.1, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="results-count-number"
            >
              {filteredData.length}
            </motion.div>
            <div className="results-count-label">Matching Subjects</div>
          </div>

          {/* Stats Grid */}
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value" style={{ color: "#10b981" }}>{normalCount}</div>
              <div className="stat-label">Normal</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ color: "#f59e0b" }}>{impairedCount}</div>
              <div className="stat-label">Impaired</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ color: "#60a5fa" }}>{avgAge}</div>
              <div className="stat-label">Avg Age</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ color: "#a78bfa" }}>{avgNwbv}</div>
              <div className="stat-label">Avg nWBV</div>
            </div>
          </div>
        </div>

        {/* Subject Table */}
        <div className="subjects-table" style={{ maxWidth: 900, margin: "40px auto 0" }}>
          <div className="subjects-table-header">
            <div className="subjects-table-title">Sample Subjects</div>
            <div className="subjects-table-count">
              {Math.min(12, filteredData.length)} of {filteredData.length}
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Subject ID</th>
                <th>Age</th>
                <th>Sex</th>
                <th>CDR</th>
                <th>MMSE</th>
                <th>nWBV</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.slice(0, 12).map((subject, i) => (
                <motion.tr
                  key={subject.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.02 }}
                >
                  <td className="mono">{subject.id}</td>
                  <td>{subject.age}</td>
                  <td style={{ color: subject.gender === "Female" ? "#f472b6" : "#60a5fa" }}>
                    {subject.gender}
                  </td>
                  <td>
                    {subject.cdr !== null ? (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "3px 10px",
                          borderRadius: 4,
                          fontSize: 11,
                          background:
                            subject.cdr === 0
                              ? "rgba(16, 185, 129, 0.15)"
                              : subject.cdr === 0.5
                              ? "rgba(245, 158, 11, 0.15)"
                              : "rgba(239, 68, 68, 0.15)",
                          color:
                            subject.cdr === 0
                              ? "#10b981"
                              : subject.cdr === 0.5
                              ? "#f59e0b"
                              : "#f87171",
                        }}
                      >
                        {subject.cdr}
                      </span>
                    ) : (
                      <span style={{ color: "rgba(255,255,255,0.3)" }}>—</span>
                    )}
                  </td>
                  <td>
                    {subject.mmse !== null ? (
                      subject.mmse
                    ) : (
                      <span style={{ color: "rgba(255,255,255,0.3)" }}>—</span>
                    )}
                  </td>
                  <td
                    style={{
                      color:
                        subject.nwbv >= 0.8
                          ? "#10b981"
                          : subject.nwbv >= 0.7
                          ? "#f59e0b"
                          : "#f87171",
                    }}
                  >
                    {subject.nwbv.toFixed(3)}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

      </div>
    </div>
  )
}
