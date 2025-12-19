"use client"

import { useState, useMemo, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { RadialFilter, type RadialSegment } from "./radial-filter"
import { RotateCcw, Loader2, X } from "lucide-react"

// =============================================================================
// TYPES
// =============================================================================
interface OasisSubject {
  id: string
  gender: string
  age: number
  educ: number | null
  mmse: number | null
  cdr: number | null
  nwbv: number
  etiv: number
}

interface FilterState {
  diagnosis: string[]
  ageGroup: string[]
  gender: string[]
  mmseRange: string[]
  brainVolume: string[]
}

function isValidNumber(val: number | null | undefined): val is number {
  return val !== null && val !== undefined && !Number.isNaN(val)
}

// =============================================================================
// IMMERSIVE DATA EXPLORER
// =============================================================================
export function ImmersiveDataExplorer() {
  const [oasisData, setOasisData] = useState<OasisSubject[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [filters, setFilters] = useState<FilterState>({
    diagnosis: [],
    ageGroup: [],
    gender: [],
    mmseRange: [],
    brainVolume: [],
  })
  const [filterMode, setFilterMode] = useState<"any" | "all">("any")

  // Fetch real OASIS data
  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch("/oasis-data.json")
        if (!response.ok) throw new Error("Failed to load OASIS dataset")
        const data = await response.json()
        setOasisData(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Error loading data")
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const resetFilters = useCallback(() => {
    setFilters({
      diagnosis: [],
      ageGroup: [],
      gender: [],
      mmseRange: [],
      brainVolume: [],
    })
  }, [])

  const toggleFilter = useCallback(
    (category: keyof FilterState, value: string) => {
      setFilters((prev) => {
        const current = prev[category]
        const newValues = current.includes(value)
          ? current.filter((v) => v !== value)
          : [...current, value]
        return { ...prev, [category]: newValues }
      })
    },
    []
  )

  // Filtered data
  const filteredData = useMemo(() => {
    return oasisData.filter((subject) => {
      const checks: boolean[] = []

      if (filters.diagnosis.length > 0) {
        if (subject.cdr === null) {
          checks.push(filters.diagnosis.includes("unassessed"))
        } else {
          const map: Record<number, string> = {
            0: "normal",
            0.5: "veryMild",
            1: "mild",
            2: "moderate",
          }
          checks.push(filters.diagnosis.includes(map[subject.cdr]))
        }
      }
      if (filters.ageGroup.length > 0) {
        const ag =
          subject.age < 40
            ? "young"
            : subject.age < 60
            ? "adult"
            : subject.age < 75
            ? "senior"
            : "elderly"
        checks.push(filters.ageGroup.includes(ag))
      }
      if (filters.gender.length > 0) {
        const genderKey = subject.gender === "Female" ? "F" : "M"
        checks.push(filters.gender.includes(genderKey))
      }
      if (filters.mmseRange.length > 0) {
        if (subject.mmse === null) {
          checks.push(filters.mmseRange.includes("unassessed"))
        } else {
          const r =
            subject.mmse < 24
              ? "impaired"
              : subject.mmse < 27
              ? "borderline"
              : "normal"
          checks.push(filters.mmseRange.includes(r))
        }
      }
      if (filters.brainVolume.length > 0) {
        const v =
          subject.nwbv < 0.7 ? "low" : subject.nwbv < 0.8 ? "medium" : "high"
        checks.push(filters.brainVolume.includes(v))
      }

      if (checks.length === 0) return true
      return filterMode === "any" ? checks.some(Boolean) : checks.every(Boolean)
    })
  }, [oasisData, filters, filterMode])

  // Compute segments from real data
  const diagnosisSegments: RadialSegment[] = useMemo(() => {
    const normal = oasisData.filter((s) => s.cdr === 0).length
    const veryMild = oasisData.filter((s) => s.cdr === 0.5).length
    const mild = oasisData.filter((s) => s.cdr === 1).length
    const moderate = oasisData.filter((s) => s.cdr === 2).length
    const unassessed = oasisData.filter((s) => s.cdr === null).length
    return [
      { id: "normal", label: "Normal (CDR 0)", value: normal, color: "#10b981" },
      { id: "veryMild", label: "Very Mild (0.5)", value: veryMild, color: "#f59e0b" },
      { id: "mild", label: "Mild (CDR 1)", value: mild, color: "#f97316" },
      { id: "moderate", label: "Moderate (2)", value: moderate, color: "#ef4444" },
      { id: "unassessed", label: "Unassessed", value: unassessed, color: "#6366f1" },
    ].filter((s) => s.value > 0)
  }, [oasisData])

  const ageSegments: RadialSegment[] = useMemo(
    () =>
      [
        { id: "young", label: "18-39 years", value: oasisData.filter((s) => s.age < 40).length, color: "#06b6d4" },
        { id: "adult", label: "40-59 years", value: oasisData.filter((s) => s.age >= 40 && s.age < 60).length, color: "#3b82f6" },
        { id: "senior", label: "60-74 years", value: oasisData.filter((s) => s.age >= 60 && s.age < 75).length, color: "#8b5cf6" },
        { id: "elderly", label: "75+ years", value: oasisData.filter((s) => s.age >= 75).length, color: "#ec4899" },
      ].filter((s) => s.value > 0),
    [oasisData]
  )

  const genderSegments: RadialSegment[] = useMemo(
    () => [
      { id: "F", label: "Female", value: oasisData.filter((s) => s.gender === "Female").length, color: "#f472b6" },
      { id: "M", label: "Male", value: oasisData.filter((s) => s.gender === "Male").length, color: "#60a5fa" },
    ],
    [oasisData]
  )

  const mmseSegments: RadialSegment[] = useMemo(() => {
    const impaired = oasisData.filter((s) => s.mmse !== null && s.mmse < 24).length
    const borderline = oasisData.filter((s) => s.mmse !== null && s.mmse >= 24 && s.mmse < 27).length
    const normal = oasisData.filter((s) => s.mmse !== null && s.mmse >= 27).length
    const unassessed = oasisData.filter((s) => s.mmse === null).length
    return [
      { id: "impaired", label: "Impaired (<24)", value: impaired, color: "#ef4444" },
      { id: "borderline", label: "Borderline (24-26)", value: borderline, color: "#f59e0b" },
      { id: "normal", label: "Normal (27+)", value: normal, color: "#10b981" },
      { id: "unassessed", label: "Unassessed", value: unassessed, color: "#6366f1" },
    ].filter((s) => s.value > 0)
  }, [oasisData])

  const brainVolumeSegments: RadialSegment[] = useMemo(
    () =>
      [
        { id: "low", label: "Low (<0.7)", value: oasisData.filter((s) => s.nwbv < 0.7).length, color: "#ef4444" },
        { id: "medium", label: "Medium (0.7-0.8)", value: oasisData.filter((s) => s.nwbv >= 0.7 && s.nwbv < 0.8).length, color: "#f59e0b" },
        { id: "high", label: "High (≥0.8)", value: oasisData.filter((s) => s.nwbv >= 0.8).length, color: "#10b981" },
      ].filter((s) => s.value > 0),
    [oasisData]
  )

  // Stats
  const stats = useMemo(() => {
    const withCdr = filteredData.filter((s) => s.cdr !== null)
    const avgAge =
      filteredData.length > 0
        ? Math.round(filteredData.reduce((sum, s) => sum + s.age, 0) / filteredData.length)
        : 0
    const avgNwbv =
      filteredData.length > 0
        ? (filteredData.reduce((sum, s) => sum + s.nwbv, 0) / filteredData.length).toFixed(3)
        : "0"
    const impaired = withCdr.filter((s) => (s.cdr ?? 0) >= 0.5).length
    const impairedRate = withCdr.length > 0 ? Math.round((impaired / withCdr.length) * 100) : 0
    return { avgAge, avgNwbv, impaired, impairedRate, withCdr: withCdr.length }
  }, [filteredData])

  const hasFilters = Object.values(filters).some((arr) => arr.length > 0)
  const activeFilterCount = Object.values(filters).reduce((sum, arr) => sum + arr.length, 0)

  // Loading state
  if (loading) {
    return (
      <div className="min-h-[600px] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-2 border-cyan-500/30 animate-ping absolute" />
            <Loader2 className="w-16 h-16 text-cyan-400 animate-spin" />
          </div>
          <p className="text-white/60 text-sm tracking-wide">Loading OASIS-1 Dataset...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-[600px] flex items-center justify-center">
        <div className="text-center p-8 rounded-2xl bg-red-500/10 border border-red-500/30">
          <p className="text-red-400 mb-2">Error loading data</p>
          <p className="text-white/50 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div 
      className="min-h-screen p-8 relative overflow-hidden"
      style={{
        background: "radial-gradient(ellipse at center, #0f172a 0%, #020617 50%, #000000 100%)",
      }}
    >
      {/* Ambient background effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div 
          className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full opacity-20"
          style={{
            background: "radial-gradient(circle, rgba(56, 189, 248, 0.3) 0%, transparent 70%)",
            filter: "blur(80px)",
          }}
        />
        <div 
          className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full opacity-15"
          style={{
            background: "radial-gradient(circle, rgba(168, 85, 247, 0.4) 0%, transparent 70%)",
            filter: "blur(60px)",
          }}
        />
      </div>

      {/* Header */}
      <div className="relative z-10 mb-12">
        <div className="flex items-center justify-between">
          <div>
            <h1 
              className="text-3xl font-light tracking-wide mb-2"
              style={{
                background: "linear-gradient(135deg, #ffffff 0%, #94a3b8 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              OASIS-1 Research Explorer
            </h1>
            <p className="text-white/40 text-sm tracking-wide">
              Interactive cohort analysis • {oasisData.length} MRI scans • Cross-sectional study
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Filter mode toggle */}
            <div 
              className="flex rounded-full p-1"
              style={{
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
              }}
            >
              <button
                onClick={() => setFilterMode("any")}
                className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
                  filterMode === "any"
                    ? "bg-cyan-500/20 text-cyan-300 shadow-lg shadow-cyan-500/20"
                    : "text-white/40 hover:text-white/60"
                }`}
              >
                ANY
              </button>
              <button
                onClick={() => setFilterMode("all")}
                className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
                  filterMode === "all"
                    ? "bg-purple-500/20 text-purple-300 shadow-lg shadow-purple-500/20"
                    : "text-white/40 hover:text-white/60"
                }`}
              >
                ALL
              </button>
            </div>

            {/* Reset button */}
            <button
              onClick={resetFilters}
              disabled={!hasFilters}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all duration-300 ${
                hasFilters
                  ? "bg-white/5 text-white/70 hover:bg-white/10 hover:text-white border border-white/10"
                  : "opacity-30 cursor-not-allowed text-white/30"
              }`}
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Radial Filter Ring */}
      <div className="relative z-10 flex justify-center items-center gap-8 flex-wrap mb-16">
        <RadialFilter
          title="CDR"
          segments={diagnosisSegments}
          selectedIds={filters.diagnosis}
          onToggle={(id) => toggleFilter("diagnosis", id)}
          size={200}
        />
        <RadialFilter
          title="Age"
          segments={ageSegments}
          selectedIds={filters.ageGroup}
          onToggle={(id) => toggleFilter("ageGroup", id)}
          size={200}
        />
        <RadialFilter
          title="Sex"
          segments={genderSegments}
          selectedIds={filters.gender}
          onToggle={(id) => toggleFilter("gender", id)}
          size={200}
        />
        <RadialFilter
          title="MMSE"
          segments={mmseSegments}
          selectedIds={filters.mmseRange}
          onToggle={(id) => toggleFilter("mmseRange", id)}
          size={200}
        />
        <RadialFilter
          title="nWBV"
          segments={brainVolumeSegments}
          selectedIds={filters.brainVolume}
          onToggle={(id) => toggleFilter("brainVolume", id)}
          size={200}
        />
      </div>

      {/* Active filters display */}
      <AnimatePresence>
        {hasFilters && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="relative z-10 flex justify-center mb-12"
          >
            <div 
              className="flex flex-wrap items-center gap-2 px-6 py-3 rounded-full"
              style={{
                background: "rgba(255, 255, 255, 0.03)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
                backdropFilter: "blur(10px)",
              }}
            >
              <span className="text-white/40 text-xs mr-2">Active:</span>
              {Object.entries(filters).flatMap(([category, values]) =>
                (values as string[]).map((value: string) => (
                  <motion.button
                    key={`${category}-${value}`}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    onClick={() => toggleFilter(category as keyof FilterState, value)}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs bg-white/5 text-white/70 hover:bg-white/10 transition-colors border border-white/10"
                  >
                    {value}
                    <X className="w-3 h-3 text-white/40" />
                  </motion.button>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Panel */}
      <div className="relative z-10 max-w-4xl mx-auto">
        <div 
          className="rounded-2xl p-8"
          style={{
            background: "linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%)",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            backdropFilter: "blur(20px)",
          }}
        >
          {/* Main count */}
          <div className="text-center mb-8">
            <motion.div
              key={filteredData.length}
              initial={{ scale: 1.2, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
              className="text-6xl font-extralight text-white mb-2"
            >
              {filteredData.length}
            </motion.div>
            <p className="text-white/40 text-sm tracking-widest uppercase">
              Matching Subjects
            </p>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <StatCard 
              label="Normal (CDR 0)" 
              value={filteredData.filter((s) => s.cdr === 0).length.toString()}
              color="#10b981"
            />
            <StatCard 
              label="With Impairment" 
              value={stats.impaired.toString()}
              color="#f59e0b"
            />
            <StatCard 
              label="Avg Age" 
              value={`${stats.avgAge} yrs`}
              color="#60a5fa"
            />
            <StatCard 
              label="Avg nWBV" 
              value={stats.avgNwbv}
              color="#a78bfa"
            />
          </div>
        </div>

        {/* Sample subjects */}
        <div 
          className="mt-8 rounded-2xl overflow-hidden"
          style={{
            background: "rgba(255, 255, 255, 0.02)",
            border: "1px solid rgba(255, 255, 255, 0.05)",
          }}
        >
          <div className="px-6 py-4 border-b border-white/5">
            <div className="flex items-center justify-between">
              <h3 className="text-white/70 text-sm font-medium">Sample Subjects</h3>
              <span className="text-white/30 text-xs">
                {Math.min(10, filteredData.length)} of {filteredData.length}
              </span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">ID</th>
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">Age</th>
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">Sex</th>
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">CDR</th>
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">MMSE</th>
                  <th className="text-left py-3 px-4 text-white/40 text-xs font-medium">nWBV</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.slice(0, 10).map((subject, i) => (
                  <motion.tr
                    key={subject.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="border-b border-white/5 hover:bg-white/5 transition-colors"
                  >
                    <td className="py-3 px-4 font-mono text-xs text-white/60">{subject.id}</td>
                    <td className="py-3 px-4 text-white/70 text-sm">{subject.age}</td>
                    <td className="py-3 px-4">
                      <span className={subject.gender === "Female" ? "text-pink-400" : "text-blue-400"}>
                        {subject.gender}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {subject.cdr !== null ? (
                        <span
                          className="px-2 py-0.5 rounded text-xs"
                          style={{
                            background:
                              subject.cdr === 0
                                ? "rgba(16, 185, 129, 0.2)"
                                : subject.cdr === 0.5
                                ? "rgba(245, 158, 11, 0.2)"
                                : "rgba(239, 68, 68, 0.2)",
                            color:
                              subject.cdr === 0 ? "#10b981" : subject.cdr === 0.5 ? "#f59e0b" : "#ef4444",
                          }}
                        >
                          {subject.cdr}
                        </span>
                      ) : (
                        <span className="text-white/30">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-white/70 text-sm">
                      {subject.mmse !== null ? subject.mmse : <span className="text-white/30">—</span>}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={
                          subject.nwbv >= 0.8
                            ? "text-emerald-400"
                            : subject.nwbv >= 0.7
                            ? "text-amber-400"
                            : "text-red-400"
                        }
                      >
                        {subject.nwbv.toFixed(3)}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// STAT CARD COMPONENT
// =============================================================================
function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="text-center">
      <motion.div
        key={value}
        initial={{ scale: 1.1, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="text-2xl font-light mb-1"
        style={{ color }}
      >
        {value}
      </motion.div>
      <p className="text-white/40 text-xs tracking-wide">{label}</p>
    </div>
  )
}
