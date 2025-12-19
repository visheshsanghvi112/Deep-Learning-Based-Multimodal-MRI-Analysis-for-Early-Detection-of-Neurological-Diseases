"use client"

import { useState, useCallback, useEffect } from "react"

// =============================================================================
// TYPES
// =============================================================================
export interface OasisSubject {
  id: string
  gender: string
  age: number
  educ: number | null
  mmse: number | null
  cdr: number | null
  nwbv: number
  etiv: number
}

export interface FilterState {
  cdr: string[]
  age: string[]
  sex: string[]
  mmse: string[]
  nwbv: string[]
}

export interface FilterSegment {
  id: string
  label: string
  value: number
  baseColor: string
  glowColor: string
}

// =============================================================================
// FILTER HOOK
// =============================================================================
export function useRadialFilters() {
  const [data, setData] = useState<OasisSubject[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<FilterState>({
    cdr: [],
    age: [],
    sex: [],
    mmse: [],
    nwbv: [],
  })
  const [matchMode, setMatchMode] = useState<"any" | "all">("any")

  // Fetch data on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("/oasis-data.json")
        if (!res.ok) throw new Error("Failed to load dataset")
        const json = await res.json()
        setData(json)
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error")
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  // Toggle a filter value
  const toggleFilter = useCallback((category: keyof FilterState, id: string) => {
    setFilters((prev) => {
      const current = prev[category]
      const updated = current.includes(id)
        ? current.filter((v) => v !== id)
        : [...current, id]
      return { ...prev, [category]: updated }
    })
  }, [])

  // Reset all filters
  const resetFilters = useCallback(() => {
    setFilters({ cdr: [], age: [], sex: [], mmse: [], nwbv: [] })
  }, [])

  // Check if any filters are active
  const hasActiveFilters = Object.values(filters).some((arr) => arr.length > 0)

  // Filter the data
  const filteredData = data.filter((subject) => {
    const checks: boolean[] = []

    if (filters.cdr.length > 0) {
      if (subject.cdr === null) {
        checks.push(filters.cdr.includes("unassessed"))
      } else {
        const cdrMap: Record<number, string> = { 0: "normal", 0.5: "veryMild", 1: "mild", 2: "moderate" }
        checks.push(filters.cdr.includes(cdrMap[subject.cdr]))
      }
    }

    if (filters.age.length > 0) {
      const ageGroup = subject.age < 40 ? "young" : subject.age < 60 ? "adult" : subject.age < 75 ? "senior" : "elderly"
      checks.push(filters.age.includes(ageGroup))
    }

    if (filters.sex.length > 0) {
      const sexKey = subject.gender === "Female" ? "female" : "male"
      checks.push(filters.sex.includes(sexKey))
    }

    if (filters.mmse.length > 0) {
      if (subject.mmse === null) {
        checks.push(filters.mmse.includes("unassessed"))
      } else {
        const mmseGroup = subject.mmse < 24 ? "impaired" : subject.mmse < 27 ? "borderline" : "normal"
        checks.push(filters.mmse.includes(mmseGroup))
      }
    }

    if (filters.nwbv.length > 0) {
      const nwbvGroup = subject.nwbv < 0.7 ? "atrophied" : subject.nwbv < 0.8 ? "borderline" : "preserved"
      checks.push(filters.nwbv.includes(nwbvGroup))
    }

    if (checks.length === 0) return true
    return matchMode === "any" ? checks.some(Boolean) : checks.every(Boolean)
  })

  // Compute segment data from real dataset
  const cdrSegments: FilterSegment[] = [
    { id: "normal", label: "Normal", value: data.filter((s) => s.cdr === 0).length, baseColor: "#059669", glowColor: "#10b981" },
    { id: "veryMild", label: "Very Mild", value: data.filter((s) => s.cdr === 0.5).length, baseColor: "#d97706", glowColor: "#fbbf24" },
    { id: "mild", label: "Mild", value: data.filter((s) => s.cdr === 1).length, baseColor: "#ea580c", glowColor: "#fb923c" },
    { id: "moderate", label: "Moderate", value: data.filter((s) => s.cdr === 2).length, baseColor: "#dc2626", glowColor: "#f87171" },
    { id: "unassessed", label: "Unassessed", value: data.filter((s) => s.cdr === null).length, baseColor: "#4f46e5", glowColor: "#818cf8" },
  ].filter((s) => s.value > 0)

  const ageSegments: FilterSegment[] = [
    { id: "young", label: "18-39", value: data.filter((s) => s.age < 40).length, baseColor: "#0891b2", glowColor: "#22d3ee" },
    { id: "adult", label: "40-59", value: data.filter((s) => s.age >= 40 && s.age < 60).length, baseColor: "#2563eb", glowColor: "#60a5fa" },
    { id: "senior", label: "60-74", value: data.filter((s) => s.age >= 60 && s.age < 75).length, baseColor: "#7c3aed", glowColor: "#a78bfa" },
    { id: "elderly", label: "75+", value: data.filter((s) => s.age >= 75).length, baseColor: "#db2777", glowColor: "#f472b6" },
  ].filter((s) => s.value > 0)

  const sexSegments: FilterSegment[] = [
    { id: "female", label: "Female", value: data.filter((s) => s.gender === "Female").length, baseColor: "#db2777", glowColor: "#f472b6" },
    { id: "male", label: "Male", value: data.filter((s) => s.gender === "Male").length, baseColor: "#2563eb", glowColor: "#60a5fa" },
  ]

  const mmseSegments: FilterSegment[] = [
    { id: "impaired", label: "Impaired", value: data.filter((s) => s.mmse !== null && s.mmse < 24).length, baseColor: "#dc2626", glowColor: "#f87171" },
    { id: "borderline", label: "Borderline", value: data.filter((s) => s.mmse !== null && s.mmse >= 24 && s.mmse < 27).length, baseColor: "#d97706", glowColor: "#fbbf24" },
    { id: "normal", label: "Normal", value: data.filter((s) => s.mmse !== null && s.mmse >= 27).length, baseColor: "#059669", glowColor: "#10b981" },
    { id: "unassessed", label: "Unassessed", value: data.filter((s) => s.mmse === null).length, baseColor: "#4f46e5", glowColor: "#818cf8" },
  ].filter((s) => s.value > 0)

  const nwbvSegments: FilterSegment[] = [
    { id: "atrophied", label: "Atrophied", value: data.filter((s) => s.nwbv < 0.7).length, baseColor: "#dc2626", glowColor: "#f87171" },
    { id: "borderline", label: "Borderline", value: data.filter((s) => s.nwbv >= 0.7 && s.nwbv < 0.8).length, baseColor: "#d97706", glowColor: "#fbbf24" },
    { id: "preserved", label: "Preserved", value: data.filter((s) => s.nwbv >= 0.8).length, baseColor: "#059669", glowColor: "#10b981" },
  ].filter((s) => s.value > 0)

  return {
    data,
    loading,
    error,
    filters,
    matchMode,
    setMatchMode,
    toggleFilter,
    resetFilters,
    hasActiveFilters,
    filteredData,
    segments: {
      cdr: cdrSegments,
      age: ageSegments,
      sex: sexSegments,
      mmse: mmseSegments,
      nwbv: nwbvSegments,
    },
  }
}
