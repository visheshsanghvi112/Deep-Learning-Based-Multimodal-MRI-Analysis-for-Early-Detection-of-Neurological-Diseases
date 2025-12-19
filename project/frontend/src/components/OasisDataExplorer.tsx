"use client"

import { useState, useEffect, useMemo, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Brain, Users, Activity, RotateCcw, Filter, ChevronDown } from "lucide-react"

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

interface FilterSegment {
  id: string
  label: string
  value: number
  color: string
}

type FilterCategory = "cdr" | "age" | "sex" | "mmse" | "nwbv"

// =============================================================================
// RING FILTER COMPONENT - Lightweight SVG donut selector
// =============================================================================
function RingFilter({
  label,
  segments,
  selectedIds,
  onToggle,
  size = 140,
}: {
  label: string
  segments: FilterSegment[]
  selectedIds: string[]
  onToggle: (id: string) => void
  size?: number
}) {
  const total = segments.reduce((sum, s) => sum + s.value, 0)
  if (total === 0) return null

  const outerR = size / 2 - 8
  const innerR = outerR * 0.6
  const gap = 0.02 // gap between segments in radians

  // Calculate arc paths
  let currentAngle = -Math.PI / 2

  const arcs = segments.map((seg) => {
    const segmentAngle = (seg.value / total) * Math.PI * 2 - gap
    const startAngle = currentAngle
    const endAngle = currentAngle + segmentAngle
    currentAngle = endAngle + gap

    const x1 = Math.cos(startAngle) * outerR
    const y1 = Math.sin(startAngle) * outerR
    const x2 = Math.cos(endAngle) * outerR
    const y2 = Math.sin(endAngle) * outerR
    const x3 = Math.cos(endAngle) * innerR
    const y3 = Math.sin(endAngle) * innerR
    const x4 = Math.cos(startAngle) * innerR
    const y4 = Math.sin(startAngle) * innerR

    const largeArc = segmentAngle > Math.PI ? 1 : 0

    const path = `
      M ${x1} ${y1}
      A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2}
      L ${x3} ${y3}
      A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}
      Z
    `

    return { ...seg, path, midAngle: (startAngle + endAngle) / 2 }
  })

  const isSelected = (id: string) => selectedIds.includes(id)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width={size}
        height={size}
        viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`}
        className="cursor-pointer"
      >
        {arcs.map((arc) => (
          <path
            key={arc.id}
            d={arc.path}
            fill={arc.color}
            opacity={selectedIds.length === 0 || isSelected(arc.id) ? 1 : 0.3}
            stroke={isSelected(arc.id) ? arc.color : "white"}
            strokeWidth={isSelected(arc.id) ? 3 : 1}
            onClick={() => onToggle(arc.id)}
            className="transition-all duration-200 hover:opacity-80"
            style={{
              filter: isSelected(arc.id) ? `drop-shadow(0 0 4px ${arc.color})` : undefined,
            }}
          />
        ))}
        {/* Center circle */}
        <circle r={innerR - 4} fill="white" className="dark:fill-zinc-900" />
        <text
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-xs font-semibold fill-foreground"
        >
          {label}
        </text>
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-1 max-w-[160px]">
        {segments.map((seg) => (
          <button
            key={seg.id}
            onClick={() => onToggle(seg.id)}
            className={`text-[10px] px-1.5 py-0.5 rounded-full border transition-all ${
              isSelected(seg.id)
                ? "border-current font-medium"
                : selectedIds.length > 0
                ? "opacity-40 border-transparent"
                : "border-transparent hover:border-muted-foreground/30"
            }`}
            style={{ color: seg.color }}
          >
            {seg.label} ({seg.value})
          </button>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// MAIN EXPLORER COMPONENT
// =============================================================================
export function OasisDataExplorer() {
  const [data, setData] = useState<OasisSubject[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState<Record<FilterCategory, string[]>>({
    cdr: [],
    age: [],
    sex: [],
    mmse: [],
    nwbv: [],
  })
  const [matchMode, setMatchMode] = useState<"any" | "all">("any")

  // Fetch data
  useEffect(() => {
    fetch("/oasis-data.json")
      .then((res) => res.json())
      .then((json) => {
        setData(json)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // Compute segments from data
  const segments = useMemo(() => {
    const cdrCounts: Record<string, number> = {}
    const ageCounts: Record<string, number> = {}
    const sexCounts: Record<string, number> = {}
    const mmseCounts: Record<string, number> = {}
    const nwbvCounts: Record<string, number> = {}

    data.forEach((s) => {
      // CDR
      const cdrLabel = s.cdr === null ? "Unknown" : s.cdr === 0 ? "Normal" : s.cdr === 0.5 ? "Very Mild" : s.cdr === 1 ? "Mild" : "Moderate+"
      cdrCounts[cdrLabel] = (cdrCounts[cdrLabel] || 0) + 1

      // Age
      const ageLabel = s.age < 60 ? "<60" : s.age < 70 ? "60-69" : s.age < 80 ? "70-79" : s.age < 90 ? "80-89" : "90+"
      ageCounts[ageLabel] = (ageCounts[ageLabel] || 0) + 1

      // Sex
      sexCounts[s.gender] = (sexCounts[s.gender] || 0) + 1

      // MMSE
      const mmseLabel = s.mmse === null ? "Unknown" : s.mmse >= 27 ? "Normal (27+)" : s.mmse >= 24 ? "Mild (24-26)" : "Impaired (<24)"
      mmseCounts[mmseLabel] = (mmseCounts[mmseLabel] || 0) + 1

      // nWBV
      const nwbvLabel = s.nwbv >= 0.8 ? "High (≥0.8)" : s.nwbv >= 0.7 ? "Normal (0.7-0.8)" : "Low (<0.7)"
      nwbvCounts[nwbvLabel] = (nwbvCounts[nwbvLabel] || 0) + 1
    })

    const cdrColors: Record<string, string> = { Normal: "#10b981", "Very Mild": "#f59e0b", Mild: "#f97316", "Moderate+": "#ef4444", Unknown: "#94a3b8" }
    const ageColors: Record<string, string> = { "<60": "#3b82f6", "60-69": "#06b6d4", "70-79": "#8b5cf6", "80-89": "#ec4899", "90+": "#f43f5e" }
    const sexColors: Record<string, string> = { Female: "#ec4899", Male: "#3b82f6" }
    const mmseColors: Record<string, string> = { "Normal (27+)": "#10b981", "Mild (24-26)": "#f59e0b", "Impaired (<24)": "#ef4444", Unknown: "#94a3b8" }
    const nwbvColors: Record<string, string> = { "High (≥0.8)": "#10b981", "Normal (0.7-0.8)": "#3b82f6", "Low (<0.7)": "#f59e0b" }

    return {
      cdr: Object.entries(cdrCounts).map(([label, value]) => ({ id: label, label, value, color: cdrColors[label] || "#6b7280" })),
      age: ["<60", "60-69", "70-79", "80-89", "90+"].filter(k => ageCounts[k]).map(label => ({ id: label, label, value: ageCounts[label], color: ageColors[label] })),
      sex: Object.entries(sexCounts).map(([label, value]) => ({ id: label, label, value, color: sexColors[label] || "#6b7280" })),
      mmse: Object.entries(mmseCounts).map(([label, value]) => ({ id: label, label, value, color: mmseColors[label] || "#6b7280" })),
      nwbv: Object.entries(nwbvCounts).map(([label, value]) => ({ id: label, label, value, color: nwbvColors[label] || "#6b7280" })),
    }
  }, [data])

  // Filter toggle
  const toggleFilter = useCallback((category: FilterCategory, id: string) => {
    setFilters((prev) => {
      const current = prev[category]
      return {
        ...prev,
        [category]: current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
      }
    })
  }, [])

  const resetFilters = useCallback(() => {
    setFilters({ cdr: [], age: [], sex: [], mmse: [], nwbv: [] })
  }, [])

  const hasActiveFilters = Object.values(filters).some((arr) => arr.length > 0)

  // Filter data
  const filteredData = useMemo(() => {
    if (!hasActiveFilters) return data

    return data.filter((s) => {
      const checks: boolean[] = []

      if (filters.cdr.length > 0) {
        const cdrLabel = s.cdr === null ? "Unknown" : s.cdr === 0 ? "Normal" : s.cdr === 0.5 ? "Very Mild" : s.cdr === 1 ? "Mild" : "Moderate+"
        checks.push(filters.cdr.includes(cdrLabel))
      }
      if (filters.age.length > 0) {
        const ageLabel = s.age < 60 ? "<60" : s.age < 70 ? "60-69" : s.age < 80 ? "70-79" : s.age < 90 ? "80-89" : "90+"
        checks.push(filters.age.includes(ageLabel))
      }
      if (filters.sex.length > 0) {
        checks.push(filters.sex.includes(s.gender))
      }
      if (filters.mmse.length > 0) {
        const mmseLabel = s.mmse === null ? "Unknown" : s.mmse >= 27 ? "Normal (27+)" : s.mmse >= 24 ? "Mild (24-26)" : "Impaired (<24)"
        checks.push(filters.mmse.includes(mmseLabel))
      }
      if (filters.nwbv.length > 0) {
        const nwbvLabel = s.nwbv >= 0.8 ? "High (≥0.8)" : s.nwbv >= 0.7 ? "Normal (0.7-0.8)" : "Low (<0.7)"
        checks.push(filters.nwbv.includes(nwbvLabel))
      }

      return matchMode === "any" ? checks.some(Boolean) : checks.every(Boolean)
    })
  }, [data, filters, matchMode, hasActiveFilters])

  // Stats
  const stats = useMemo(() => {
    const normal = filteredData.filter((s) => s.cdr === 0).length
    const impaired = filteredData.filter((s) => s.cdr !== null && s.cdr > 0).length
    const avgAge = filteredData.length > 0 ? Math.round(filteredData.reduce((sum, s) => sum + s.age, 0) / filteredData.length) : 0
    const avgNwbv = filteredData.length > 0 ? (filteredData.reduce((sum, s) => sum + s.nwbv, 0) / filteredData.length).toFixed(3) : "0"
    return { normal, impaired, avgAge, avgNwbv }
  }, [filteredData])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Filter className="h-5 w-5 text-blue-500" />
            Interactive Data Explorer
          </h2>
          <p className="text-sm text-muted-foreground">
            Click ring segments to filter • {data.length} total subjects
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-md border text-xs">
            <button
              onClick={() => setMatchMode("any")}
              className={`px-3 py-1.5 transition-colors ${matchMode === "any" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              Any
            </button>
            <button
              onClick={() => setMatchMode("all")}
              className={`px-3 py-1.5 transition-colors ${matchMode === "all" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            >
              All
            </button>
          </div>
          <button
            onClick={resetFilters}
            disabled={!hasActiveFilters}
            className="flex items-center gap-1 px-3 py-1.5 text-xs border rounded-md hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <RotateCcw className="h-3 w-3" />
            Reset
          </button>
        </div>
      </div>

      {/* Filter Rings */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap justify-center gap-8">
            <RingFilter label="CDR" segments={segments.cdr} selectedIds={filters.cdr} onToggle={(id) => toggleFilter("cdr", id)} />
            <RingFilter label="Age" segments={segments.age} selectedIds={filters.age} onToggle={(id) => toggleFilter("age", id)} />
            <RingFilter label="Sex" segments={segments.sex} selectedIds={filters.sex} onToggle={(id) => toggleFilter("sex", id)} />
            <RingFilter label="MMSE" segments={segments.mmse} selectedIds={filters.mmse} onToggle={(id) => toggleFilter("mmse", id)} />
            <RingFilter label="nWBV" segments={segments.nwbv} selectedIds={filters.nwbv} onToggle={(id) => toggleFilter("nwbv", id)} />
          </div>
        </CardContent>
      </Card>

      {/* Active Filters */}
      <AnimatePresence>
        {hasActiveFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="flex flex-wrap items-center gap-2"
          >
            <span className="text-xs text-muted-foreground">Active:</span>
            {Object.entries(filters).flatMap(([category, values]) =>
              values.map((value) => (
                <Badge
                  key={`${category}-${value}`}
                  variant="secondary"
                  className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground"
                  onClick={() => toggleFilter(category as FilterCategory, value)}
                >
                  {value} ×
                </Badge>
              ))
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-3xl font-bold text-primary">{filteredData.length}</div>
            <div className="text-xs text-muted-foreground">Matching</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-3xl font-bold text-green-600">{stats.normal}</div>
            <div className="text-xs text-muted-foreground">Normal</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-3xl font-bold text-amber-600">{stats.impaired}</div>
            <div className="text-xs text-muted-foreground">Impaired</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-3xl font-bold text-blue-600">{stats.avgAge}</div>
            <div className="text-xs text-muted-foreground">Avg Age</div>
          </CardContent>
        </Card>
      </div>

      {/* Subject Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Users className="h-4 w-4" />
            Sample Subjects
          </CardTitle>
          <CardDescription>
            Showing {Math.min(15, filteredData.length)} of {filteredData.length} matching subjects
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Age</TableHead>
                <TableHead>Sex</TableHead>
                <TableHead>CDR</TableHead>
                <TableHead>MMSE</TableHead>
                <TableHead>nWBV</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredData.slice(0, 15).map((subject) => (
                <TableRow key={subject.id}>
                  <TableCell className="font-mono text-xs">{subject.id}</TableCell>
                  <TableCell>{subject.age}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={subject.gender === "Female" ? "text-pink-600" : "text-blue-600"}>
                      {subject.gender}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {subject.cdr !== null ? (
                      <Badge
                        variant={subject.cdr === 0 ? "default" : "secondary"}
                        className={
                          subject.cdr === 0
                            ? "bg-green-100 text-green-700 hover:bg-green-100"
                            : subject.cdr === 0.5
                            ? "bg-amber-100 text-amber-700 hover:bg-amber-100"
                            : "bg-red-100 text-red-700 hover:bg-red-100"
                        }
                      >
                        {subject.cdr}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell>{subject.mmse ?? "—"}</TableCell>
                  <TableCell className="font-mono text-xs">{subject.nwbv.toFixed(3)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
