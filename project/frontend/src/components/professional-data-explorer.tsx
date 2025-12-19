"use client"

import { useState, useMemo, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { RotateCcw, Users, Brain, Activity, TrendingUp, Check, Loader2, Database, BarChart3 } from "lucide-react"

// =============================================================================
// TYPES FOR REAL OASIS DATA
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

// =============================================================================
// TYPES
// =============================================================================
interface FilterState {
  diagnosis: string[]
  ageGroup: string[]
  gender: string[]
  mmseRange: string[]
  brainVolume: string[]
}

interface DonutSegment {
  id: string
  label: string
  value: number
  color: string
}

// Helper to check for NaN values from JSON
function isValidNumber(val: number | null | undefined): val is number {
  return val !== null && val !== undefined && !Number.isNaN(val)
}

// =============================================================================
// CLEAN DONUT CHART COMPONENT
// =============================================================================
function DonutChart({
  title,
  description,
  segments,
  selectedIds,
  onToggle,
  size = 160,
}: {
  title: string
  description?: string
  segments: DonutSegment[]
  selectedIds: string[]
  onToggle: (id: string) => void
  size?: number
}) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  
  const total = segments.reduce((sum, s) => sum + s.value, 0)
  const center = size / 2
  const outerRadius = (size / 2) - 8
  const innerRadius = outerRadius * 0.6
  const allSelected = selectedIds.length === 0

  // Calculate arc paths
  const arcs = useMemo(() => {
    let currentAngle = -90
    return segments.map((segment) => {
      const percentage = segment.value / total
      const angle = percentage * 360
      const startAngle = currentAngle
      const endAngle = currentAngle + angle
      currentAngle = endAngle

      const startRad = (startAngle * Math.PI) / 180
      const endRad = (endAngle * Math.PI) / 180

      const x1 = center + outerRadius * Math.cos(startRad)
      const y1 = center + outerRadius * Math.sin(startRad)
      const x2 = center + outerRadius * Math.cos(endRad)
      const y2 = center + outerRadius * Math.sin(endRad)
      const x3 = center + innerRadius * Math.cos(endRad)
      const y3 = center + innerRadius * Math.sin(endRad)
      const x4 = center + innerRadius * Math.cos(startRad)
      const y4 = center + innerRadius * Math.sin(startRad)

      const largeArc = angle > 180 ? 1 : 0

      return {
        ...segment,
        path: `M ${x1} ${y1} A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x4} ${y4} Z`,
        percentage,
        isSelected: selectedIds.includes(segment.id),
      }
    })
  }, [segments, total, selectedIds, center, outerRadius, innerRadius])

  const selectedCount = selectedIds.length

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
      </CardHeader>
      <CardContent className="flex flex-col items-center pt-0">
        {/* Donut Chart */}
        <svg width={size} height={size} className="mb-3">
          {arcs.map((arc, i) => (
            <motion.path
              key={arc.id}
              d={arc.path}
              fill={arc.isSelected || allSelected ? arc.color : `${arc.color}30`}
              stroke={hoveredId === arc.id ? "var(--foreground)" : "var(--background)"}
              strokeWidth={hoveredId === arc.id ? 2 : 1}
              className="cursor-pointer transition-colors"
              onClick={() => onToggle(arc.id)}
              onMouseEnter={() => setHoveredId(arc.id)}
              onMouseLeave={() => setHoveredId(null)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.1 }}
            />
          ))}
          {/* Center text */}
          <text
            x={center}
            y={center - 6}
            textAnchor="middle"
            className="fill-muted-foreground text-[10px] font-medium"
          >
            {allSelected ? "Any" : `${selectedCount} sel`}
          </text>
          <text
            x={center}
            y={center + 10}
            textAnchor="middle"
            className="fill-foreground text-lg font-semibold"
          >
            {total}
          </text>
        </svg>

        {/* Legend with checkboxes */}
        <div className="flex flex-wrap justify-center gap-2 w-full">
          {segments.map((segment) => (
            <button
              key={segment.id}
              onClick={() => onToggle(segment.id)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs transition-all border
                ${selectedIds.includes(segment.id) || allSelected
                  ? "bg-muted border-border"
                  : "bg-transparent border-transparent opacity-50 hover:opacity-75"
                }`}
            >
              <div
                className="w-3 h-3 rounded-sm flex items-center justify-center border"
                style={{
                  backgroundColor: selectedIds.includes(segment.id) || allSelected ? segment.color : "transparent",
                  borderColor: segment.color,
                }}
              >
                {(selectedIds.includes(segment.id) || allSelected) && (
                  <Check className="w-2 h-2 text-white" />
                )}
              </div>
              <span className="text-muted-foreground">{segment.label}</span>
              <span className="text-foreground font-medium">{segment.value}</span>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function ProfessionalDataExplorer() {
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
  const [filterMode, setFilterMode] = useState<'any' | 'all'>('any')

  // Fetch real OASIS data
  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch('/oasis-data.json')
        if (!response.ok) throw new Error('Failed to load data')
        const data = await response.json()
        // Convert NaN strings/null to proper null values
        const cleanData = data.map((subject: Record<string, unknown>) => ({
          ...subject,
          educ: isValidNumber(subject.educ as number) ? subject.educ : null,
          mmse: isValidNumber(subject.mmse as number) ? subject.mmse : null,
          cdr: isValidNumber(subject.cdr as number) ? subject.cdr : null,
        }))
        setOasisData(cleanData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error loading data')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const resetFilters = useCallback(() => {
    setFilters({ diagnosis: [], ageGroup: [], gender: [], mmseRange: [], brainVolume: [] })
  }, [])

  const toggleFilter = useCallback((category: keyof FilterState, value: string) => {
    setFilters(prev => {
      const current = prev[category]
      const newValues = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value]
      return { ...prev, [category]: newValues }
    })
  }, [])

  // Filtered data with real OASIS subjects
  const filteredData = useMemo(() => {
    return oasisData.filter(subject => {
      const checks: boolean[] = []

      if (filters.diagnosis.length > 0) {
        if (subject.cdr === null) {
          checks.push(filters.diagnosis.includes('unassessed'))
        } else {
          const map: Record<number, string> = { 0: 'normal', 0.5: 'veryMild', 1: 'mild', 2: 'moderate' }
          checks.push(filters.diagnosis.includes(map[subject.cdr]))
        }
      }
      if (filters.ageGroup.length > 0) {
        let ag = subject.age < 40 ? 'young' : subject.age < 60 ? 'adult' : subject.age < 75 ? 'senior' : 'elderly'
        checks.push(filters.ageGroup.includes(ag))
      }
      if (filters.gender.length > 0) {
        const genderKey = subject.gender === 'Female' ? 'F' : 'M'
        checks.push(filters.gender.includes(genderKey))
      }
      if (filters.mmseRange.length > 0) {
        if (subject.mmse === null) {
          checks.push(filters.mmseRange.includes('unassessed'))
        } else {
          let r = subject.mmse < 24 ? 'impaired' : subject.mmse < 27 ? 'borderline' : 'normal'
          checks.push(filters.mmseRange.includes(r))
        }
      }
      if (filters.brainVolume.length > 0) {
        let v = subject.nwbv < 0.7 ? 'low' : subject.nwbv < 0.8 ? 'medium' : 'high'
        checks.push(filters.brainVolume.includes(v))
      }

      if (checks.length === 0) return true
      return filterMode === 'any' ? checks.some(Boolean) : checks.every(Boolean)
    })
  }, [oasisData, filters, filterMode])

  // Compute real segment data from actual OASIS subjects
  const diagnosisSegments: DonutSegment[] = useMemo(() => {
    const normal = oasisData.filter(s => s.cdr === 0).length
    const veryMild = oasisData.filter(s => s.cdr === 0.5).length
    const mild = oasisData.filter(s => s.cdr === 1).length
    const moderate = oasisData.filter(s => s.cdr === 2).length
    const unassessed = oasisData.filter(s => s.cdr === null).length
    return [
      { id: 'normal', label: 'Normal (CDR 0)', value: normal, color: '#22c55e' },
      { id: 'veryMild', label: 'Very Mild (0.5)', value: veryMild, color: '#eab308' },
      { id: 'mild', label: 'Mild (CDR 1)', value: mild, color: '#f97316' },
      { id: 'moderate', label: 'Moderate (2)', value: moderate, color: '#ef4444' },
      { id: 'unassessed', label: 'Unassessed', value: unassessed, color: '#94a3b8' },
    ].filter(s => s.value > 0)
  }, [oasisData])

  const ageSegments: DonutSegment[] = useMemo(() => [
    { id: 'young', label: '18-39', value: oasisData.filter(s => s.age < 40).length, color: '#06b6d4' },
    { id: 'adult', label: '40-59', value: oasisData.filter(s => s.age >= 40 && s.age < 60).length, color: '#3b82f6' },
    { id: 'senior', label: '60-74', value: oasisData.filter(s => s.age >= 60 && s.age < 75).length, color: '#8b5cf6' },
    { id: 'elderly', label: '75+', value: oasisData.filter(s => s.age >= 75).length, color: '#ec4899' },
  ].filter(s => s.value > 0), [oasisData])

  const genderSegments: DonutSegment[] = useMemo(() => [
    { id: 'F', label: 'Female', value: oasisData.filter(s => s.gender === 'Female').length, color: '#ec4899' },
    { id: 'M', label: 'Male', value: oasisData.filter(s => s.gender === 'Male').length, color: '#3b82f6' },
  ], [oasisData])

  const mmseSegments: DonutSegment[] = useMemo(() => {
    const impaired = oasisData.filter(s => s.mmse !== null && s.mmse < 24).length
    const borderline = oasisData.filter(s => s.mmse !== null && s.mmse >= 24 && s.mmse < 27).length
    const normal = oasisData.filter(s => s.mmse !== null && s.mmse >= 27).length
    const unassessed = oasisData.filter(s => s.mmse === null).length
    return [
      { id: 'impaired', label: 'Impaired (<24)', value: impaired, color: '#ef4444' },
      { id: 'borderline', label: 'Borderline (24-26)', value: borderline, color: '#eab308' },
      { id: 'normal', label: 'Normal (27+)', value: normal, color: '#22c55e' },
      { id: 'unassessed', label: 'Unassessed', value: unassessed, color: '#94a3b8' },
    ].filter(s => s.value > 0)
  }, [oasisData])

  const brainVolumeSegments: DonutSegment[] = useMemo(() => [
    { id: 'low', label: 'Low (<0.7)', value: oasisData.filter(s => s.nwbv < 0.7).length, color: '#ef4444' },
    { id: 'medium', label: 'Medium (0.7-0.8)', value: oasisData.filter(s => s.nwbv >= 0.7 && s.nwbv < 0.8).length, color: '#eab308' },
    { id: 'high', label: 'High (≥0.8)', value: oasisData.filter(s => s.nwbv >= 0.8).length, color: '#22c55e' },
  ].filter(s => s.value > 0), [oasisData])

  // Compute statistics from filtered data
  const stats = useMemo(() => {
    const withCdr = filteredData.filter(s => s.cdr !== null)
    const avgAge = filteredData.length > 0 
      ? Math.round(filteredData.reduce((sum, s) => sum + s.age, 0) / filteredData.length) 
      : 0
    const avgNwbv = filteredData.length > 0
      ? (filteredData.reduce((sum, s) => sum + s.nwbv, 0) / filteredData.length).toFixed(3)
      : '0'
    const impaired = withCdr.filter(s => (s.cdr ?? 0) >= 0.5).length
    const impairedRate = withCdr.length > 0 ? Math.round((impaired / withCdr.length) * 100) : 0
    return { avgAge, avgNwbv, impaired, impairedRate, withCdr: withCdr.length }
  }, [filteredData])

  const hasFilters = Object.values(filters).some(arr => arr.length > 0)

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading OASIS-1 dataset...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center py-20">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <p className="text-destructive mb-2">Error loading data</p>
            <p className="text-muted-foreground text-sm">{error}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with Dataset Info */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Database className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold">OASIS-1 Interactive Explorer</h3>
          </div>
          <p className="text-sm text-muted-foreground">
            Real-time filtering of {oasisData.length} MRI scans from the Open Access Series of Imaging Studies
          </p>
        </div>
        <button
          onClick={resetFilters}
          disabled={!hasFilters}
          className="flex items-center gap-2 px-3 py-2 text-sm rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RotateCcw className="h-4 w-4" />
          Reset Filters
        </button>
      </div>

      {/* Filter Charts Grid - 5 Charts */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <DonutChart
          title="CDR Classification"
          description="Clinical Dementia Rating"
          segments={diagnosisSegments}
          selectedIds={filters.diagnosis}
          onToggle={(id) => toggleFilter('diagnosis', id)}
        />
        <DonutChart
          title="Age Distribution"
          description="Age at MRI scan"
          segments={ageSegments}
          selectedIds={filters.ageGroup}
          onToggle={(id) => toggleFilter('ageGroup', id)}
        />
        <DonutChart
          title="Gender"
          description="Biological sex"
          segments={genderSegments}
          selectedIds={filters.gender}
          onToggle={(id) => toggleFilter('gender', id)}
        />
        <DonutChart
          title="MMSE Score"
          description="Mini Mental State Exam"
          segments={mmseSegments}
          selectedIds={filters.mmseRange}
          onToggle={(id) => toggleFilter('mmseRange', id)}
        />
        <DonutChart
          title="Brain Volume"
          description="Normalized WBV"
          segments={brainVolumeSegments}
          selectedIds={filters.brainVolume}
          onToggle={(id) => toggleFilter('brainVolume', id)}
        />
      </div>

      {/* Filter Mode */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-center gap-3 text-sm">
            <span className="text-muted-foreground">Searching for subjects that match</span>
            <div className="flex rounded-md overflow-hidden border border-border">
              <button
                onClick={() => setFilterMode('any')}
                className={`px-3 py-1 text-sm font-medium transition-colors ${
                  filterMode === 'any'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                Any
              </button>
              <button
                onClick={() => setFilterMode('all')}
                className={`px-3 py-1 text-sm font-medium transition-colors ${
                  filterMode === 'all'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                All
              </button>
            </div>
            <span className="text-muted-foreground">selected criteria</span>
          </div>
        </CardContent>
      </Card>

      {/* Results Summary - 6 Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <Users className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <motion.div
                  key={filteredData.length}
                  initial={{ scale: 1.1 }}
                  animate={{ scale: 1 }}
                  className="text-2xl font-semibold"
                >
                  {filteredData.length}
                </motion.div>
                <div className="text-xs text-muted-foreground">Matching Scans</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <Activity className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <div className="text-2xl font-semibold">
                  {filteredData.filter(s => s.cdr === 0).length}
                </div>
                <div className="text-xs text-muted-foreground">Normal (CDR 0)</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Brain className="h-5 w-5 text-orange-500" />
              </div>
              <div>
                <div className="text-2xl font-semibold">
                  {stats.impaired}
                </div>
                <div className="text-xs text-muted-foreground">With Impairment</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <TrendingUp className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <div className="text-2xl font-semibold">
                  {stats.impairedRate}%
                </div>
                <div className="text-xs text-muted-foreground">Impairment Rate</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-cyan-500/10">
                <BarChart3 className="h-5 w-5 text-cyan-500" />
              </div>
              <div>
                <div className="text-2xl font-semibold">
                  {stats.avgAge}
                </div>
                <div className="text-xs text-muted-foreground">Avg Age (yrs)</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-pink-500/10">
                <Brain className="h-5 w-5 text-pink-500" />
              </div>
              <div>
                <div className="text-2xl font-semibold">
                  {stats.avgNwbv}
                </div>
                <div className="text-xs text-muted-foreground">Avg nWBV</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Filters */}
      <AnimatePresence>
        {hasFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex flex-wrap items-center gap-2"
          >
            <span className="text-sm text-muted-foreground">Active filters:</span>
            {Object.entries(filters).flatMap(([category, values]) =>
              (values as string[]).map((value: string) => (
                <Badge
                  key={`${category}-${value}`}
                  variant="secondary"
                  className="cursor-pointer hover:bg-destructive/20 transition-colors"
                  onClick={() => toggleFilter(category as keyof FilterState, value)}
                >
                  {value}
                  <span className="ml-1 text-muted-foreground hover:text-destructive">×</span>
                </Badge>
              ))
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sample Subjects Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm">Matching OASIS-1 Subjects</CardTitle>
              <CardDescription>Real subjects from the dataset with clinical assessments</CardDescription>
            </div>
            <Badge variant="outline">
              Showing {Math.min(15, filteredData.length)} of {filteredData.length}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Subject ID</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Age</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">Gender</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">CDR</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">MMSE</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">nWBV</th>
                  <th className="text-left py-2 px-3 text-muted-foreground font-medium">eTIV</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.slice(0, 15).map((subject, i) => (
                  <motion.tr
                    key={subject.id}
                    className="border-b border-border/50 hover:bg-muted/50 transition-colors"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.03 }}
                  >
                    <td className="py-2 px-3 font-mono text-xs">{subject.id}</td>
                    <td className="py-2 px-3">{subject.age}</td>
                    <td className="py-2 px-3">
                      <span className={subject.gender === 'Female' ? 'text-pink-500' : 'text-blue-500'}>
                        {subject.gender}
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      {subject.cdr !== null ? (
                        <Badge
                          variant={subject.cdr === 0 ? 'default' : subject.cdr === 0.5 ? 'secondary' : 'destructive'}
                          className={subject.cdr === 0 ? 'bg-green-500/20 text-green-700 dark:text-green-400' : ''}
                        >
                          {subject.cdr}
                        </Badge>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      {subject.mmse !== null ? (
                        <span className={subject.mmse >= 27 ? 'text-green-500' : subject.mmse >= 24 ? 'text-yellow-500' : 'text-red-500'}>
                          {subject.mmse}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      <span className={subject.nwbv >= 0.8 ? 'text-green-500' : subject.nwbv >= 0.7 ? 'text-yellow-500' : 'text-red-500'}>
                        {subject.nwbv.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-muted-foreground">{subject.etiv}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
