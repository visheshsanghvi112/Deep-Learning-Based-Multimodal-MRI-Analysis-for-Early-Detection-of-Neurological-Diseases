"use client"

import { useRef, useEffect, useState, useCallback, useId } from "react"
import * as d3 from "d3"
import type { FilterSegment } from "@/hooks/useRadialFilters"

// =============================================================================
// TYPES
// =============================================================================
interface RadialFilterProps {
  label: string
  segments: FilterSegment[]
  selectedIds: string[]
  onToggle: (id: string) => void
  size?: number
}

// =============================================================================
// RADIAL FILTER COMPONENT
// D3.js powered volumetric radial selector control
// =============================================================================
export function RadialFilter({
  label,
  segments,
  selectedIds,
  onToggle,
  size = 220,
}: RadialFilterProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const uniqueId = useId().replace(/:/g, "")
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [isClient, setIsClient] = useState(false)

  const total = segments.reduce((sum, s) => sum + s.value, 0)
  const anySelected = selectedIds.length === 0
  const center = size / 2

  // Layer radii (5 layers: outer glow, arc ring, gap separators implicit, inner shadow, core)
  const outerGlowRadius = size / 2 - 4
  const arcOuterRadius = size / 2 - 16
  const arcInnerRadius = arcOuterRadius * 0.58
  const innerShadowRadius = arcInnerRadius - 4
  const coreRadius = arcInnerRadius * 0.75

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!svgRef.current || !isClient || segments.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll("*").remove()

    const defs = svg.append("defs")

    // ==========================================================================
    // LAYER 1: Outer Glow Halo Filter
    // ==========================================================================
    const outerGlow = defs.append("filter")
      .attr("id", `outer-glow-${uniqueId}`)
      .attr("x", "-100%")
      .attr("y", "-100%")
      .attr("width", "300%")
      .attr("height", "300%")

    outerGlow.append("feGaussianBlur")
      .attr("in", "SourceGraphic")
      .attr("stdDeviation", "8")
      .attr("result", "blur")

    outerGlow.append("feColorMatrix")
      .attr("in", "blur")
      .attr("type", "matrix")
      .attr("values", "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.4 0")

    // ==========================================================================
    // LAYER 2: Create radial gradients for each arc (volumetric depth)
    // ==========================================================================
    segments.forEach((seg) => {
      const isActive = selectedIds.includes(seg.id) || anySelected
      const isHovered = hoveredId === seg.id

      // Main arc gradient (inner darker, outer lighter for volume)
      const arcGradient = defs.append("radialGradient")
        .attr("id", `arc-grad-${uniqueId}-${seg.id}`)
        .attr("cx", "50%")
        .attr("cy", "50%")
        .attr("r", "50%")
        .attr("fx", "30%")
        .attr("fy", "30%")

      const brightness = isActive ? (isHovered ? 1.3 : 1.0) : 0.35
      const baseColor = d3.color(seg.baseColor)!
      const glowColor = d3.color(seg.glowColor)!

      // Inner edge (darker)
      arcGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", d3.rgb(baseColor).darker(0.8).toString())
        .attr("stop-opacity", brightness * 0.9)

      // Middle
      arcGradient.append("stop")
        .attr("offset", "50%")
        .attr("stop-color", seg.baseColor)
        .attr("stop-opacity", brightness)

      // Outer edge (lighter highlight)
      arcGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", d3.rgb(glowColor).brighter(0.3).toString())
        .attr("stop-opacity", brightness * 1.1)

      // Active glow filter for selected arcs
      if (isActive) {
        const activeGlow = defs.append("filter")
          .attr("id", `active-glow-${uniqueId}-${seg.id}`)
          .attr("x", "-50%")
          .attr("y", "-50%")
          .attr("width", "200%")
          .attr("height", "200%")

        activeGlow.append("feGaussianBlur")
          .attr("in", "SourceGraphic")
          .attr("stdDeviation", isHovered ? "6" : "4")
          .attr("result", "blur")

        activeGlow.append("feColorMatrix")
          .attr("in", "blur")
          .attr("type", "matrix")
          .attr("values", "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.6 0")

        const merge = activeGlow.append("feMerge")
        merge.append("feMergeNode").attr("in", "blur")
        merge.append("feMergeNode").attr("in", "SourceGraphic")
      }
    })

    // ==========================================================================
    // LAYER 5: Core gradient (recessed center)
    // ==========================================================================
    const coreGrad = defs.append("radialGradient")
      .attr("id", `core-${uniqueId}`)
      .attr("cx", "40%")
      .attr("cy", "40%")

    coreGrad.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#151a24")

    coreGrad.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#080a0f")

    // Inner shadow gradient
    const innerShadowGrad = defs.append("radialGradient")
      .attr("id", `inner-shadow-${uniqueId}`)
      .attr("cx", "50%")
      .attr("cy", "50%")

    innerShadowGrad.append("stop")
      .attr("offset", "70%")
      .attr("stop-color", "transparent")

    innerShadowGrad.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "rgba(0, 0, 0, 0.6)")

    // ==========================================================================
    // Main group
    // ==========================================================================
    const g = svg.append("g")
      .attr("transform", `translate(${center}, ${center})`)

    // ==========================================================================
    // LAYER 1: Outer glow halo ring
    // ==========================================================================
    g.append("circle")
      .attr("r", outerGlowRadius)
      .attr("fill", "none")
      .attr("stroke", "rgba(100, 180, 220, 0.15)")
      .attr("stroke-width", 2)
      .style("filter", `url(#outer-glow-${uniqueId})`)

    // Subtle outer decorative ring
    g.append("circle")
      .attr("r", outerGlowRadius - 2)
      .attr("fill", "none")
      .attr("stroke", "rgba(255, 255, 255, 0.03)")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3,6")

    // ==========================================================================
    // LAYER 4: Inner shadow ring (creates recessed depth)
    // ==========================================================================
    g.append("circle")
      .attr("r", innerShadowRadius)
      .attr("fill", `url(#inner-shadow-${uniqueId})`)

    // ==========================================================================
    // LAYER 5: Core circle
    // ==========================================================================
    g.append("circle")
      .attr("r", coreRadius)
      .attr("fill", `url(#core-${uniqueId})`)
      .attr("stroke", "rgba(255, 255, 255, 0.04)")
      .attr("stroke-width", 1)

    // ==========================================================================
    // LAYER 2 & 3: Primary arc ring with gaps (D3 pie + arc)
    // ==========================================================================
    const pie = d3.pie<FilterSegment>()
      .value((d) => d.value)
      .sort(null)
      .padAngle(0.04) // LAYER 3: Arc separation gaps

    const arcData = pie(segments)

    // Base arc generator
    const arcGen = d3.arc<d3.PieArcDatum<FilterSegment>>()
      .innerRadius(arcInnerRadius)
      .outerRadius(arcOuterRadius)
      .cornerRadius(5)

    // Hover arc generator (expanded)
    const hoverArcGen = d3.arc<d3.PieArcDatum<FilterSegment>>()
      .innerRadius(arcInnerRadius - 4)
      .outerRadius(arcOuterRadius + 8)
      .cornerRadius(6)

    // Draw arcs
    const arcs = g.selectAll(".arc-segment")
      .data(arcData)
      .enter()
      .append("g")
      .attr("class", "arc-segment")

    arcs.append("path")
      .attr("d", (d) => {
        const isHovered = hoveredId === d.data.id
        return isHovered ? hoverArcGen(d) : arcGen(d)
      })
      .attr("fill", (d) => `url(#arc-grad-${uniqueId}-${d.data.id})`)
      .attr("stroke", (d) => {
        const isActive = selectedIds.includes(d.data.id) || anySelected
        return isActive ? "rgba(255, 255, 255, 0.25)" : "rgba(255, 255, 255, 0.06)"
      })
      .attr("stroke-width", (d) => {
        const isHovered = hoveredId === d.data.id
        return isHovered ? 2 : 1
      })
      .style("cursor", "pointer")
      .style("filter", (d) => {
        const isActive = selectedIds.includes(d.data.id) || anySelected
        return isActive ? `url(#active-glow-${uniqueId}-${d.data.id})` : "none"
      })
      .on("mouseenter", function (event, d) {
        setHoveredId(d.data.id)
      })
      .on("mouseleave", function () {
        setHoveredId(null)
      })
      .on("click", function (event, d) {
        onToggle(d.data.id)
      })

    // ==========================================================================
    // Center text
    // ==========================================================================
    g.append("text")
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("class", "radial-center-label")
      .attr("fill", "rgba(255, 255, 255, 0.45)")
      .attr("font-size", "9px")
      .attr("font-weight", "500")
      .attr("letter-spacing", "0.12em")
      .text(label.toUpperCase())

    g.append("text")
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .attr("class", "radial-center-value")
      .attr("fill", "rgba(255, 255, 255, 0.9)")
      .attr("font-size", "20px")
      .attr("font-weight", "200")
      .text(anySelected ? "ANY" : selectedIds.length.toString())

  }, [
    segments,
    selectedIds,
    anySelected,
    hoveredId,
    size,
    center,
    uniqueId,
    label,
    isClient,
    outerGlowRadius,
    arcOuterRadius,
    arcInnerRadius,
    innerShadowRadius,
    coreRadius,
    onToggle,
  ])

  // Hovered segment for tooltip
  const hoveredSegment = segments.find((s) => s.id === hoveredId)

  if (!isClient) {
    return <div style={{ width: size, height: size }} />
  }

  return (
    <div className="radial-filter" style={{ width: size }}>
      <svg
        ref={svgRef}
        width={size}
        height={size}
        style={{ overflow: "visible" }}
      />

      {/* Tooltip */}
      {hoveredSegment && (
        <div className="radial-tooltip">
          <div className="radial-tooltip-label">{hoveredSegment.label}</div>
          <div className="radial-tooltip-value">
            {hoveredSegment.value} subjects • {((hoveredSegment.value / total) * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  )
}
