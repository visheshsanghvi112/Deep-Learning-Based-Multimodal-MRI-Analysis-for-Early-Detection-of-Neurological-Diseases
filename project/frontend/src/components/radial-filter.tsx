"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import * as d3 from "d3"

// =============================================================================
// TYPES
// =============================================================================
export interface RadialSegment {
  id: string
  label: string
  value: number
  color: string
}

interface RadialFilterProps {
  title: string
  segments: RadialSegment[]
  selectedIds: string[]
  onToggle: (id: string) => void
  size?: number
}

// =============================================================================
// RADIAL FILTER COMPONENT - D3.js Powered
// =============================================================================
export function RadialFilter({
  title,
  segments,
  selectedIds,
  onToggle,
  size = 200,
}: RadialFilterProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [mounted, setMounted] = useState(false)

  const total = segments.reduce((sum, s) => sum + s.value, 0)
  const allSelected = selectedIds.length === 0
  const center = size / 2

  // Radii for layered ring effect
  const outerRadius = size / 2 - 10
  const innerRadius = outerRadius * 0.55
  const coreRadius = innerRadius * 0.7
  const glowRadius = outerRadius + 8

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!svgRef.current || !mounted) return

    const svg = d3.select(svgRef.current)
    
    // Clear previous content
    svg.selectAll("*").remove()

    // Create defs for gradients and filters
    const defs = svg.append("defs")

    // Outer glow filter
    const glowFilter = defs.append("filter")
      .attr("id", `glow-${title.replace(/\s/g, '')}`)
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%")

    glowFilter.append("feGaussianBlur")
      .attr("stdDeviation", "4")
      .attr("result", "coloredBlur")

    const glowMerge = glowFilter.append("feMerge")
    glowMerge.append("feMergeNode").attr("in", "coloredBlur")
    glowMerge.append("feMergeNode").attr("in", "SourceGraphic")

    // Inner shadow filter
    const innerShadow = defs.append("filter")
      .attr("id", `inner-shadow-${title.replace(/\s/g, '')}`)
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%")

    innerShadow.append("feOffset")
      .attr("dx", "0")
      .attr("dy", "2")

    innerShadow.append("feGaussianBlur")
      .attr("stdDeviation", "3")
      .attr("result", "shadow")

    innerShadow.append("feComposite")
      .attr("operator", "out")
      .attr("in", "SourceGraphic")
      .attr("in2", "shadow")
      .attr("result", "inverse")

    innerShadow.append("feFlood")
      .attr("flood-color", "black")
      .attr("flood-opacity", "0.4")

    innerShadow.append("feComposite")
      .attr("operator", "in")
      .attr("in2", "inverse")

    innerShadow.append("feComposite")
      .attr("operator", "over")
      .attr("in2", "SourceGraphic")

    // Create radial gradients for each segment
    segments.forEach((segment, i) => {
      const gradient = defs.append("radialGradient")
        .attr("id", `gradient-${title.replace(/\s/g, '')}-${segment.id}`)
        .attr("cx", "30%")
        .attr("cy", "30%")
        .attr("r", "70%")

      const isActive = selectedIds.includes(segment.id) || allSelected
      const baseColor = d3.color(segment.color)!
      
      gradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", isActive ? d3.rgb(baseColor).brighter(0.8).toString() : d3.rgb(baseColor).darker(0.5).toString())
        .attr("stop-opacity", isActive ? 1 : 0.4)

      gradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", isActive ? segment.color : d3.rgb(baseColor).darker(1.5).toString())
        .attr("stop-opacity", isActive ? 0.9 : 0.3)
    })

    // Main group centered
    const g = svg.append("g")
      .attr("transform", `translate(${center}, ${center})`)

    // Outer ambient glow ring
    g.append("circle")
      .attr("r", glowRadius)
      .attr("fill", "none")
      .attr("stroke", "rgba(100, 200, 255, 0.1)")
      .attr("stroke-width", 2)
      .style("filter", `url(#glow-${title.replace(/\s/g, '')})`)

    // Background recessed core
    const coreGradient = defs.append("radialGradient")
      .attr("id", `core-gradient-${title.replace(/\s/g, '')}`)
      .attr("cx", "40%")
      .attr("cy", "40%")

    coreGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#1a1a2e")

    coreGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#0a0a15")

    g.append("circle")
      .attr("r", coreRadius)
      .attr("fill", `url(#core-gradient-${title.replace(/\s/g, '')})`)
      .style("filter", `url(#inner-shadow-${title.replace(/\s/g, '')})`)

    // Inner ring border
    g.append("circle")
      .attr("r", innerRadius)
      .attr("fill", "none")
      .attr("stroke", "rgba(255, 255, 255, 0.05)")
      .attr("stroke-width", 1)

    // Create pie layout
    const pie = d3.pie<RadialSegment>()
      .value(d => d.value)
      .sort(null)
      .padAngle(0.03)

    const arcData = pie(segments)

    // Arc generator
    const arcGenerator = d3.arc<d3.PieArcDatum<RadialSegment>>()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .cornerRadius(4)

    // Hover arc generator (slightly larger)
    const hoverArcGenerator = d3.arc<d3.PieArcDatum<RadialSegment>>()
      .innerRadius(innerRadius - 3)
      .outerRadius(outerRadius + 6)
      .cornerRadius(5)

    // Draw arcs
    const arcs = g.selectAll(".arc")
      .data(arcData)
      .enter()
      .append("g")
      .attr("class", "arc")

    arcs.append("path")
      .attr("d", d => arcGenerator(d)!)
      .attr("fill", d => `url(#gradient-${title.replace(/\s/g, '')}-${d.data.id})`)
      .attr("stroke", d => {
        const isActive = selectedIds.includes(d.data.id) || allSelected
        return isActive ? "rgba(255, 255, 255, 0.3)" : "rgba(255, 255, 255, 0.08)"
      })
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .style("filter", d => {
        const isActive = selectedIds.includes(d.data.id) || allSelected
        return isActive ? `url(#glow-${title.replace(/\s/g, '')})` : "none"
      })
      .style("transition", "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)")
      .on("mouseenter", function(event, d) {
        setHoveredId(d.data.id)
        d3.select(this)
          .transition()
          .duration(200)
          .ease(d3.easeCubicOut)
          .attr("d", hoverArcGenerator(d)!)
          .attr("stroke", "rgba(255, 255, 255, 0.5)")
          .attr("stroke-width", 2)
      })
      .on("mouseleave", function(event, d) {
        setHoveredId(null)
        const isActive = selectedIds.includes(d.data.id) || allSelected
        d3.select(this)
          .transition()
          .duration(300)
          .ease(d3.easeCubicInOut)
          .attr("d", arcGenerator(d)!)
          .attr("stroke", isActive ? "rgba(255, 255, 255, 0.3)" : "rgba(255, 255, 255, 0.08)")
          .attr("stroke-width", 1)
      })
      .on("click", function(event, d) {
        onToggle(d.data.id)
        
        // Click ripple effect
        const [x, y] = d3.arc<d3.PieArcDatum<RadialSegment>>()
          .innerRadius(innerRadius)
          .outerRadius(outerRadius)
          .centroid(d)
        
        g.append("circle")
          .attr("cx", x)
          .attr("cy", y)
          .attr("r", 5)
          .attr("fill", "rgba(255, 255, 255, 0.6)")
          .transition()
          .duration(400)
          .attr("r", 25)
          .style("opacity", 0)
          .remove()
      })

    // Center text
    const centerGroup = g.append("g").attr("class", "center-text")

    // Title (small, above count)
    centerGroup.append("text")
      .attr("y", -12)
      .attr("text-anchor", "middle")
      .attr("fill", "rgba(255, 255, 255, 0.5)")
      .attr("font-size", "10px")
      .attr("font-weight", "500")
      .attr("letter-spacing", "0.5px")
      .text(title.toUpperCase())

    // Count or status
    centerGroup.append("text")
      .attr("y", 12)
      .attr("text-anchor", "middle")
      .attr("fill", "rgba(255, 255, 255, 0.95)")
      .attr("font-size", "22px")
      .attr("font-weight", "600")
      .text(allSelected ? "ANY" : selectedIds.length.toString())

    // Outer decorative ring
    g.append("circle")
      .attr("r", outerRadius + 2)
      .attr("fill", "none")
      .attr("stroke", "rgba(255, 255, 255, 0.05)")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,4")

  }, [segments, selectedIds, allSelected, size, title, center, outerRadius, innerRadius, coreRadius, glowRadius, mounted, onToggle])

  // Tooltip
  const hoveredSegment = segments.find(s => s.id === hoveredId)

  return (
    <div className="relative flex flex-col items-center">
      <svg
        ref={svgRef}
        width={size}
        height={size}
        className="overflow-visible"
      />
      
      {/* Floating tooltip */}
      {hoveredSegment && (
        <div 
          className="absolute pointer-events-none z-10 px-3 py-2 rounded-lg text-xs"
          style={{
            top: size + 8,
            background: "rgba(0, 0, 0, 0.85)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            backdropFilter: "blur(10px)",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
          }}
        >
          <div className="flex items-center gap-2">
            <div 
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: hoveredSegment.color }}
            />
            <span className="text-white/90 font-medium">{hoveredSegment.label}</span>
            <span className="text-white/60">({hoveredSegment.value})</span>
          </div>
        </div>
      )}
    </div>
  )
}
