"use client"

import { lazy, Suspense } from "react"
import { Card, CardContent } from "@/components/ui/card"

// Lazy load heavy components provided via named exports
export const LazyResultsDiagram = lazy(() =>
  import("@/components/results-diagram").then((module) => ({ default: module.ResultsDiagram }))
)

export const LazyHero3D = lazy(() =>
  import("@/components/hero-3d").then((module) => ({ default: module.Hero3D }))
)

// Loading fallback
export function ComponentSkeleton() {
  return (
    <Card className="animate-pulse">
      <CardContent className="p-6">
        <div className="h-4 bg-muted rounded w-3/4 mb-4"></div>
        <div className="h-4 bg-muted rounded w-1/2"></div>
      </CardContent>
    </Card>
  )
}
