"use client"

import { lazy, Suspense } from "react"
import { Card, CardContent } from "@/components/ui/card"

// Lazy load heavy components for better performance
export const LazyResultsDiagram = lazy(() => import("@/components/results-diagram-lazy"))
export const LazyHero3D = lazy(() => import("@/components/hero-3d-lazy"))

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

