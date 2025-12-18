"use client"

import { memo, Suspense } from "react"
import dynamic from "next/dynamic"

// Dynamically import the heavy 3D background component
const NeuroscienceBackground = dynamic(
  () => import("@/components/neuroscience-background").then(mod => ({ 
    default: mod.NeuroscienceBackground 
  })),
  { 
    ssr: false // Disable SSR for 3D components
  }
)

export const OptimizedBackground = memo(function OptimizedBackground() {
  return (
    <Suspense fallback={null}>
      <NeuroscienceBackground />
    </Suspense>
  )
})

