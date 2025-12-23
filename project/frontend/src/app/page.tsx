"use client"

import { Alert } from "@/components/ui/alert"
import { ResearchStats } from "@/components/research-stats"
import { FeatureGrid } from "@/components/feature-grid"
import { QuickStats } from "@/components/quick-stats"
import { Hero3D } from "@/components/hero-3d"
import { ResultsDiagram } from "@/components/results-diagram"
import { RobustnessBanner } from "@/components/robustness-banner"
import Link from "next/link"
import Image from "next/image"
import { AlertTriangle } from "lucide-react"

export default function Home() {
  return (
    <div className="flex w-full flex-col gap-8">
      {/* Keep the beautiful 3D Hero */}
      <Hero3D />

      <ResearchStats />

      {/* Main Feature Bento Grid */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold tracking-tight">Research Portal</h2>
          <Link href="/dataset" className="text-sm text-muted-foreground hover:text-foreground">
            View All &rarr;
          </Link>
        </div>
        <FeatureGrid />
      </section>

      {/* Cross-Dataset Highlight Banner (Animated) */}
      <RobustnessBanner />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold tracking-tight">Quick Statistics</h2>
        <QuickStats />
      </section>

      <ResultsDiagram />

      {/* Warning Section */}
      <Alert className="border-orange-500/20 bg-orange-500/10">
        <AlertTriangle className="h-4 w-4 text-orange-500" />
        <div className="ml-3 text-sm">
          <strong className="font-medium text-orange-700 dark:text-orange-400">Label Shift Warning: </strong>
          <span className="text-muted-foreground">
            OASIS targets very mild dementia (CDR 0.5), while ADNI includes a broader spectrum (MCI/AD).
            This definition shift contributes to the observed accuracy drops during transfer.
          </span>
        </div>
      </Alert>

      <div className="text-center text-xs text-muted-foreground">
        NeuroScope Research Portal · Validated on OASIS-1 & ADNI-1 · Not for Clinical Use
      </div>
    </div>
  )
}
