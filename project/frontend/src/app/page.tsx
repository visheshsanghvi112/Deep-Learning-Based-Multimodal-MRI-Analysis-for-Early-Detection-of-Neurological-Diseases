"use client"

import { Alert } from "@/components/ui/alert"
import { ResearchStats } from "@/components/research-stats"
import { FeatureGrid } from "@/components/feature-grid"
import { QuickStats } from "@/components/quick-stats"
import { Hero3D } from "@/components/hero-3d"
import { ResultsDiagram } from "@/components/results-diagram"
import { RobustnessBanner } from "@/components/robustness-banner"
import Link from "next/link"
import { AlertTriangle, FileText, TrendingDown, TrendingUp, Zap, CheckCircle2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex w-full flex-col gap-8">
      {/* Keep the beautiful 3D Hero */}
      <Hero3D />

      {/* Critical Research Updates Banner */}
      <Alert className="border-blue-500/20 bg-blue-500/10">
        <Zap className="h-4 w-4 text-blue-500" />
        <div className="ml-3">
          <strong className="font-medium text-blue-700 dark:text-blue-400">
            Complete Research Documentation Available
          </strong>
          <p className="text-sm text-muted-foreground mt-1">
            Comprehensive data cleaning, honest assessment, and publication strategy now documented.
          </p>
        </div>
      </Alert>

      <ResearchStats />

      {/* Research Insights Grid */}
      <section className="grid gap-4 md:grid-cols-3">
        <Card className="border-green-500/20 bg-green-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              Data Integrity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              100%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Zero leakage verified across all experiments
            </p>
          </CardContent>
        </Card>

        <Card className="border-orange-500/20 bg-orange-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-orange-500" />
              Honest Baseline
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              0.60 AUC
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Level-1 (no cognitive scores) - realistic early detection
            </p>
          </CardContent>
        </Card>

        <Card className="border-purple-500/20 bg-purple-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-500" />
              Path to Publication
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              0.72-0.75
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Target AUC with biomarkers (Level-1.5) - 2-3 weeks
            </p>
          </CardContent>
        </Card>
      </section>

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

      {/* Key Findings Section */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold tracking-tight">Key Research Findings</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Fusion Performance Analysis</CardTitle>
              <CardDescription>Why fusion models underperform</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="text-xs">Issue</Badge>
                <p className="text-muted-foreground">
                  512 strong MRI features + 2 weak features (Age, Sex) = dimension imbalance
                </p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="text-xs">Impact</Badge>
                <p className="text-muted-foreground">
                  Clinical encoder creates 30 dims of noise, diluting MRI signal
                </p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline" className="text-xs">Solution</Badge>
                <p className="text-muted-foreground">
                  Level-1.5: Add CSF biomarkers (ABETA, TAU, PTAU) + APOE4
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Data Cleaning Rigor</CardTitle>
              <CardDescription>7 major cleaning steps applied</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-3 w-3 text-green-500" />
                <p className="text-muted-foreground">Subject-level de-duplication (1,825 to 629)</p>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-3 w-3 text-green-500" />
                <p className="text-muted-foreground">Baseline-only selection (no temporal leakage)</p>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-3 w-3 text-green-500" />
                <p className="text-muted-foreground">Subject-wise splits (zero overlap verified)</p>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-3 w-3 text-green-500" />
                <p className="text-muted-foreground">MMSE/CDR-SB excluded (no circular features)</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

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
