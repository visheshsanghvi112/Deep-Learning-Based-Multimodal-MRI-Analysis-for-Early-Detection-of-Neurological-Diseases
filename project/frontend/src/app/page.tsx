"use client"

import { Alert } from "@/components/ui/alert"
import { ResearchStats } from "@/components/research-stats"
import { FeatureGrid } from "@/components/feature-grid"
import { QuickStats } from "@/components/quick-stats"
import { Hero3D } from "@/components/hero-3d"
import { ResultsDiagram } from "@/components/results-diagram"
import { RobustnessBanner } from "@/components/robustness-banner"
import Link from "next/link"
import { AlertTriangle, FileText, TrendingDown, TrendingUp, Zap, CheckCircle2, ArrowRight, Map } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex w-full flex-col gap-8">
      {/* Keep the beautiful 3D Hero */}
      <Hero3D />

      {/* Research Journey Banner */}
      <Card className="border-border">
        <CardContent className="py-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="p-2 bg-muted rounded-lg">
                <Map className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <h3 className="font-semibold">
                  New Here? Start with the Research Journey
                </h3>
                <p className="text-sm text-muted-foreground">
                  A visual step-by-step guide showing exactly what we did and discovered
                </p>
              </div>
            </div>
            <Link href="/roadmap">
              <Button variant="outline">
                View Journey <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>

      <ResearchStats />

      {/* Research Insights Grid */}
      <section className="grid gap-4 grid-cols-2 lg:grid-cols-4">
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

        <Card className="border-emerald-500/30 bg-emerald-500/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4 text-emerald-500" />
              🔬 Longitudinal
              <Badge className="ml-1 text-[10px] bg-emerald-600">NEW</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              0.83 AUC
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              +9.5% with temporal biomarker change
            </p>
          </CardContent>
        </Card>

        <Card className="border-purple-500/20 bg-purple-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-purple-500" />
              Best Predictor
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              Hippocampus
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              0.725 AUC alone - beats cognitive tests
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
        <div className="grid gap-4 md:grid-cols-3">
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

          <Card className="border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 to-transparent">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                🔬 Longitudinal Breakthrough
                <Badge className="bg-emerald-600 text-[10px]">NEW</Badge>
              </CardTitle>
              <CardDescription>Temporal biomarker analysis</CardDescription>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">✅</span>
                <p className="text-muted-foreground">Hippocampus: <strong className="text-foreground">0.725 AUC</strong> alone</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-emerald-500">✅</span>
                <p className="text-muted-foreground">+ Longitudinal Δ: <strong className="text-emerald-600">0.83 AUC</strong></p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-purple-500">🧬</span>
                <p className="text-muted-foreground">APOE4 carriers: <strong className="text-foreground">2x risk</strong></p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-500">💡</span>
                <p className="text-muted-foreground">Simple LR beats complex LSTM</p>
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
