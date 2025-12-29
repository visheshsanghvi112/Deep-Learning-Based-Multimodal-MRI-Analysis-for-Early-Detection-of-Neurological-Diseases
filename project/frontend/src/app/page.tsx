"use client"

import { Alert } from "@/components/ui/alert"
import { ResearchStats } from "@/components/research-stats"
import { FeatureGrid } from "@/components/feature-grid"
import { QuickStats } from "@/components/quick-stats"
import { Hero3D } from "@/components/hero-3d"
import { ResultsDiagram } from "@/components/results-diagram"
import { RobustnessBanner } from "@/components/robustness-banner"
import Link from "next/link"
import { AlertTriangle, FileText, TrendingDown, TrendingUp, Zap, CheckCircle2, ArrowRight, Map, Sparkles } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  SpotlightCard,
  TextGradient,
  AnimatedCounter,
  RevealOnScroll,
  MagneticButton,
  Card3D,
  GridBackground
} from "@/components/ui/aceternity-effects"

export default function Home() {
  return (
    <div className="flex w-full flex-col gap-8">
      {/* Keep the beautiful 3D Hero */}
      <Hero3D />

      {/* Research Journey Banner - With Spotlight Effect */}
      <RevealOnScroll delay={0.1}>
        <SpotlightCard className="p-0" spotlightColor="rgba(139, 92, 246, 0.15)">
          <CardContent className="py-5 px-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className="p-2.5 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-xl">
                  <Map className="h-5 w-5 text-purple-500" />
                </div>
                <div>
                  <h3 className="font-semibold flex items-center gap-2">
                    New Here? Start with the Research Journey
                    <Sparkles className="h-4 w-4 text-yellow-500" />
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    A visual step-by-step guide showing exactly what we did and discovered
                  </p>
                </div>
              </div>
              <MagneticButton>
                <Link href="/roadmap">
                  <Button variant="outline" className="group relative overflow-hidden">
                    <span className="relative z-10">View Journey</span>
                    <ArrowRight className="h-4 w-4 ml-2 relative z-10 group-hover:translate-x-1 transition-transform" />
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </Button>
                </Link>
              </MagneticButton>
            </div>
          </CardContent>
        </SpotlightCard>
      </RevealOnScroll>

      <ResearchStats />

      {/* Research Insights Grid - With Animated Counters & 3D Cards */}
      <section className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <RevealOnScroll delay={0}>
          <Card3D>
            <SpotlightCard className="h-full border-green-500/20 bg-gradient-to-br from-green-500/10 to-transparent" spotlightColor="rgba(34, 197, 94, 0.15)">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Data Integrity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  <AnimatedCounter value={100} suffix="%" duration={1.5} />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Zero leakage verified across all experiments
                </p>
              </CardContent>
            </SpotlightCard>
          </Card3D>
        </RevealOnScroll>

        <RevealOnScroll delay={0.1}>
          <Card3D>
            <SpotlightCard className="h-full border-orange-500/20 bg-gradient-to-br from-orange-500/10 to-transparent" spotlightColor="rgba(249, 115, 22, 0.15)">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <TrendingDown className="h-4 w-4 text-orange-500" />
                  Honest Baseline
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  <AnimatedCounter value={0.60} suffix=" AUC" duration={1.5} decimals={2} />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Level-1 (no cognitive scores) - realistic early detection
                </p>
              </CardContent>
            </SpotlightCard>
          </Card3D>
        </RevealOnScroll>

        <RevealOnScroll delay={0.2}>
          <Card3D>
            <SpotlightCard className="h-full border-emerald-500/30 bg-gradient-to-br from-emerald-500/15 to-transparent animate-pulse-glow" spotlightColor="rgba(16, 185, 129, 0.2)">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Zap className="h-4 w-4 text-emerald-500" />
                  🔬 Longitudinal
                  <Badge className="ml-1 text-[10px] bg-emerald-600 animate-pulse">NEW</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  <TextGradient colors="from-emerald-400 to-cyan-400">
                    <AnimatedCounter value={0.83} suffix=" AUC" duration={1.5} decimals={2} />
                  </TextGradient>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  +9.5% with temporal biomarker change
                </p>
              </CardContent>
            </SpotlightCard>
          </Card3D>
        </RevealOnScroll>

        <RevealOnScroll delay={0.3}>
          <Card3D>
            <SpotlightCard className="h-full border-purple-500/20 bg-gradient-to-br from-purple-500/10 to-transparent" spotlightColor="rgba(168, 85, 247, 0.15)">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-purple-500" />
                  Best Predictor
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  <TextGradient colors="from-purple-400 via-pink-400 to-purple-400">
                    Hippocampus
                  </TextGradient>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  0.725 AUC alone - beats cognitive tests
                </p>
              </CardContent>
            </SpotlightCard>
          </Card3D>
        </RevealOnScroll>
      </section>

      {/* Main Feature Bento Grid */}
      <RevealOnScroll delay={0.1}>
        <section>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold tracking-tight">
              <TextGradient>Research Portal</TextGradient>
            </h2>
            <Link href="/dataset" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              View All &rarr;
            </Link>
          </div>
          <FeatureGrid />
        </section>
      </RevealOnScroll>

      {/* Cross-Dataset Highlight Banner (Animated) */}
      <RevealOnScroll delay={0.1}>
        <RobustnessBanner />
      </RevealOnScroll>

      <RevealOnScroll delay={0.1}>
        <section className="space-y-4">
          <h2 className="text-lg font-semibold tracking-tight">Quick Statistics</h2>
          <QuickStats />
        </section>
      </RevealOnScroll>

      <RevealOnScroll delay={0.1}>
        <ResultsDiagram />
      </RevealOnScroll>

      {/* Key Findings Section - With Enhanced Cards */}
      <RevealOnScroll delay={0.1}>
        <section className="space-y-4">
          <h2 className="text-lg font-semibold tracking-tight flex items-center gap-2">
            Key Research Findings
            <Sparkles className="h-4 w-4 text-yellow-500" />
          </h2>
          <div className="grid gap-4 md:grid-cols-3">
            <Card3D>
              <SpotlightCard className="h-full" spotlightColor="rgba(59, 130, 246, 0.1)">
                <CardHeader>
                  <CardTitle className="text-base">Fusion Performance Analysis</CardTitle>
                  <CardDescription>Why fusion models underperform</CardDescription>
                </CardHeader>
                <CardContent className="text-sm space-y-2">
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs shrink-0">Issue</Badge>
                    <p className="text-muted-foreground">
                      512 strong MRI features + 2 weak features (Age, Sex) = dimension imbalance
                    </p>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs shrink-0">Impact</Badge>
                    <p className="text-muted-foreground">
                      Clinical encoder creates 30 dims of noise, diluting MRI signal
                    </p>
                  </div>
                  <div className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs shrink-0">Solution</Badge>
                    <p className="text-muted-foreground">
                      Level-1.5: Add CSF biomarkers (ABETA, TAU, PTAU) + APOE4
                    </p>
                  </div>
                </CardContent>
              </SpotlightCard>
            </Card3D>

            <Card3D>
              <SpotlightCard className="h-full" spotlightColor="rgba(34, 197, 94, 0.1)">
                <CardHeader>
                  <CardTitle className="text-base">Data Cleaning Rigor</CardTitle>
                  <CardDescription>7 major cleaning steps applied</CardDescription>
                </CardHeader>
                <CardContent className="text-sm space-y-2">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                    <p className="text-muted-foreground">Subject-level de-duplication (1,825 to 629)</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                    <p className="text-muted-foreground">Baseline-only selection (no temporal leakage)</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                    <p className="text-muted-foreground">Subject-wise splits (zero overlap verified)</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                    <p className="text-muted-foreground">MMSE/CDR-SB excluded (no circular features)</p>
                  </div>
                </CardContent>
              </SpotlightCard>
            </Card3D>

            <Card3D>
              <SpotlightCard className="h-full border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 to-transparent" spotlightColor="rgba(16, 185, 129, 0.15)">
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
              </SpotlightCard>
            </Card3D>
          </div>
        </section>
      </RevealOnScroll>

      {/* Warning Section */}
      <RevealOnScroll delay={0.1}>
        <Alert className="border-orange-500/20 bg-gradient-to-r from-orange-500/10 to-transparent">
          <AlertTriangle className="h-4 w-4 text-orange-500" />
          <div className="ml-3 text-sm">
            <strong className="font-medium text-orange-700 dark:text-orange-400">Label Shift Warning: </strong>
            <span className="text-muted-foreground">
              OASIS targets very mild dementia (CDR 0.5), while ADNI includes a broader spectrum (MCI/AD).
              This definition shift contributes to the observed accuracy drops during transfer.
            </span>
          </div>
        </Alert>
      </RevealOnScroll>

      <div className="text-center text-xs text-muted-foreground">
        NeuroScope Research Portal · Validated on OASIS-1 & ADNI-1 · Not for Clinical Use
      </div>
    </div>
  )
}
