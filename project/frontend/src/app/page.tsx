"use client"

import { Alert } from "@/components/ui/alert"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ResearchStats } from "@/components/research-stats"
import { OptimizedAnimatedSection } from "@/components/performance-optimized"
import { FeatureGrid } from "@/components/feature-grid"
import { QuickStats } from "@/components/quick-stats"
import { Hero3D } from "@/components/hero-3d"
import { ResultsDiagram } from "@/components/results-diagram"

export default function Home() {
  return (
    <div className="flex w-full flex-col gap-8">
      <Hero3D />

      <ResearchStats />

      <OptimizedAnimatedSection delay={0.1}>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold mb-2">Research Features</h2>
            <p className="text-sm text-muted-foreground mb-4">
              Comprehensive overview of dataset, pipeline, results, and research capabilities
            </p>
          </div>
          <FeatureGrid />
        </div>
      </OptimizedAnimatedSection>

      <OptimizedAnimatedSection delay={0.2}>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold mb-2">Quick Statistics</h2>
            <QuickStats />
          </div>
            </div>
      </OptimizedAnimatedSection>

      <OptimizedAnimatedSection delay={0.3}>
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader>
            <CardTitle className="text-sm">Research Notes</CardTitle>
            <CardDescription>Key considerations</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>• CNN features extracted for all 436 OASIS-1 subjects</p>
            <p>• 205 subjects usable for CDR classification (135 normal, 70 very mild)</p>
            <p>• MMSE excluded for realistic early detection (data leakage concern)</p>
            <p>• Results are for research purposes only, not clinical diagnosis</p>
          </CardContent>
        </Card>
      </OptimizedAnimatedSection>

      <ResultsDiagram />

      <Alert className="text-xs">
        This is a research visualization portal. All numbers are computed on
        the OASIS-1 dataset and should not be used for clinical decisions.
      </Alert>
    </div>
  )
}
