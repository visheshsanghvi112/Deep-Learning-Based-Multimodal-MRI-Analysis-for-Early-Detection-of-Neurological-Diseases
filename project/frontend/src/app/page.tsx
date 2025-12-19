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
import { MRISliceViewer } from "@/components/mri-slice-viewer"

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
        <div className="grid md:grid-cols-2 gap-4">
          <Card className="hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle className="text-sm">OASIS-1 Analysis</CardTitle>
              <CardDescription>Current Validation Cohort</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>• <span className="text-foreground font-medium">436</span> Total Subjects Processed</p>
              <p>• <span className="text-foreground font-medium">205</span> Labeled for Classification (CDR 0 vs 0.5)</p>
              <p>• 512-dim ResNet18 Features Extracted</p>
              <p>• 5-Fold Stratified Cross-Validation</p>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow border-l-4 border-l-blue-500">
            <CardHeader>
              <CardTitle className="text-sm flex items-center justify-between">
                <span>ADNI Expansion Data</span>
                <span className="text-[10px] bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">READY</span>
              </CardTitle>
              <CardDescription>Professional Deep Analysis (2025-12-13)</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>• <span className="text-foreground font-medium">230</span> NIfTI Scans Analyzed</p>
              <p>• <span className="text-foreground font-medium">203</span> Unique Subjects identified</p>
              <p>• <span className="text-foreground font-medium">3.0</span> Year Acquisition Span (2005-2008)</p>
              <p>• 100% NIfTI Format Compliance</p>
            </CardContent>
          </Card>
        </div>
      </OptimizedAnimatedSection>

      <OptimizedAnimatedSection delay={0.4}>
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-semibold mb-2">Interactive Analysis</h2>
            <MRISliceViewer />
          </div>
        </div>
      </OptimizedAnimatedSection>

      <ResultsDiagram />

      <Alert className="text-xs">
        This is a research visualization portal. All numbers are computed on
        the OASIS-1 dataset and should not be used for clinical decisions.
      </Alert>
    </div >
  )
}
