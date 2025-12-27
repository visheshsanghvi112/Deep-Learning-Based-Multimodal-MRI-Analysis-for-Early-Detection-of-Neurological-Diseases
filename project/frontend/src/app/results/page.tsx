"use client"

import { useState } from "react"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

export default function ResultsPage() {
  const [tab, setTab] = useState<"oasis" | "adni" | "crossdataset" | "longitudinal">("oasis")

  return (
    <div className="flex w-full flex-col gap-6 sm:gap-8 px-2 sm:px-0">
      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <h2 className="text-lg sm:text-xl font-semibold tracking-tight">
            Classification Results
          </h2>
          <Badge variant="outline" className="text-xs">OASIS-1 + ADNI-1</Badge>
          <Badge className="bg-emerald-600 text-xs">NEW: Longitudinal</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Binary classification results across datasets. OASIS: CDR 0 vs 0.5. ADNI: CN vs MCI+AD.
          Cross-dataset robustness validated. NEW: Longitudinal progression analysis.
        </p>
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as any)}>
        <TabsList className="flex flex-wrap h-auto gap-1 bg-transparent p-0 sm:bg-muted sm:p-1 sm:h-10">
          <TabsTrigger value="oasis" className="text-xs sm:text-sm data-[state=active]:bg-background">
            OASIS-1
          </TabsTrigger>
          <TabsTrigger value="adni" className="text-xs sm:text-sm data-[state=active]:bg-background">
            ADNI-1
          </TabsTrigger>
          <TabsTrigger value="crossdataset" className="text-xs sm:text-sm data-[state=active]:bg-background">
            Cross-Dataset
          </TabsTrigger>
          <TabsTrigger value="longitudinal" className="text-xs sm:text-sm bg-emerald-500/10 data-[state=active]:bg-emerald-500/20">
            🔬 Longitudinal
          </TabsTrigger>
        </TabsList>

        <TabsContent value="oasis">
          <div className="mt-4 grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">MRI-Only AUC</CardTitle>
                <CardDescription>
                  ResNet18 CNN embeddings (512-dim)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-emerald-600">0.78</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Pure imaging biomarker without clinical data dependency.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Feature Dimension</CardTitle>
                <CardDescription>Deep features per subject</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">512</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  2.5D multi-slice approach (axial, coronal, sagittal).
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Key Insight</CardTitle>
                <CardDescription>Scientific interpretation</CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                MRI provides meaningful signal beyond brain volume alone
                (AUC 0.78 vs nWBV baseline of 0.75). ResNet18 captures
                dementia-related structural patterns.
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ADNI Results Tab */}
        <TabsContent value="adni">
          <div className="mt-4 grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">ADNI Level-1 MRI-Only</CardTitle>
                <CardDescription>
                  No cognitive scores (realistic)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-blue-600">0.583</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Honest baseline with Age+Sex only.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">ADNI Level-1 Late Fusion</CardTitle>
                <CardDescription>MRI + Demographics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-purple-600">0.598</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  +1.5% improvement with multimodal.
                </p>
              </CardContent>
            </Card>

            <Card className="border-orange-500/30">
              <CardHeader>
                <CardTitle className="text-sm text-orange-600">ADNI Level-2 (Circular)</CardTitle>
                <CardDescription>⚠️ Uses MMSE/CDR-SB</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-orange-600">0.988</div>
                <p className="mt-1 text-xs text-orange-600">
                  NOT for early detection claims.
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Cross-Dataset Robustness Tab */}
        <TabsContent value="crossdataset">
          <div className="mt-4 space-y-6">
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-2">
              <Card className="bg-gradient-to-br from-blue-500/5 to-transparent">
                <CardHeader>
                  <CardTitle className="text-sm">OASIS → ADNI Transfer</CardTitle>
                  <CardDescription>High-quality to heterogeneous</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">MRI-Only</span>
                    <span className="font-bold text-green-600">0.607</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Late Fusion</span>
                    <span className="font-bold">0.575</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Attention Fusion</span>
                    <span className="font-bold">0.557</span>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    MRI-Only was most robust. Clinical features hurt transfer.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-purple-500/5 to-transparent">
                <CardHeader>
                  <CardTitle className="text-sm">ADNI → OASIS Transfer</CardTitle>
                  <CardDescription>Heterogeneous to high-quality</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">MRI-Only</span>
                    <span className="font-bold">0.569</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Late Fusion</span>
                    <span className="font-bold text-purple-600">0.624</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Attention Fusion</span>
                    <span className="font-bold">0.548</span>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Late Fusion was most robust. Clinical features helped here.
                  </p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Key Insight</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Robustness is <strong className="text-foreground">direction-dependent</strong>.
                MRI features provide a stable baseline, while multimodal benefit depends on
                domain alignment. Attention Fusion was consistently unstable across transfers.
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* NEW: Longitudinal Experiment Tab */}
        <TabsContent value="longitudinal">
          <div className="mt-4 space-y-6">
            {/* Research Question */}
            <Card className="border-emerald-500/30 bg-gradient-to-br from-emerald-500/5 to-transparent">
              <CardHeader>
                <CardTitle className="text-sm flex items-center gap-2">
                  🔬 Research Question
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="italic text-muted-foreground text-sm">
                  Does observing CHANGE over time (multiple MRIs per subject) improve prediction of dementia progression?
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Badge>2,262 Scans</Badge>
                  <Badge>639 Subjects</Badge>
                  <Badge>Avg 3.6 scans/person</Badge>
                </div>
              </CardContent>
            </Card>

            {/* Three Phases */}
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {/* Phase 1 */}
              <Card className="border-red-500/20">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-red-600">Phase 1: ResNet Features</CardTitle>
                  <CardDescription>Initial Experiment</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Single-Scan</span>
                    <span className="font-mono">0.51 AUC</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Delta Model</span>
                    <span className="font-mono">0.52 AUC</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>LSTM Sequence</span>
                    <span className="font-mono text-red-600">0.44 AUC</span>
                  </div>
                  <p className="mt-2 text-xs text-red-600">
                    ❌ Near-chance performance. Prompted investigation.
                  </p>
                </CardContent>
              </Card>

              {/* Phase 2 */}
              <Card className="border-amber-500/20">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-amber-600">Phase 2: Investigation</CardTitle>
                  <CardDescription>Root Cause Analysis</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-xs">
                  <div className="flex items-start gap-2">
                    <span className="text-amber-600">⚠️</span>
                    <span>136 Dementia patients labeled &quot;Stable&quot;</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-amber-600">⚠️</span>
                    <span>ResNet features are scale-invariant</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-amber-600">⚠️</span>
                    <span>Cannot detect volume changes</span>
                  </div>
                  <p className="mt-2 text-amber-600">
                    🔍 Issues identified. Need biomarkers.
                  </p>
                </CardContent>
              </Card>

              {/* Phase 3 */}
              <Card className="border-emerald-500/30 bg-emerald-500/5">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-emerald-600">Phase 3: Biomarkers</CardTitle>
                  <CardDescription>Corrected Experiment</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Biomarkers Only</span>
                    <span className="font-mono">0.74 AUC</span>
                  </div>
                  <div className="flex justify-between text-sm font-bold">
                    <span>+ Longitudinal Δ</span>
                    <span className="font-mono text-emerald-600">0.83 AUC</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>+ ADAS13</span>
                    <span className="font-mono">0.84 AUC</span>
                  </div>
                  <p className="mt-2 text-xs text-emerald-600">
                    ✅ +31 points vs ResNet! Longitudinal WORKS!
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Key Discoveries */}
            <div className="grid gap-4 grid-cols-2 sm:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">🏆 Best Predictor</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-emerald-600">Hippocampus</div>
                  <p className="text-xs text-muted-foreground">0.725 AUC alone</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">🧬 APOE4 Effect</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-purple-600">2x Risk</div>
                  <p className="text-xs text-muted-foreground">44% vs 23% conversion</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">📈 Longitudinal Boost</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">+9.5%</div>
                  <p className="text-xs text-muted-foreground">0.74 → 0.83 AUC</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">💡 Key Insight</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-lg font-bold">Simple Wins</div>
                  <p className="text-xs text-muted-foreground">LR (0.83) {">"} LSTM (0.44)</p>
                </CardContent>
              </Card>
            </div>

            {/* Conclusion */}
            <Card className="bg-gradient-to-r from-emerald-500/10 to-blue-500/10">
              <CardHeader>
                <CardTitle className="text-sm">🎯 Final Conclusion</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-sm">
                  <strong className="text-emerald-600">Longitudinal MRI data DOES help</strong> (+9.5% AUC)
                  when using proper structural biomarkers (hippocampus, ventricles, entorhinal).
                </p>
                <p className="text-sm text-muted-foreground">
                  ResNet features are unsuitable for progression prediction due to scale-invariance.
                  Hippocampal atrophy RATE is the key predictor.
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      <Alert className="text-xs">
        Results validated across two independent datasets: OASIS-1 (single-site, 436 subjects)
        and ADNI-1 (multi-site, 629 subjects). Longitudinal experiment used 2,262 scans from 639 subjects.
        Cross-dataset and longitudinal experiments confirm representation robustness.
      </Alert>
    </div>
  )
}

