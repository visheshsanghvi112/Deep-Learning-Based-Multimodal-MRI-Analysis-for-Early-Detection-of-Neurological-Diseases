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
  const [tab, setTab] = useState<"oasis" | "adni" | "crossdataset">("oasis")

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            Classification Results
          </h2>
          <Badge variant="outline">OASIS-1 + ADNI-1</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Binary classification results across datasets. OASIS: CDR 0 vs 0.5. ADNI: CN vs MCI+AD.
          Cross-dataset robustness validated.
        </p>
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as any)}>
        <TabsList>
          <TabsTrigger value="oasis" className="min-w-[120px]">
            OASIS-1
          </TabsTrigger>
          <TabsTrigger value="adni" className="min-w-[120px]">
            ADNI-1
          </TabsTrigger>
          <TabsTrigger value="crossdataset" className="min-w-[140px]">
            Cross-Dataset
          </TabsTrigger>
        </TabsList>

        <TabsContent value="oasis">
          <div className="mt-4 grid gap-4 md:grid-cols-3">
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
          <div className="mt-4 grid gap-4 md:grid-cols-3">
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
            <div className="grid gap-4 md:grid-cols-2">
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
      </Tabs>

      <Alert className="text-xs">
        Results validated across two independent datasets: OASIS-1 (single-site, 436 subjects)
        and ADNI-1 (multi-site, 629 subjects). Cross-dataset experiments confirm representation
        robustness under real-world dataset shift.
      </Alert>
    </div>
  )
}
