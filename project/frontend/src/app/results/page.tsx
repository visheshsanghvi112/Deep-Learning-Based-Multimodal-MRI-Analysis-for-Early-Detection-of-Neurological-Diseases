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
  const [tab, setTab] = useState<"mri" | "clinical" | "combined">("mri")

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            Classification Results (OASIS-1)
          </h2>
          <Badge variant="outline">436 subjects • 205 labeled</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Binary classification of CDR=0 (Normal) vs CDR=0.5 (Very Mild Dementia).
          CNN features extracted for all 436 subjects using ResNet18.
        </p>
      </section>

      <Tabs value={tab} onValueChange={(v) => setTab(v as any)}>
        <TabsList>
          <TabsTrigger value="mri" className="min-w-[120px]">
            MRI Only
          </TabsTrigger>
          <TabsTrigger value="clinical" className="min-w-[120px]">
            Clinical
          </TabsTrigger>
          <TabsTrigger value="combined" className="min-w-[120px]">
            Combined
          </TabsTrigger>
        </TabsList>

        <TabsContent value="mri">
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

        <TabsContent value="clinical">
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Clinical AUC (with MMSE)</CardTitle>
                <CardDescription>
                  6 features: Age, MMSE, nWBV, eTIV, ASF, Educ
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-purple-600">0.87</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Best performance, but MMSE has data leakage concern.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Clinical AUC (no MMSE)</CardTitle>
                <CardDescription>Realistic early detection</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">0.74</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Without cognitive test dependency.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">MMSE Warning</CardTitle>
                <CardDescription>Data leakage concern</CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                MMSE alone achieves AUC 0.85, indicating high correlation
                with CDR. For realistic early detection, exclude MMSE and
                rely on imaging biomarkers.
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="combined">
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Combined AUC</CardTitle>
                <CardDescription>
                  MRI (512) + Clinical (6) = 518 features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold text-blue-600">0.82</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Multimodal fusion of imaging and clinical data.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Combined (no MMSE)</CardTitle>
                <CardDescription>Realistic scenario</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">0.78</div>
                <p className="mt-1 text-xs text-muted-foreground">
                  MRI + demographics for true early detection.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Interpretation</CardTitle>
                <CardDescription>Multimodal value</CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Multimodal approach validates that MRI and clinical features
                provide complementary information. Best for research;
                MRI-only preferred for early detection applications.
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      <Alert className="text-xs">
        All results computed using 5-fold stratified cross-validation on
        205 labeled subjects (135 CDR=0, 70 CDR=0.5). CNN features extracted
        for all 436 OASIS-1 subjects.
      </Alert>
    </div>
  )
}
