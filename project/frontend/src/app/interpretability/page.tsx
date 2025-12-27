"use client"

import { useState } from "react"
import Image from "next/image"
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
import { Alert } from "@/components/ui/alert"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { X, ZoomIn } from "lucide-react"

// Visualization data organized by category
const visualizations = {
  oasis: [
    {
      id: "A1",
      src: "/figures/A1_oasis_model_comparison.png",
      title: "OASIS Model Comparison",
      description: "Comparison of different model architectures on OASIS-1 dataset showing MRI-only, Late Fusion, and Attention Fusion performance.",
      insight: "MRI-only achieves 0.78 AUC, with minimal gains from fusion approaches."
    },
    {
      id: "A2",
      src: "/figures/A2_oasis_class_distribution.png",
      title: "OASIS Class Distribution",
      description: "Distribution of CDR 0 (healthy) vs CDR 0.5+ (very mild dementia) in OASIS-1 dataset.",
      insight: "Well-balanced dataset with 206 CDR=0 and 230 CDR≥0.5 subjects."
    },
  ],
  adni: [
    {
      id: "B1",
      src: "/figures/B1_adni_level1_honest.png",
      title: "ADNI Level-1 Honest Baseline",
      description: "Performance on ADNI without cognitive scores (MMSE, CDR-SB) - the realistic early detection scenario.",
      insight: "Level-1 achieves 0.60 AUC - the honest baseline for early detection."
    },
    {
      id: "B2",
      src: "/figures/B2_level1_vs_level2_circularity.png",
      title: "Level-1 vs Level-2 Circularity",
      description: "Dramatic performance gap when including vs excluding cognitive scores reveals circularity problem.",
      insight: "Level-2 (0.99 AUC) uses cognitive scores that ARE the diagnosis - not early detection!"
    },
    {
      id: "B3",
      src: "/figures/B3_adni_class_distribution.png",
      title: "ADNI Class Distribution",
      description: "Distribution of CN (cognitively normal) vs MCI/AD in ADNI-1 dataset.",
      insight: "629 unique subjects after de-duplication: 195 CN, 298 MCI, 136 AD."
    },
  ],
  transfer: [
    {
      id: "C1",
      src: "/figures/C1_in_vs_cross_dataset_collapse.png",
      title: "In-Dataset vs Cross-Dataset Collapse",
      description: "Performance comparison showing how models perform in-dataset vs cross-dataset transfer.",
      insight: "All models show 15-25% AUC drop during cross-dataset transfer."
    },
    {
      id: "C2",
      src: "/figures/C2_transfer_robustness_heatmap.png",
      title: "Transfer Robustness Heatmap",
      description: "Heatmap showing transfer performance in both directions: OASIS→ADNI and ADNI→OASIS.",
      insight: "MRI-only is most robust for OASIS→ADNI; Late Fusion best for ADNI→OASIS."
    },
    {
      id: "C3",
      src: "/figures/C3_auc_drop_robustness.png",
      title: "AUC Drop Analysis",
      description: "Quantification of performance degradation during cross-dataset transfer.",
      insight: "Attention Fusion shows highest variance and poorest robustness."
    },
  ],
  preprocessing: [
    {
      id: "D1",
      src: "/figures/D1_preprocessing_pipeline.png",
      title: "Preprocessing Pipeline",
      description: "Complete data preprocessing workflow from raw MRI to extracted features.",
      insight: "7 major cleaning steps ensure zero data leakage and proper validation."
    },
    {
      id: "D2",
      src: "/figures/D2_sample_size_reduction.png",
      title: "Sample Size Reduction",
      description: "How subject-level de-duplication reduces ADNI from 1,825 scans to 629 unique subjects.",
      insight: "Baseline-only selection prevents temporal leakage in cross-sectional analysis."
    },
    {
      id: "D3",
      src: "/figures/D3_age_distribution.png",
      title: "Age Distribution",
      description: "Age distribution across datasets and diagnostic groups.",
      insight: "ADNI subjects generally older (avg 75) than OASIS (avg 70)."
    },
    {
      id: "D4",
      src: "/figures/D4_sex_distribution.png",
      title: "Sex Distribution",
      description: "Sex distribution across OASIS and ADNI datasets.",
      insight: "Both datasets show slight female predominance (~55%)."
    },
    {
      id: "D5",
      src: "/figures/D5_feature_dimensions.png",
      title: "Feature Dimensions",
      description: "Breakdown of feature dimensions: 512 MRI + 2 clinical vs ideal biomarker setup.",
      insight: "512 MRI features vs 2 clinical creates dimension imbalance in fusion."
    },
  ],
  embeddings: [
    {
      id: "E1",
      src: "/static/attention_weights.png",
      title: "Attention Weights",
      description: "Attention distribution across modalities in the fusion model.",
      insight: "Attention concentrates on age and global atrophy measures."
    },
    {
      id: "E2",
      src: "/static/embeddings_tsne.png",
      title: "t-SNE Embeddings",
      description: "t-SNE visualization of learned subject embeddings colored by diagnosis.",
      insight: "Smooth gradients with respect to age and CDR show meaningful representations."
    },
  ],
}

function VisualizationCard({
  viz,
}: {
  viz: { id: string; src: string; title: string; description: string; insight: string }
}) {
  const [imageError, setImageError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isOpen, setIsOpen] = useState(false)

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <Card className="overflow-hidden hover:shadow-lg transition-shadow">
        <DialogTrigger asChild>
          <div className="relative h-48 sm:h-56 w-full cursor-pointer group bg-muted/30">
            {isLoading && !imageError && (
              <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
                <div className="text-sm text-muted-foreground">Loading...</div>
              </div>
            )}
            {imageError ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-sm text-muted-foreground">Image unavailable</div>
              </div>
            ) : (
              <>
                <Image
                  src={viz.src}
                  alt={viz.title}
                  fill
                  style={{ objectFit: "contain" }}
                  className="p-2"
                  onError={() => setImageError(true)}
                  onLoad={() => setIsLoading(false)}
                  unoptimized
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                  <ZoomIn className="h-8 w-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </>
            )}
          </div>
        </DialogTrigger>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Badge variant="outline" className="text-xs">{viz.id}</Badge>
            {viz.title}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-xs text-muted-foreground line-clamp-2">{viz.description}</p>
          <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
            💡 {viz.insight}
          </p>
        </CardContent>
      </Card>

      <DialogContent className="max-w-4xl w-[95vw] max-h-[90vh] overflow-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Badge>{viz.id}</Badge>
            {viz.title}
          </DialogTitle>
          <DialogDescription>{viz.description}</DialogDescription>
        </DialogHeader>
        <div className="relative w-full h-[60vh] bg-muted/30 rounded-lg">
          {!imageError && (
            <Image
              src={viz.src}
              alt={viz.title}
              fill
              style={{ objectFit: "contain" }}
              className="p-4"
              unoptimized
            />
          )}
        </div>
        <div className="p-4 bg-emerald-500/10 rounded-lg">
          <p className="text-sm font-medium">🔬 Key Insight:</p>
          <p className="text-sm text-muted-foreground">{viz.insight}</p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default function InterpretabilityPage() {
  const [tab, setTab] = useState("oasis")

  return (
    <div className="flex w-full flex-col gap-6 px-2 sm:px-0">
      {/* Header */}
      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <h2 className="text-lg sm:text-xl font-semibold tracking-tight">
            Research Visualizations
          </h2>
          <Badge variant="outline" className="text-xs">13 Figures</Badge>
          <Badge className="bg-emerald-600 text-xs">Interactive</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Click any visualization to view full-size with detailed description.
          All figures support the research findings documented in the paper.
        </p>
      </section>

      {/* Tabs */}
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="flex flex-wrap h-auto gap-1 bg-transparent p-0 sm:bg-muted sm:p-1 sm:h-10">
          <TabsTrigger value="oasis" className="text-xs sm:text-sm data-[state=active]:bg-background">
            OASIS ({visualizations.oasis.length})
          </TabsTrigger>
          <TabsTrigger value="adni" className="text-xs sm:text-sm data-[state=active]:bg-background">
            ADNI ({visualizations.adni.length})
          </TabsTrigger>
          <TabsTrigger value="transfer" className="text-xs sm:text-sm data-[state=active]:bg-background">
            Transfer ({visualizations.transfer.length})
          </TabsTrigger>
          <TabsTrigger value="preprocessing" className="text-xs sm:text-sm data-[state=active]:bg-background">
            Data ({visualizations.preprocessing.length})
          </TabsTrigger>
          <TabsTrigger value="embeddings" className="text-xs sm:text-sm data-[state=active]:bg-background">
            Embeddings ({visualizations.embeddings.length})
          </TabsTrigger>
        </TabsList>

        {/* OASIS Tab */}
        <TabsContent value="oasis" className="mt-4">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-2">
            {visualizations.oasis.map((viz) => (
              <VisualizationCard key={viz.id} viz={viz} />
            ))}
          </div>
        </TabsContent>

        {/* ADNI Tab */}
        <TabsContent value="adni" className="mt-4">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {visualizations.adni.map((viz) => (
              <VisualizationCard key={viz.id} viz={viz} />
            ))}
          </div>
        </TabsContent>

        {/* Transfer Tab */}
        <TabsContent value="transfer" className="mt-4">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {visualizations.transfer.map((viz) => (
              <VisualizationCard key={viz.id} viz={viz} />
            ))}
          </div>
        </TabsContent>

        {/* Preprocessing Tab */}
        <TabsContent value="preprocessing" className="mt-4">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {visualizations.preprocessing.map((viz) => (
              <VisualizationCard key={viz.id} viz={viz} />
            ))}
          </div>
        </TabsContent>

        {/* Embeddings Tab */}
        <TabsContent value="embeddings" className="mt-4">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2">
            {visualizations.embeddings.map((viz) => (
              <VisualizationCard key={viz.id} viz={viz} />
            ))}
          </div>
        </TabsContent>
      </Tabs>

      {/* Summary Cards */}
      <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">📊 Total Figures</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">13</div>
            <p className="text-xs text-muted-foreground">Publication-ready</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">🔬 Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2</div>
            <p className="text-xs text-muted-foreground">OASIS-1 + ADNI-1</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">📈 Key Finding</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold text-emerald-600">0.83 AUC</div>
            <p className="text-xs text-muted-foreground">Longitudinal biomarkers</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">⚠️ Circularity Gap</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold text-orange-600">+39%</div>
            <p className="text-xs text-muted-foreground">Level-1 vs Level-2</p>
          </CardContent>
        </Card>
      </section>

      {/* Alert */}
      <Alert className="text-xs">
        All visualizations are generated from actual research data on OASIS-1 (436 subjects)
        and ADNI-1 (629 subjects). Click any figure to view full-size with interpretation.
      </Alert>
    </div>
  )
}
