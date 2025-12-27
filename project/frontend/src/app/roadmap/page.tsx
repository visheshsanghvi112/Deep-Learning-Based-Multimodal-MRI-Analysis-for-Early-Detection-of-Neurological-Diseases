"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert } from "@/components/ui/alert"
import Link from "next/link"
import { ArrowRight, ArrowDown, CheckCircle2, XCircle, AlertTriangle, Lightbulb, Brain, Database, Zap, Target } from "lucide-react"

// Timeline step component
function TimelineStep({
  step,
  title,
  description,
  status,
  result,
  icon: Icon,
  color,
  isLast = false,
}: {
  step: number
  title: string
  description: string
  status: "success" | "warning" | "error" | "info"
  result: string
  icon: React.ElementType
  color: string
  isLast?: boolean
}) {
  const statusColors = {
    success: "bg-emerald-500",
    warning: "bg-amber-500",
    error: "bg-red-500",
    info: "bg-blue-500",
  }

  const StatusIcon = {
    success: CheckCircle2,
    warning: AlertTriangle,
    error: XCircle,
    info: Lightbulb,
  }[status]

  return (
    <div className="relative">
      {/* Connector line */}
      {!isLast && (
        <div className="absolute left-6 top-16 w-0.5 h-full bg-gradient-to-b from-border to-transparent" />
      )}

      <div className="flex gap-4">
        {/* Step number circle */}
        <div className={`relative z-10 flex-shrink-0 w-12 h-12 rounded-full ${color} flex items-center justify-center text-white font-bold shadow-lg`}>
          {step}
        </div>

        {/* Content */}
        <Card className="flex-1 mb-6">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Icon className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">{title}</CardTitle>
              <StatusIcon className={`h-4 w-4 ${status === 'success' ? 'text-emerald-500' : status === 'warning' ? 'text-amber-500' : status === 'error' ? 'text-red-500' : 'text-blue-500'}`} />
            </div>
            <CardDescription>{description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${status === 'success' ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400' :
              status === 'warning' ? 'bg-amber-500/10 text-amber-700 dark:text-amber-400' :
                status === 'error' ? 'bg-red-500/10 text-red-700 dark:text-red-400' :
                  'bg-blue-500/10 text-blue-700 dark:text-blue-400'
              }`}>
              {result}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default function RoadmapPage() {
  return (
    <div className="flex w-full flex-col gap-8 px-2 sm:px-0">
      {/* Hero Header */}
      <section className="text-center space-y-4 py-6">
        <Badge className="bg-gradient-to-r from-blue-600 to-purple-600">Research Journey</Badge>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
          How We Built This Project
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          A step-by-step visual guide to our research journey — from raw MRI data
          to breakthrough findings. Follow along to understand exactly what we did and discovered.
        </p>
      </section>

      {/* Quick Summary Banner */}
      <Card className="bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-emerald-500/10 border-0">
        <CardContent className="py-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold">2</div>
              <div className="text-xs text-muted-foreground">Datasets</div>
            </div>
            <div>
              <div className="text-2xl font-bold">1,065</div>
              <div className="text-xs text-muted-foreground">Subjects</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-emerald-600">0.83</div>
              <div className="text-xs text-muted-foreground">Best AUC</div>
            </div>
            <div>
              <div className="text-2xl font-bold">6</div>
              <div className="text-xs text-muted-foreground">Key Findings</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Timeline */}
      <section className="space-y-2">
        <h2 className="text-xl font-semibold">The Research Timeline</h2>
        <p className="text-sm text-muted-foreground mb-6">
          Each step builds on the previous. Scroll down to follow our complete journey.
        </p>

        <div className="space-y-2">
          <TimelineStep
            step={1}
            title="Collected Two Datasets"
            description="Downloaded OASIS-1 (436 subjects, single-site) and ADNI-1 (629 subjects, multi-site) brain MRI datasets with clinical metadata."
            status="success"
            result="1,065 total subjects"
            icon={Database}
            color="bg-blue-500"
          />

          <TimelineStep
            step={2}
            title="Cleaned the Data Rigorously"
            description="Applied 7 cleaning steps: de-duplication, baseline-only selection, subject-wise splits, excluded circular features (MMSE, CDR-SB)."
            status="success"
            result="Zero data leakage"
            icon={CheckCircle2}
            color="bg-blue-600"
          />

          <TimelineStep
            step={3}
            title="Extracted Deep Features"
            description="Used ResNet18 (pretrained on ImageNet) to extract 512-dimensional features from brain MRI scans using 2.5D multi-slice approach."
            status="success"
            result="512 features per subject"
            icon={Brain}
            color="bg-purple-500"
          />

          <TimelineStep
            step={4}
            title="Trained Classification Models"
            description="Trained MRI-only, Late Fusion, and Attention Fusion models on OASIS (CDR 0 vs 0.5+) and ADNI (CN vs MCI/AD)."
            status="warning"
            result="OASIS: 0.78 AUC, ADNI: 0.60 AUC"
            icon={Target}
            color="bg-purple-600"
          />

          <TimelineStep
            step={5}
            title="Discovered the Circularity Problem"
            description="Found that cognitive scores (MMSE, CDR-SB) ARE the diagnosis — using them is circular! Level-2 achieves 0.99 AUC (fake), Level-1 (honest) gets 0.60 AUC."
            status="error"
            result="⚠️ +39% gap is fake performance"
            icon={AlertTriangle}
            color="bg-red-500"
          />

          <TimelineStep
            step={6}
            title="Tested Cross-Dataset Transfer"
            description="Trained on one dataset, tested on the other. All models showed 15-25% AUC drop. MRI-only was most robust."
            status="warning"
            result="MRI-only best for transfer"
            icon={Zap}
            color="bg-amber-500"
          />

          <TimelineStep
            step={7}
            title="Ran Longitudinal Experiment"
            description="Used 2,262 scans from 639 subjects (avg 3.6 scans/person) to predict progression. ResNet features gave near-chance results."
            status="error"
            result="❌ ResNet: 0.52 AUC (near chance)"
            icon={Brain}
            color="bg-red-500"
          />

          <TimelineStep
            step={8}
            title="Investigated Why It Failed"
            description="Deep analysis revealed: 136 Dementia patients mislabeled as 'Stable', ResNet is scale-invariant (can't detect atrophy), wrong features for the task."
            status="info"
            result="🔍 Found 3 critical issues"
            icon={Lightbulb}
            color="bg-amber-500"
          />

          <TimelineStep
            step={9}
            title="Switched to Actual Biomarkers"
            description="Used hippocampus volume, ventricular size, entorhinal thickness from ADNIMERGE instead of CNN features. Focused on MCI cohort only."
            status="success"
            result="✅ Biomarkers: 0.74 → 0.83 AUC"
            icon={Brain}
            color="bg-emerald-500"
          />

          <TimelineStep
            step={10}
            title="Breakthrough: Longitudinal WORKS!"
            description="Adding temporal change (atrophy rate) improved AUC by +9.5%. Hippocampus alone achieves 0.725 AUC. APOE4 carriers have 2x conversion risk."
            status="success"
            result="🏆 0.83 AUC with biomarkers!"
            icon={CheckCircle2}
            color="bg-emerald-600"
            isLast
          />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Key Takeaways</h2>
        <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <Card className="border-emerald-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-emerald-600">✅ What Worked</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-1">
              <p>• Hippocampus volume (0.725 AUC alone)</p>
              <p>• Longitudinal atrophy rate (+9.5%)</p>
              <p>• Simple logistic regression</p>
              <p>• Proper biomarker selection</p>
            </CardContent>
          </Card>

          <Card className="border-red-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-red-600">❌ What Didn't Work</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-1">
              <p>• ResNet features for progression</p>
              <p>• LSTM sequence models</p>
              <p>• Complex attention fusion</p>
              <p>• Cognitive scores (circular!)</p>
            </CardContent>
          </Card>

          <Card className="border-blue-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-blue-600">💡 Surprising Findings</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-1">
              <p>• APOE4 doubles conversion risk</p>
              <p>• Education doesn't predict progression</p>
              <p>• Simple models beat complex ones</p>
              <p>• Data quality {'>'} dataset size</p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Final Result */}
      <Card className="bg-gradient-to-r from-emerald-500/20 to-blue-500/20 border-emerald-500/30">
        <CardContent className="py-8 text-center space-y-4">
          <div className="text-4xl font-bold text-emerald-600">0.83 AUC</div>
          <p className="text-lg font-medium">
            For predicting MCI → Dementia conversion using longitudinal biomarkers
          </p>
          <p className="text-sm text-muted-foreground max-w-lg mx-auto">
            Hippocampal atrophy rate, combined with baseline volume and APOE4 status,
            achieves state-of-the-art prediction — without using any circular features.
          </p>
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            <Badge>Hippocampus Volume</Badge>
            <Badge>Longitudinal Change</Badge>
            <Badge>APOE4 Genotype</Badge>
            <Badge>No Circularity</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex flex-wrap gap-4 justify-center">
        <Link href="/results" className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
          View Detailed Results <ArrowRight className="h-4 w-4" />
        </Link>
        <Link href="/interpretability" className="inline-flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-muted transition-colors">
          See All Visualizations <ArrowRight className="h-4 w-4" />
        </Link>
      </div>

      <Alert className="text-xs">
        This roadmap summarizes research conducted on OASIS-1 (436 subjects) and ADNI-1 (629 subjects)
        datasets. All findings are for research purposes only and not for clinical use.
      </Alert>
    </div>
  )
}
