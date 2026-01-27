"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert } from "@/components/ui/alert"
import Link from "next/link"
import { ArrowRight, CheckCircle2, XCircle, AlertTriangle, Lightbulb, Brain, Database, Zap, Target, LucideIcon, Sparkles, Trophy } from "lucide-react"
import { motion } from "framer-motion"
import {
  SpotlightCard,
  TextGradient,
  AnimatedCounter,
  RevealOnScroll,
  MagneticButton,
  Card3D
} from "@/components/ui/aceternity-effects"

// Timeline step component with enhanced animations
function TimelineStep({
  step,
  title,
  description,
  status,
  result,
  icon: Icon,
  color,
  isLast = false,
  delay = 0,
}: {
  step: number
  title: string
  description: string
  status: "success" | "warning" | "error" | "info"
  result: string
  icon: LucideIcon
  color: string
  isLast?: boolean
  delay?: number
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

  const spotlightColors = {
    success: "rgba(16, 185, 129, 0.15)",
    warning: "rgba(245, 158, 11, 0.15)",
    error: "rgba(239, 68, 68, 0.15)",
    info: "rgba(59, 130, 246, 0.15)",
  }

  return (
    <RevealOnScroll delay={delay} direction="left">
      <div className="relative">
        {/* Connector line */}
        {!isLast && (
          <motion.div
            className="absolute left-6 top-16 w-0.5 h-full"
            initial={{ scaleY: 0 }}
            animate={{ scaleY: 1 }}
            transition={{ duration: 0.5, delay: delay + 0.3 }}
            style={{
              background: "linear-gradient(to bottom, var(--border), transparent)",
              transformOrigin: "top"
            }}
          />
        )}

        <div className="flex gap-4">
          {/* Step number circle with pulse effect */}
          <motion.div
            className={`relative z-10 flex-shrink-0 w-12 h-12 rounded-full ${color} flex items-center justify-center text-white font-bold shadow-lg`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 20, delay }}
          >
            {step}
            {status === "success" && (
              <motion.div
                className="absolute inset-0 rounded-full bg-emerald-500"
                animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            )}
          </motion.div>

          {/* Content */}
          <SpotlightCard className="flex-1 mb-6" spotlightColor={spotlightColors[status]}>
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
          </SpotlightCard>
        </div>
      </div>
    </RevealOnScroll>
  )
}

export default function RoadmapPage() {
  return (
    <div className="flex w-full flex-col gap-8 px-2 sm:px-0">
      {/* Hero Header - Enhanced with gradient and animations */}
      <RevealOnScroll>
        <section className="text-center space-y-4 py-6">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
          >
            <Badge className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white px-4 py-1">
              <Sparkles className="h-3 w-3 mr-1 inline" />
              Research Journey
            </Badge>
          </motion.div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            <TextGradient colors="from-blue-500 via-purple-500 to-pink-500">
              How We Built This Project
            </TextGradient>
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            A step-by-step visual guide to our research journey — from raw MRI data
            to breakthrough findings. Follow along to understand exactly what we did and discovered.
          </p>
        </section>
      </RevealOnScroll>

      {/* Quick Summary Banner - With animated counters */}
      <RevealOnScroll delay={0.1}>
        <SpotlightCard className="bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-emerald-500/10 border-0" spotlightColor="rgba(139, 92, 246, 0.1)">
          <CardContent className="py-6">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold">
                  <AnimatedCounter value={2} duration={1} />
                </div>
                <div className="text-xs text-muted-foreground">Datasets</div>
              </div>
              <div>
                <div className="text-2xl font-bold">
                  <AnimatedCounter value={1065} suffix="" duration={1.5} />
                </div>
                <div className="text-xs text-muted-foreground">Subjects</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-emerald-600">
                  <AnimatedCounter value={0.848} decimals={3} duration={1.5} />
                </div>
                <div className="text-xs text-muted-foreground">Best AUC</div>
              </div>
              <div>
                <div className="text-2xl font-bold">
                  <AnimatedCounter value={6} duration={1} />
                </div>
                <div className="text-xs text-muted-foreground">Key Findings</div>
              </div>
            </div>
          </CardContent>
        </SpotlightCard>
      </RevealOnScroll>

      {/* Main Timeline */}
      <section className="space-y-2">
        <RevealOnScroll>
          <h2 className="text-xl font-semibold">The Research Timeline</h2>
          <p className="text-sm text-muted-foreground mb-6">
            Each step builds on the previous. Scroll down to follow our complete journey.
          </p>
        </RevealOnScroll>

        <div className="space-y-2">
          <TimelineStep
            step={1}
            title="Collected Two Datasets"
            description="Downloaded OASIS-1 (436 subjects, single-site) and ADNI-1 (629 subjects, multi-site) brain MRI datasets with clinical metadata."
            status="success"
            result="1,065 total subjects"
            icon={Database}
            color="bg-blue-500"
            delay={0}
          />

          <TimelineStep
            step={2}
            title="Cleaned the Data Rigorously"
            description="Applied 7 cleaning steps: de-duplication, baseline-only selection, subject-wise splits, excluded circular features (MMSE, CDR-SB)."
            status="success"
            result="Zero data leakage"
            icon={CheckCircle2}
            color="bg-blue-600"
            delay={0.05}
          />

          <TimelineStep
            step={3}
            title="Extracted Deep Features"
            description="Used ResNet18 (pretrained on ImageNet) to extract 512-dimensional features from brain MRI scans using 2.5D multi-slice approach."
            status="success"
            result="512 features per subject"
            icon={Brain}
            color="bg-purple-500"
            delay={0.1}
          />

          <TimelineStep
            step={4}
            title="Trained Classification Models"
            description="Trained MRI-only, Late Fusion, and Attention Fusion models on OASIS (CDR 0 vs 0.5+) and ADNI (CN vs MCI/AD)."
            status="warning"
            result="OASIS: 0.78 AUC, ADNI: 0.60 AUC"
            icon={Target}
            color="bg-purple-600"
            delay={0.15}
          />

          <TimelineStep
            step={5}
            title="Discovered the Circularity Problem"
            description="Found that cognitive scores (MMSE, CDR-SB) ARE the diagnosis — using them is circular! Level-2 achieves 0.99 AUC (fake), Level-1 (honest) gets 0.60 AUC."
            status="error"
            result="⚠️ +39% gap is fake performance"
            icon={AlertTriangle}
            color="bg-red-500"
            delay={0.2}
          />

          <TimelineStep
            step={6}
            title="Tested Cross-Dataset Transfer"
            description="Trained on one dataset, tested on the other. All models showed 15-25% AUC drop. MRI-only was most robust."
            status="warning"
            result="MRI-only best for transfer"
            icon={Zap}
            color="bg-amber-500"
            delay={0.25}
          />

          <TimelineStep
            step={7}
            title="🎯 Level-MAX: Biomarker Breakthrough"
            description="Enriched clinical features from 2D (Age/Sex) to 14D biological profile (CSF, APOE4, Hippocampus, 6 other volumetrics). Proved fusion works with quality features!"
            status="success"
            result="✅ 0.808 AUC (+16.5% over MRI-only)"
            icon={Trophy}
            color="bg-emerald-500"
            delay={0.28}
          />

          <TimelineStep
            step={8}
            title="Ran Longitudinal Experiment"
            description="Used 2,262 scans from 639 subjects (avg 3.6 scans/person) to predict progression. ResNet features gave near-chance results."
            status="error"
            result="❌ ResNet: 0.52 AUC (near chance)"
            icon={Brain}
            color="bg-red-500"
            delay={0.32}
          />

          <TimelineStep
            step={9}
            title="Investigated Why It Failed"
            description="Deep analysis revealed: 136 Dementia patients mislabeled as 'Stable', ResNet is scale-invariant (can't detect atrophy), wrong features for the task."
            status="info"
            result="🔍 Found 3 critical issues"
            icon={Lightbulb}
            color="bg-amber-500"
            delay={0.37}
          />

          <TimelineStep
            step={10}
            title="Switched to Actual Biomarkers"
            description="Used hippocampus volume, ventricular size, entorhinal thickness from ADNIMERGE instead of CNN features. Focused on MCI cohort only."
            status="success"
            result="✅ Biomarkers: 0.74 → 0.848 AUC"
            icon={Brain}
            color="bg-emerald-500"
            delay={0.42}
          />

          <TimelineStep
            step={11}
            title="Breakthrough: Longitudinal WORKS!"
            description="Adding temporal change (atrophy rate) improved AUC by +11.2%. Hippocampus alone achieves 0.725 AUC. APOE4 carriers have 2x conversion risk. Random Forest outperformed all other models."
            status="success"
            result="🏆 0.848 AUC with Random Forest!"
            icon={CheckCircle2}
            color="bg-emerald-600"
            isLast
            delay={0.47}
          />
        </div>
      </section>

      {/* Key Takeaways - With 3D Cards */}
      <RevealOnScroll delay={0.1}>
        <section className="space-y-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            Key Takeaways
            <Sparkles className="h-4 w-4 text-yellow-500" />
          </h2>
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            <Card3D>
              <SpotlightCard className="h-full border-emerald-500/30" spotlightColor="rgba(16, 185, 129, 0.15)">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-emerald-600">✅ What Worked</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-1">
                  <p>• Level-MAX biomarker fusion (0.808 AUC)</p>
                  <p>• Hippocampus volume (0.725 AUC alone)</p>
                  <p>• Longitudinal atrophy rate (+11.2%)</p>
                  <p>• Random Forest classifier (best model)</p>
                  <p>• Proper biomarker selection</p>
                </CardContent>
              </SpotlightCard>
            </Card3D>

            <Card3D>
              <SpotlightCard className="h-full border-red-500/30" spotlightColor="rgba(239, 68, 68, 0.15)">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-red-600">❌ What Didn't Work</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-1">
                  <p>• ResNet features for progression</p>
                  <p>• LSTM sequence models</p>
                  <p>• Complex attention fusion</p>
                  <p>• Cognitive scores (circular!)</p>
                </CardContent>
              </SpotlightCard>
            </Card3D>

            <Card3D>
              <SpotlightCard className="h-full border-blue-500/30" spotlightColor="rgba(59, 130, 246, 0.15)">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-blue-600">💡 Surprising Findings</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground space-y-1">
                  <p>• APOE4 doubles conversion risk</p>
                  <p>• Education doesn't predict progression</p>
                  <p>• Simple models beat complex ones</p>
                  <p>• Data quality {'>'} dataset size</p>
                </CardContent>
              </SpotlightCard>
            </Card3D>
          </div>
        </section>
      </RevealOnScroll>

      {/* Final Result - Hero Card */}
      <RevealOnScroll delay={0.1}>
        <SpotlightCard className="bg-gradient-to-r from-emerald-500/20 via-blue-500/10 to-purple-500/20 border-emerald-500/30" spotlightColor="rgba(16, 185, 129, 0.2)">
          <CardContent className="py-8 text-center space-y-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.2 }}
              className="inline-block"
            >
              <Trophy className="h-12 w-12 text-yellow-500 mx-auto mb-2" />
            </motion.div>
            <div className="text-4xl font-bold">
              <TextGradient colors="from-emerald-400 via-cyan-400 to-blue-400">
                <AnimatedCounter value={0.848} suffix=" AUC" decimals={3} duration={2} />
              </TextGradient>
            </div>
            <p className="text-lg font-medium">
              For predicting MCI → Dementia conversion using longitudinal biomarkers
            </p>
            <p className="text-sm text-muted-foreground max-w-lg mx-auto">
              Hippocampal atrophy rate, combined with baseline volume and APOE4 status,
              achieves state-of-the-art prediction — without using any circular features.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-4">
              <Badge className="bg-emerald-600">Hippocampus Volume</Badge>
              <Badge className="bg-blue-600">Longitudinal Change</Badge>
              <Badge className="bg-purple-600">APOE4 Genotype</Badge>
              <Badge variant="outline">No Circularity</Badge>
            </div>
          </CardContent>
        </SpotlightCard>
      </RevealOnScroll>

      {/* Navigation - With Magnetic Buttons */}
      <RevealOnScroll delay={0.1}>
        <div className="flex flex-wrap gap-4 justify-center">
          <MagneticButton>
            <Link href="/results" className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors group">
              View Detailed Results <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </MagneticButton>
          <MagneticButton>
            <Link href="/interpretability" className="inline-flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-muted transition-colors group">
              See All Visualizations <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </MagneticButton>
        </div>
      </RevealOnScroll>

      <Alert className="text-xs">
        This roadmap summarizes research conducted on OASIS-1 (436 subjects) and ADNI-1 (629 subjects)
        datasets. All findings are for research purposes only and not for clinical use.
      </Alert>
    </div>
  )
}
