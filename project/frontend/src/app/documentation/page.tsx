"use client"

import Image from "next/image"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert } from "@/components/ui/alert"
import {
    FileText,
    AlertTriangle,
    TrendingUp,
    CheckCircle2,
    XCircle,
    Zap,
    Database,
    Code,
    BarChart3,
    Shield,
    Lock,
    ExternalLink,
    Brain,
    Microscope,
    Award,
    Globe,
    Linkedin,
    Github
} from "lucide-react"
import Link from "next/link"

export default function DocumentationPage() {
    return (
        <div className="flex w-full flex-col gap-8">
            {/* Header */}
            <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tight">Research Documentation</h1>
                <p className="text-muted-foreground">
                    Comprehensive documentation of data cleaning, honest assessment, and publication strategy
                </p>
            </div>

            {/* Dataset Access Banner - Prestigious & Informative */}
            <Card className="border-2 border-blue-500/30 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 overflow-hidden">
                <CardHeader className="pb-4">
                    <div className="flex flex-col sm:flex-row items-start gap-4">
                        <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 shrink-0">
                            <Database className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div className="flex-1 space-y-1">
                            <CardTitle className="text-lg sm:text-xl flex flex-wrap items-center gap-2">
                                <span>Research-Grade Neuroimaging Datasets</span>
                                <Badge className="bg-gradient-to-r from-blue-600 to-purple-600 text-white text-[10px]">
                                    Prestigious Access
                                </Badge>
                            </CardTitle>
                            <CardDescription>
                                This research utilizes two of the world's most respected neuroimaging repositories
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="space-y-6">
                    {/* Dataset Cards */}
                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="rounded-xl border border-green-500/30 bg-gradient-to-br from-green-500/10 to-transparent p-4 space-y-3">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                                <div className="flex items-center gap-2">
                                    <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center shrink-0">
                                        <Globe className="h-5 w-5 text-green-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-green-700 dark:text-green-400">OASIS-1</h3>
                                        <p className="text-xs text-muted-foreground">Open Access Series of Imaging Studies</p>
                                    </div>
                                </div>
                                <Badge variant="outline" className="text-green-600 border-green-500/50 self-start sm:self-auto">
                                    Open Access
                                </Badge>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                                <div className="rounded-lg bg-green-500/10 p-2">
                                    <div className="text-xs text-muted-foreground">Subjects</div>
                                    <div className="font-bold text-green-700 dark:text-green-400">436</div>
                                </div>
                                <div className="rounded-lg bg-green-500/10 p-2">
                                    <div className="text-xs text-muted-foreground">Age Range</div>
                                    <div className="font-bold text-green-700 dark:text-green-400">18-96 yrs</div>
                                </div>
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Cross-sectional MRI data from Washington University. Freely available for research,
                                enabling reproducible science and global collaboration.
                            </p>
                            <a
                                href="https://www.oasis-brains.org/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-green-600 hover:underline"
                            >
                                Visit OASIS Project <ExternalLink className="h-3 w-3" />
                            </a>
                        </div>

                        <div className="rounded-xl border border-amber-500/30 bg-gradient-to-br from-amber-500/10 to-transparent p-4 space-y-3">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                                <div className="flex items-center gap-2">
                                    <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center shrink-0">
                                        <Lock className="h-5 w-5 text-amber-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-semibold text-amber-700 dark:text-amber-400">ADNI-1</h3>
                                        <p className="text-xs text-muted-foreground">Alzheimer's Disease Neuroimaging Initiative</p>
                                    </div>
                                </div>
                                <Badge variant="outline" className="text-amber-600 border-amber-500/50 self-start sm:self-auto">
                                    Application Required
                                </Badge>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                                <div className="rounded-lg bg-amber-500/10 p-2">
                                    <div className="text-xs text-muted-foreground">Subjects</div>
                                    <div className="font-bold text-amber-700 dark:text-amber-400">629</div>
                                </div>
                                <div className="rounded-lg bg-amber-500/10 p-2">
                                    <div className="text-xs text-muted-foreground">Modalities</div>
                                    <div className="font-bold text-amber-700 dark:text-amber-400">MRI, PET+</div>
                                </div>
                            </div>
                            <Alert className="border-amber-500/30 bg-amber-500/10 py-2">
                                <Shield className="h-3 w-3 text-amber-600" />
                                <div className="ml-2 text-xs">
                                    <strong className="text-amber-700 dark:text-amber-400">Controlled Access:</strong>
                                    <span className="text-muted-foreground ml-1">
                                        Requires formal application and Data Use Agreement approval
                                    </span>
                                </div>
                            </Alert>
                            <a
                                href="https://adni.loni.usc.edu/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-amber-600 hover:underline"
                            >
                                Apply for ADNI Access <ExternalLink className="h-3 w-3" />
                            </a>
                        </div>
                    </div>

                    {/* Why This Matters */}
                    <div className="rounded-xl bg-muted/50 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                            <Award className="h-5 w-5 text-purple-500" />
                            <h3 className="font-semibold">Why This Research Matters</h3>
                        </div>
                        <div className="grid gap-3 md:grid-cols-3 text-sm">
                            <div className="flex items-start gap-2">
                                <Microscope className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-medium">Multi-Site Validation</div>
                                    <div className="text-xs text-muted-foreground">
                                        Cross-dataset robustness testing across different scanners and protocols
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-start gap-2">
                                <Brain className="h-4 w-4 text-purple-500 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-medium">1,065 Total Subjects</div>
                                    <div className="text-xs text-muted-foreground">
                                        Combined analysis from two independent research cohorts
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-start gap-2">
                                <Shield className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                                <div>
                                    <div className="font-medium">Ethical Compliance</div>
                                    <div className="text-xs text-muted-foreground">
                                        All data obtained through proper institutional agreements
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Hero Image Section */}
            <div className="relative rounded-2xl overflow-hidden border border-border/50">
                <div className="absolute inset-0 bg-gradient-to-r from-background via-transparent to-background z-10" />
                <div className="relative h-48 md:h-64 w-full">
                    <Image
                        src="https://images.unsplash.com/photo-1559757175-5700dde675bc?w=1200&q=80"
                        alt="Brain MRI scan visualization"
                        fill
                        className="object-cover opacity-60"
                        unoptimized
                    />
                </div>
                <div className="absolute inset-0 flex items-center justify-center z-20">
                    <div className="text-center space-y-2">
                        <h2 className="text-2xl md:text-3xl font-bold">Deep Learning for Neuroimaging</h2>
                        <p className="text-sm text-muted-foreground max-w-md mx-auto">
                            Advancing early dementia detection through multimodal MRI analysis and cross-dataset validation
                        </p>
                    </div>
                </div>
            </div>

            {/* Overview Cards */}
            <div className="grid gap-4 md:grid-cols-3">
                <Card className="border-green-500/20 bg-green-500/5">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                            Data Integrity
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-1">
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">100%</div>
                            <p className="text-xs text-muted-foreground">
                                Zero leakage verified
                            </p>
                            <p className="text-xs text-muted-foreground">
                                7 cleaning steps documented
                            </p>
                            <p className="text-xs text-muted-foreground">
                                Subject-wise splits enforced
                            </p>
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-emerald-500/20 bg-emerald-500/5">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                            <TrendingUp className="h-4 w-4 text-emerald-500" />
                            Honest Results
                            <Badge className="ml-1 text-[10px] bg-emerald-600">Level-MAX</Badge>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-1">
                            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">0.81 AUC</div>
                            <p className="text-xs text-emerald-600 font-semibold">
                                Level-MAX (with biomarkers)
                            </p>
                            <p className="text-xs text-muted-foreground">
                                Level-1: 0.60 (Age/Sex only)
                            </p>
                            <p className="text-xs text-muted-foreground">
                                +16.5% with CSF, APOE4, Volumetrics
                            </p>
                        </div>
                    </CardContent>
                </Card>

                <Card className="border-emerald-500/20 bg-emerald-500/5">
                    <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                            Publication Ready
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-1">
                            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">0.81 AUC</div>
                            <p className="text-xs text-emerald-600 font-semibold">
                                ✅ Target exceeded!
                            </p>
                            <p className="text-xs text-muted-foreground">
                                Level-MAX with 14D biomarkers
                            </p>
                            <p className="text-xs text-muted-foreground">
                                Publishable competitive result
                            </p>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Main Documentation Sections */}
            <div className="space-y-6">
                {/* Section 1: Data Cleaning */}
                <Card>
                    <CardHeader>
                        <div className="flex items-start justify-between">
                            <div className="space-y-1">
                                <CardTitle className="text-xl flex items-center gap-2">
                                    <Database className="h-5 w-5 text-primary" />
                                    Data Cleaning & Preprocessing
                                </CardTitle>
                                <CardDescription>
                                    Complete enumeration of structural and semantic data cleaning steps
                                </CardDescription>
                            </div>
                            <Badge className="bg-green-500/10 text-green-600 border-green-500/20">
                                Complete
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">7 Major Cleaning Steps</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        Subject-level de-duplication
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        Baseline-only visit selection
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        Removal of longitudinal leakage
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        Subject-wise train/test splitting
                                    </li>
                                </ul>
                            </div>
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">Data Flow</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li>ADNI: 1,825 scans → 629 subjects (-65.5%)</li>
                                    <li>OASIS: 436 scans → 205 usable (-52.8%)</li>
                                    <li>Feature intersection: MRI (512) + Clin (2)</li>
                                    <li>Level-1.5 target: + CSF (3) + APOE4 (1)</li>
                                </ul>
                            </div>
                        </div>

                        <div className="pt-2">
                            <h3 className="font-medium text-sm mb-2">Key Highlights</h3>
                            <div className="grid gap-2 md:grid-cols-2 text-xs">
                                <Alert className="border-green-500/20 bg-green-500/5">
                                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                                    <div className="ml-2">
                                        <strong className="text-green-700 dark:text-green-400">Zero Leakage:</strong>
                                        <span className="text-muted-foreground ml-1">
                                            Temporal, subject, label, and distribution leakage all prevented
                                        </span>
                                    </div>
                                </Alert>
                                <Alert className="border-blue-500/20 bg-blue-500/5">
                                    <Zap className="h-3 w-3 text-blue-500" />
                                    <div className="ml-2">
                                        <strong className="text-blue-700 dark:text-blue-400">Feature Exclusion:</strong>
                                        <span className="text-muted-foreground ml-1">
                                            MMSE, CDR-SB, ADAS excluded to prevent circular reasoning
                                        </span>
                                    </div>
                                </Alert>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Infrastructure Constraints Section */}
                <Card className="border-yellow-500/20">
                    <CardHeader>
                        <div className="flex items-start justify-between">
                            <div className="space-y-1">
                                <CardTitle className="text-xl flex items-center gap-2">
                                    <Database className="h-5 w-5 text-yellow-600" />
                                    Infrastructure & Computational Constraints
                                </CardTitle>
                                <CardDescription>
                                    Practical limitations that influenced data subset selection
                                </CardDescription>
                            </div>
                            <Badge className="bg-yellow-500/10 text-yellow-600 border-yellow-500/20">
                                Methodological Note
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">Storage Requirements</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-600 font-bold mt-0.5">→</span>
                                        <span><strong>OASIS-1 raw:</strong> 50GB compressed → 70GB extracted</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-600 font-bold mt-0.5">→</span>
                                        <span><strong>ADNI-1 raw:</strong> Similar size (50GB+ compressed)</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-600 font-bold mt-0.5">→</span>
                                        <span><strong>Feature extraction:</strong> Intermediate files (preprocessed MRI)</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-600 font-bold mt-0.5">→</span>
                                        <span><strong>Model checkpoints:</strong> Training artifacts, logs</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-600 font-bold mt-0.5">→</span>
                                        <span className="font-bold text-yellow-700 dark:text-yellow-500">Total pipeline: 200GB+</span>
                                    </li>
                                </ul>
                            </div>
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">Impact on Research Design</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li>Used baseline-only scans (not full longitudinal)</li>
                                    <li>Extracted features once, stored as .npz (compressed)</li>
                                    <li>Limited to OASIS-1 and ADNI-1 (not OASIS-2/3, ADNI-2/3)</li>
                                    <li>Focused on structural MRI (excluded PET, DTI)</li>
                                </ul>
                            </div>
                        </div>

                        <div className="pt-2">
                            <h3 className="font-medium text-sm mb-2">Justification & Context</h3>
                            <Alert className="border-yellow-500/20 bg-yellow-500/5">
                                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                                <div className="ml-3 text-sm">
                                    <strong className="text-yellow-700 dark:text-yellow-500">This is not an excuse - it's a real constraint.</strong>
                                    <p className="text-muted-foreground mt-1">
                                        Many researchers face infrastructure limitations. What matters is:
                                        (1) we documented this constraint transparently,
                                        (2) we ensured the data we DID use was rigorously cleaned,
                                        (3) we didn't cherry-pick favorable subsets - we used standard baseline protocols.
                                    </p>
                                    <p className="text-muted-foreground mt-2">
                                        <strong className="text-yellow-700 dark:text-yellow-500">Sample size (N=205-629)</strong> is comparable to many published studies.
                                        Our contribution lies in <strong>honest methodology</strong> and <strong>cross-dataset validation</strong>,
                                        not maximal dataset size.
                                    </p>
                                </div>
                            </Alert>
                        </div>

                        <div className="grid gap-2 md:grid-cols-2 text-xs">
                            <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
                                <div className="font-medium text-blue-700 dark:text-blue-400 mb-1">What We Did</div>
                                <ul className="space-y-0.5 text-muted-foreground">
                                    <li>✓ Selected baseline scans (standard protocol)</li>
                                    <li>✓ De-duplicated subjects rigorously</li>
                                    <li>✓ Used all available baseline data</li>
                                    <li>✓ Documented storage constraints</li>
                                </ul>
                            </div>
                            <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
                                <div className="font-medium text-red-700 dark:text-red-400 mb-1">What We Avoided</div>
                                <ul className="space-y-0.5 text-muted-foreground">
                                    <li>✗ Cherry-picking "easy" subjects</li>
                                    <li>✗ Hiding infrastructure limitations</li>
                                    <li>✗ Using only favorable scans</li>
                                    <li>✗ Inflating results with circular features</li>
                                </ul>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Section 2: Honest Assessment */}
                <Card>
                    <CardHeader>
                        <div className="flex items-start justify-between">
                            <div className="space-y-1">
                                <CardTitle className="text-xl flex items-center gap-2">
                                    <BarChart3 className="h-5 w-5 text-orange-500" />
                                    Honest Project Assessment
                                </CardTitle>
                                <CardDescription>
                                    Why fusion models underperform and what the results actually mean
                                </CardDescription>
                            </div>
                            <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">
                                Critical Analysis
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">The Evolution</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li className="flex items-start gap-2">
                                        <XCircle className="h-3 w-3 text-red-500 mt-0.5" />
                                        <span>ADNI Level-1: 0.60 AUC (Age/Sex only)</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-emerald-500 mt-0.5" />
                                        <span className="text-emerald-600 font-semibold">Level-MAX: 0.81 AUC (+16.5% with biomarkers!)</span>
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500 mt-0.5" />
                                        <span>Level-2 (with MMSE): 0.99 AUC (circular)</span>
                                    </li>
                                </ul>
                            </div>
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">Root Causes</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li>Feature quality mismatch (512 strong vs 2 weak)</li>
                                    <li>Dimension imbalance (2 → 32 creates 30 dims of noise)</li>
                                    <li>Small dataset + high variance (N=205-629)</li>
                                    <li>Age as confounder, not biomarker</li>
                                </ul>
                            </div>
                        </div>

                        <div className="pt-2">
                            <h3 className="font-medium text-sm mb-2">The Breakthrough</h3>
                            <Alert className="border-emerald-500/20 bg-emerald-500/10">
                                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                <div className="ml-3 text-sm">
                                    <strong className="text-emerald-700 dark:text-emerald-400">Level-MAX proves fusion works with quality features!</strong>
                                    <p className="text-muted-foreground mt-1">
                                        By enriching clinical features from 2D (Age/Sex) to 14D biological profile (CSF, APOE4, Volumetrics),
                                        we achieved <strong className="text-emerald-600">0.81 AUC</strong> - validating that the fusion architecture was never broken,
                                        it just needed complementary biological signals instead of weak demographics.
                                    </p>
                                </div>
                            </Alert>
                        </div>
                    </CardContent>
                </Card>

                {/* Section 3: Publication Strategy */}
                <Card>
                    <CardHeader>
                        <div className="flex items-start justify-between">
                            <div className="space-y-1">
                                <CardTitle className="text-xl flex items-center gap-2">
                                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                                    Level-MAX Achievement
                                </CardTitle>
                                <CardDescription>
                                    How we achieved competitive 0.81 AUC with biomarker-enhanced fusion
                                </CardDescription>
                            </div>
                            <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
                                ✅ Completed
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">What We Implemented</h3>
                                <ul className="text-sm space-y-1 text-muted-foreground">
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                                        <span className="text-emerald-600 font-semibold">14D Biological Profile (Level-MAX)</span>
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        CSF biomarkers (ABETA, TAU, PTAU)
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        APOE4 genetic risk factor
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        7 Volumetric measures (Hippocampus, etc.)
                                    </li>
                                    <li className="flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-green-500" />
                                        Still honest (no cognitive scores)
                                    </li>
                                </ul>
                            </div>
                            <div className="space-y-2">
                                <h3 className="font-medium text-sm">Achieved Results</h3>
                                <div className="space-y-3">
                                    <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                                        <div className="text-xs text-muted-foreground">Late Fusion AUC</div>
                                        <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                                            0.808
                                        </div>
                                        <div className="text-xs text-emerald-600 font-semibold mt-1">
                                            +16.5% gain over MRI-only!
                                        </div>
                                    </div>
                                    <div className="text-xs text-muted-foreground">
                                        <strong className="text-emerald-600">Status: ✅ Complete</strong><br />
                                        <strong>Publishable:</strong> Yes - competitive result
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="pt-2">
                            <h3 className="font-medium text-sm mb-2">Week-by-Week Plan</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex items-start gap-3">
                                    <Badge variant="outline" className="mt-0.5">Week 1</Badge>
                                    <span className="text-muted-foreground">
                                        Extract biomarkers from ADNIMERGE, verify CSF coverage (expected ~400/629 subjects)
                                    </span>
                                </div>
                                <div className="flex items-start gap-3">
                                    <Badge variant="outline" className="mt-0.5">Week 2</Badge>
                                    <span className="text-muted-foreground">
                                        Modify training script (clinical_dim: 2 → 6), retrain all models
                                    </span>
                                </div>
                                <div className="flex items-start gap-3">
                                    <Badge variant="outline" className="mt-0.5">Week 3</Badge>
                                    <span className="text-muted-foreground">
                                        Write paper draft, create figures, submit to target venue
                                    </span>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Download Section */}
            <Card className="border-primary/20">
                <CardHeader>
                    <CardTitle className="text-lg">Access Complete Documentation</CardTitle>
                    <CardDescription>
                        Download the full markdown files for thesis integration
                    </CardDescription>
                </CardHeader>
                <CardContent className="grid gap-3 md:grid-cols-3">
                    <a
                        href="/DATA_CLEANING_AND_PREPROCESSING.md"
                        download
                        className="flex items-center gap-2 rounded-lg border p-3 hover:bg-accent transition-colors"
                    >
                        <FileText className="h-4 w-4 text-primary" />
                        <div className="text-sm">
                            <div className="font-medium">Data Cleaning</div>
                            <div className="text-xs text-muted-foreground">20+ pages · Thesis-ready</div>
                        </div>
                    </a>
                    <a
                        href="/PROJECT_ASSESSMENT_HONEST_TAKE.md"
                        download
                        className="flex items-center gap-2 rounded-lg border p-3 hover:bg-accent transition-colors"
                    >
                        <FileText className="h-4 w-4 text-orange-500" />
                        <div className="text-sm">
                            <div className="font-medium">Honest Assessment</div>
                            <div className="text-xs text-muted-foreground">15+ pages · Critical analysis</div>
                        </div>
                    </a>
                    <a
                        href="/REALISTIC_PATH_TO_PUBLICATION.md"
                        download
                        className="flex items-center gap-2 rounded-lg border p-3 hover:bg-accent transition-colors"
                    >
                        <FileText className="h-4 w-4 text-purple-500" />
                        <div className="text-sm">
                            <div className="font-medium">Publication Path</div>
                            <div className="text-xs text-muted-foreground">12+ pages · Action plan</div>
                        </div>
                    </a>
                </CardContent>
            </Card>

            {/* Author Section */}
            <Card className="border-primary/30 bg-gradient-to-br from-primary/5 to-transparent">
                <CardContent className="py-6">
                    <div className="flex flex-col sm:flex-row items-center gap-6">
                        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold">
                            VS
                        </div>
                        <div className="flex-1 text-center sm:text-left">
                            <h3 className="text-lg font-bold">Vishesh Sanghvi</h3>
                            <p className="text-sm text-muted-foreground mb-3">
                                Researcher & Developer · Deep Learning for Healthcare
                            </p>
                            <div className="flex items-center justify-center sm:justify-start gap-3">
                                <a
                                    href="https://www.visheshsanghvi.me/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                                >
                                    <Globe className="h-4 w-4" /> Portfolio
                                </a>
                                <a
                                    href="https://linkedin.com/in/vishesh-sanghvi-96b16a237/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-sm text-[#0A66C2] hover:underline"
                                >
                                    <Linkedin className="h-4 w-4" /> LinkedIn
                                </a>
                                <a
                                    href="https://github.com/visheshsanghvi112"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-sm hover:underline"
                                >
                                    <Github className="h-4 w-4" /> GitHub
                                </a>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Footer */}
            <div className="text-center text-xs text-muted-foreground">
                All documentation generated: December 29, 2025 · Research validated on OASIS-1 & ADNI-1
            </div>
        </div>
    )
}
