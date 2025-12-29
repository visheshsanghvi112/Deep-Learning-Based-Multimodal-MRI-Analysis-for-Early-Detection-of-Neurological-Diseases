"use client"
// ADNI Page - Cross-Dataset Validation

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
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Users, Activity, Database, Eye, Layers, AlertTriangle, TrendingUp, ArrowRightLeft, Sparkles } from "lucide-react"
import { AdniDataExplorer } from "@/components/AdniDataExplorer"

// ADNI dataset statistics
const adniStats = {
    totalSubjects: 629,
    trainSubjects: 503,
    testSubjects: 126,
    groups: { CN: 194, MCI: 302, AD: 133 },
    mriDim: 512,
    clinicalFeatures: ["Age", "Sex", "Education", "APOE4"],
}

// ADNI sample subjects
const sampleSubjects = [
    { id: "002_S_0295", age: 76, sex: "M", group: "CN", mmse: 29, educ: 16 },
    { id: "002_S_0413", age: 82, sex: "F", group: "MCI", mmse: 26, educ: 14 },
    { id: "002_S_0619", age: 71, sex: "M", group: "AD", mmse: 21, educ: 18 },
    { id: "003_S_0907", age: 68, sex: "F", group: "CN", mmse: 30, educ: 16 },
    { id: "003_S_1059", age: 79, sex: "M", group: "MCI", mmse: 24, educ: 12 },
    { id: "005_S_0223", age: 85, sex: "F", group: "AD", mmse: 18, educ: 14 },
]

// Cross-dataset robustness results
const robustnessResults = {
    oasisToAdni: [
        { model: "MRI-Only", srcAuc: 0.814, tgtAuc: 0.607, drop: 0.207 },
        { model: "Late Fusion", srcAuc: 0.864, tgtAuc: 0.575, drop: 0.289 },
        { model: "Attention Fusion", srcAuc: 0.826, tgtAuc: 0.557, drop: 0.269 },
    ],
    adniToOasis: [
        { model: "MRI-Only", srcAuc: 0.686, tgtAuc: 0.569, drop: 0.117 },
        { model: "Late Fusion", srcAuc: 0.734, tgtAuc: 0.624, drop: 0.110 },
        { model: "Attention Fusion", srcAuc: 0.713, tgtAuc: 0.548, drop: 0.165 },
    ],
}

// All available MRI samples
const allMriSamples = [
    { src: "/adni-samples/cn_1.gif", group: "CN", label: "CN Subject 1" },
    { src: "/adni-samples/cn_2.gif", group: "CN", label: "CN Subject 2" },
    { src: "/adni-samples/cn_3.gif", group: "CN", label: "CN Subject 3" },
    { src: "/adni-samples/cn_4.gif", group: "CN", label: "CN Subject 4" },
    { src: "/adni-samples/cn_5.gif", group: "CN", label: "CN Subject 5" },
    { src: "/adni-samples/cn_7.gif", group: "CN", label: "CN Subject 6" },
    { src: "/adni-samples/mci_8.gif", group: "MCI", label: "MCI Subject 1" },
    { src: "/adni-samples/mci_9.gif", group: "MCI", label: "MCI Subject 2" },
    { src: "/adni-samples/ad_6.gif", group: "AD", label: "AD Subject 1" },
]

function MRISamplesGallery() {
    const [showAll, setShowAll] = useState(false)

    const groupColors: Record<string, string> = {
        CN: "bg-green-500/10 text-green-600 border-green-500/20",
        MCI: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
        AD: "bg-red-500/10 text-red-600 border-red-500/20",
    }

    const displaySamples = showAll ? allMriSamples : allMriSamples.slice(0, 3)

    return (
        <Card className="mt-6">
            <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                    <Eye className="h-5 w-5 text-teal-500" />
                    Sample MRI Scans
                    <Badge variant="outline" className="ml-2">{allMriSamples.length} samples</Badge>
                </CardTitle>
                <CardDescription>
                    Animated sagittal slices from ADNI-1 subjects across diagnostic groups
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className={`grid gap-4 ${showAll ? 'grid-cols-3 md:grid-cols-3 lg:grid-cols-3' : 'grid-cols-3'}`}>
                    {displaySamples.map((sample, idx) => (
                        <div key={idx} className="space-y-2">
                            <Badge className={groupColors[sample.group]}>{sample.group}</Badge>
                            <div className="aspect-square rounded-lg overflow-hidden border bg-black">
                                <img
                                    src={sample.src}
                                    alt={sample.label}
                                    className="w-full h-full object-contain"
                                    loading="lazy"
                                    onError={(e) => { e.currentTarget.style.display = 'none' }}
                                />
                            </div>
                            <p className="text-xs text-muted-foreground text-center">{sample.label}</p>
                        </div>
                    ))}
                </div>

                <div className="flex flex-col items-center gap-3 mt-4">
                    <button
                        onClick={() => setShowAll(!showAll)}
                        className="px-4 py-2 text-sm font-medium rounded-lg bg-teal-500/10 text-teal-600 hover:bg-teal-500/20 transition-colors flex items-center gap-2"
                    >
                        {showAll ? (
                            <>Show Less</>
                        ) : (
                            <>Show All {allMriSamples.length} Samples</>
                        )}
                    </button>
                    <p className="text-xs text-muted-foreground text-center">
                        Animated GIFs show 20 sagittal slices near the brain midline.
                    </p>
                </div>
            </CardContent>
        </Card>
    )
}

export default function ADNIPage() {
    const [selectedView, setSelectedView] = useState("overview")

    return (
        <div className="flex w-full flex-col gap-8">
            {/* Header */}
            <section className="space-y-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600">
                        <Brain className="h-6 w-6 text-white" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">
                            ADNI-1 Dataset & Cross-Dataset Analysis
                        </h2>
                        <p className="text-sm text-muted-foreground">
                            Alzheimer's Disease Neuroimaging Initiative - Robustness Validation
                        </p>
                    </div>
                </div>
                <div className="flex flex-wrap gap-2">
                    <Badge className="bg-teal-500/10 text-teal-600 border-teal-500/20">629 Subjects</Badge>
                    <Badge className="bg-cyan-500/10 text-cyan-600 border-cyan-500/20">Multi-Site</Badge>
                    <Badge className="bg-purple-500/10 text-purple-600 border-purple-500/20">1.5T MRI</Badge>
                    <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">Cross-Dataset Tested</Badge>
                </div>
            </section>

            {/* Stats Cards */}
            <section className="grid gap-4 grid-cols-2 md:grid-cols-4">
                <Card className="border-l-4 border-l-teal-500">
                    <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                            <Users className="h-8 w-8 text-teal-500" />
                            <div>
                                <div className="text-2xl font-bold">{adniStats.totalSubjects}</div>
                                <div className="text-xs text-muted-foreground">Total Subjects</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
                <Card className="border-l-4 border-l-green-500">
                    <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                            <Activity className="h-8 w-8 text-green-500" />
                            <div>
                                <div className="text-2xl font-bold">{adniStats.groups.CN}</div>
                                <div className="text-xs text-muted-foreground">Cognitively Normal</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
                <Card className="border-l-4 border-l-yellow-500">
                    <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                            <Layers className="h-8 w-8 text-yellow-500" />
                            <div>
                                <div className="text-2xl font-bold">{adniStats.groups.MCI}</div>
                                <div className="text-xs text-muted-foreground">Mild Cognitive Impairment</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
                <Card className="border-l-4 border-l-red-500">
                    <CardContent className="pt-4">
                        <div className="flex items-center gap-3">
                            <Database className="h-8 w-8 text-red-500" />
                            <div>
                                <div className="text-2xl font-bold">{adniStats.groups.AD}</div>
                                <div className="text-xs text-muted-foreground">Alzheimer's Disease</div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </section>

            {/* Main Tabs */}
            <Tabs value={selectedView} onValueChange={setSelectedView} className="w-full">
                <TabsList className="flex w-full overflow-x-auto md:grid md:grid-cols-5 scrollbar-hide">
                    <TabsTrigger value="overview" className="flex items-center gap-2">
                        <Eye className="h-4 w-4" />
                        <span className="hidden sm:inline">Overview</span>
                    </TabsTrigger>
                    <TabsTrigger value="results" className="flex items-center gap-2">
                        <TrendingUp className="h-4 w-4" />
                        <span className="hidden sm:inline">Results</span>
                    </TabsTrigger>
                    <TabsTrigger value="robustness" className="flex items-center gap-2">
                        <ArrowRightLeft className="h-4 w-4" />
                        <span className="hidden sm:inline">Robustness</span>
                    </TabsTrigger>
                    <TabsTrigger value="explorer" className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4" />
                        <span className="hidden sm:inline">Explorer</span>
                    </TabsTrigger>
                    <TabsTrigger value="subjects" className="flex items-center gap-2">
                        <Users className="h-4 w-4" />
                        <span className="hidden sm:inline">Subjects</span>
                    </TabsTrigger>
                </TabsList>

                {/* Overview Tab */}
                <TabsContent value="overview" className="mt-6">
                    <div className="grid gap-6 md:grid-cols-2">
                        <Card>
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Brain className="h-5 w-5 text-teal-500" />
                                    About ADNI-1
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4 text-sm text-muted-foreground">
                                <p>
                                    The <strong className="text-foreground">Alzheimer's Disease Neuroimaging Initiative (ADNI)</strong> is
                                    a multi-site longitudinal study tracking biomarkers for AD progression.
                                </p>
                                <p>
                                    ADNI-1 contains <strong className="text-foreground">629 baseline scans</strong> from
                                    CN, MCI, and AD subjects across 50+ sites in North America.
                                </p>
                                <div className="grid grid-cols-2 gap-4 pt-4">
                                    <div className="p-3 rounded-lg bg-muted/50">
                                        <div className="text-lg font-bold text-foreground">55-90</div>
                                        <div className="text-xs">Age Range (years)</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-muted/50">
                                        <div className="text-lg font-bold text-foreground">1.5T</div>
                                        <div className="text-xs">MRI Field Strength</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-muted/50">
                                        <div className="text-lg font-bold text-foreground">50+</div>
                                        <div className="text-xs">Collection Sites</div>
                                    </div>
                                    <div className="p-3 rounded-lg bg-muted/50">
                                        <div className="text-lg font-bold text-foreground">T1-W</div>
                                        <div className="text-xs">MRI Modality</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                                    Label Definition Shift
                                </CardTitle>
                                <CardDescription>Critical difference from OASIS</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                                        <h4 className="font-semibold text-yellow-600 mb-2">⚠️ Important</h4>
                                        <p className="text-sm text-muted-foreground">
                                            <strong className="text-foreground">ADNI Positive Class</strong> = MCI + AD (broader disease spectrum)
                                        </p>
                                        <p className="text-sm text-muted-foreground mt-1">
                                            <strong className="text-foreground">OASIS Positive Class</strong> = CDR 0.5 only (very mild / early)
                                        </p>
                                    </div>
                                    <p className="text-sm text-muted-foreground">
                                        This <strong className="text-foreground">label shift</strong> explains why cross-dataset
                                        transfer shows accuracy collapse (~34%) despite reasonable AUC (~0.60).
                                    </p>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* MRI Samples Section */}
                    <MRISamplesGallery />
                </TabsContent>

                {/* Results Tab */}
                <TabsContent value="results" className="mt-6">
                    <div className="space-y-6">
                        {/* Level 1 Results */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5 text-green-500" />
                                    ADNI Level-1: Honest Baseline
                                </CardTitle>
                                <CardDescription>
                                    MRI + Basic Demographics (Age, Sex). No cognitive scores. Realistic early detection.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid gap-4 md:grid-cols-3">
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="text-sm text-muted-foreground mb-1">MRI-Only AUC</div>
                                        <div className="text-3xl font-bold text-blue-600">0.583</div>
                                        <div className="text-xs text-muted-foreground mt-1">CI: 0.47 - 0.68</div>
                                    </div>
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="text-sm text-muted-foreground mb-1">Late Fusion AUC</div>
                                        <div className="text-3xl font-bold text-purple-600">0.598</div>
                                        <div className="text-xs text-muted-foreground mt-1">+1.5% vs MRI</div>
                                    </div>
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="text-sm text-muted-foreground mb-1">Key Finding</div>
                                        <div className="text-sm">Minimal multimodal benefit with basic demographics only.</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Level 2 Results */}
                        <Card className="border-orange-500/30">
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-orange-500" />
                                    ADNI Level-2: Circular Upper Bound
                                </CardTitle>
                                <CardDescription>
                                    ⚠️ Uses MMSE/CDR-SB. NOT for early detection claims. Reference only.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid gap-4 md:grid-cols-3">
                                    <div className="p-4 rounded-lg border bg-orange-500/5 border-orange-500/20">
                                        <div className="text-sm text-muted-foreground mb-1">Late Fusion AUC</div>
                                        <div className="text-3xl font-bold text-orange-600">0.988</div>
                                        <div className="text-xs text-orange-600 mt-1">⚠️ Circular (MMSE used)</div>
                                    </div>
                                    <div className="p-4 rounded-lg border bg-card col-span-2">
                                        <div className="text-sm font-medium mb-2">Why so high?</div>
                                        <p className="text-sm text-muted-foreground">
                                            MMSE and CDR-SB are <strong>cognitively downstream</strong> measures that
                                            directly define the diagnostic groups. Using them is circular and
                                            demonstrates the model's capacity, not real-world early detection ability.
                                        </p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* Robustness Tab */}
                <TabsContent value="robustness" className="mt-6">
                    <div className="space-y-6">
                        {/* OASIS -> ADNI */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <ArrowRightLeft className="h-5 w-5 text-blue-500" />
                                    Experiment A: OASIS → ADNI
                                </CardTitle>
                                <CardDescription>
                                    Train on high-quality single-site data, test on heterogeneous multi-site data.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
                                    <Table className="min-w-[450px]">
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Model</TableHead>
                                                <TableHead>Source AUC</TableHead>
                                                <TableHead>Target AUC</TableHead>
                                                <TableHead>Drop</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {robustnessResults.oasisToAdni.map((r) => (
                                                <TableRow key={r.model}>
                                                    <TableCell className="font-medium">{r.model}</TableCell>
                                                    <TableCell>{r.srcAuc.toFixed(3)}</TableCell>
                                                    <TableCell className={r.model === "MRI-Only" ? "text-green-600 font-bold" : ""}>
                                                        {r.tgtAuc.toFixed(3)}
                                                    </TableCell>
                                                    <TableCell className="text-red-500">-{r.drop.toFixed(3)}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </div>
                                <div className="mt-4 p-3 rounded-lg bg-green-500/10 border border-green-500/20 text-sm">
                                    <strong className="text-green-600">Finding:</strong> MRI-Only was most robust (0.607).
                                    Adding clinical features hurt transfer performance.
                                </div>
                            </CardContent>
                        </Card>

                        {/* ADNI -> OASIS */}
                        <Card>
                            <CardHeader>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <ArrowRightLeft className="h-5 w-5 text-purple-500" />
                                    Experiment B: ADNI → OASIS
                                </CardTitle>
                                <CardDescription>
                                    Train on heterogeneous multi-site data, test on high-quality single-site data.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
                                    <Table className="min-w-[450px]">
                                        <TableHeader>
                                            <TableRow>
                                                <TableHead>Model</TableHead>
                                                <TableHead>Source AUC</TableHead>
                                                <TableHead>Target AUC</TableHead>
                                                <TableHead>Drop</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {robustnessResults.adniToOasis.map((r) => (
                                                <TableRow key={r.model}>
                                                    <TableCell className="font-medium">{r.model}</TableCell>
                                                    <TableCell>{r.srcAuc.toFixed(3)}</TableCell>
                                                    <TableCell className={r.model === "Late Fusion" ? "text-purple-600 font-bold" : ""}>
                                                        {r.tgtAuc.toFixed(3)}
                                                    </TableCell>
                                                    <TableCell className="text-red-500">-{r.drop.toFixed(3)}</TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </div>
                                <div className="mt-4 p-3 rounded-lg bg-purple-500/10 border border-purple-500/20 text-sm">
                                    <strong className="text-purple-600">Finding:</strong> Late Fusion was most robust (0.624).
                                    Clinical features helped transfer in this direction.
                                </div>
                            </CardContent>
                        </Card>

                        {/* Key Insights */}
                        <Card className="bg-gradient-to-r from-blue-500/5 to-purple-500/5">
                            <CardHeader>
                                <CardTitle className="text-lg">Key Robustness Insights</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3 text-sm">
                                <div className="flex gap-2">
                                    <Badge variant="outline" className="border-blue-500 text-blue-600">1</Badge>
                                    <span><strong>Asymmetric robustness:</strong> Best model depends on transfer direction.</span>
                                </div>
                                <div className="flex gap-2">
                                    <Badge variant="outline" className="border-yellow-500 text-yellow-600">2</Badge>
                                    <span><strong>Attention Fusion unstable:</strong> Consistently underperformed across transfers.</span>
                                </div>
                                <div className="flex gap-2">
                                    <Badge variant="outline" className="border-green-500 text-green-600">3</Badge>
                                    <span><strong>OASIS as Teacher:</strong> OASIS-trained MRI model (0.607) beat ADNI's own baseline (0.583).</span>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* Explorer Tab */}
                <TabsContent value="explorer" className="mt-6">
                    <AdniDataExplorer />
                </TabsContent>

                {/* Subjects Tab */}
                <TabsContent value="subjects" className="mt-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-lg flex items-center gap-2">
                                <Users className="h-5 w-5 text-teal-500" />
                                Sample Subjects
                            </CardTitle>
                            <CardDescription>
                                Representative examples from each diagnostic group
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
                                <Table className="min-w-[500px]">
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Subject ID</TableHead>
                                            <TableHead>Age</TableHead>
                                            <TableHead>Sex</TableHead>
                                            <TableHead>Group</TableHead>
                                            <TableHead>MMSE</TableHead>
                                            <TableHead>Education</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {sampleSubjects.map((s) => (
                                            <TableRow key={s.id}>
                                                <TableCell className="font-mono text-xs">{s.id}</TableCell>
                                                <TableCell>{s.age}</TableCell>
                                                <TableCell>{s.sex}</TableCell>
                                                <TableCell>
                                                    <Badge
                                                        variant={s.group === "CN" ? "default" : s.group === "MCI" ? "secondary" : "destructive"}
                                                    >
                                                        {s.group}
                                                    </Badge>
                                                </TableCell>
                                                <TableCell>{s.mmse}</TableCell>
                                                <TableCell>{s.educ} yrs</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>

                            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="p-4 rounded-lg border bg-green-500/5 border-green-500/20">
                                    <div className="text-sm font-medium text-green-600 mb-1">CN (Normal)</div>
                                    <div className="text-2xl font-bold">194</div>
                                    <div className="text-xs text-muted-foreground">Cognitively Normal</div>
                                </div>
                                <div className="p-4 rounded-lg border bg-yellow-500/5 border-yellow-500/20">
                                    <div className="text-sm font-medium text-yellow-600 mb-1">MCI</div>
                                    <div className="text-2xl font-bold">302</div>
                                    <div className="text-xs text-muted-foreground">Mild Cognitive Impairment</div>
                                </div>
                                <div className="p-4 rounded-lg border bg-red-500/5 border-red-500/20">
                                    <div className="text-sm font-medium text-red-600 mb-1">AD</div>
                                    <div className="text-2xl font-bold">133</div>
                                    <div className="text-xs text-muted-foreground">Alzheimer's Disease</div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>

            <Alert className="text-xs">
                ADNI data is used under the ADNI Data Use Agreement. This portal presents research results
                for methodological evaluation. Results are <strong>not intended for clinical use</strong>.
            </Alert>
        </div>
    )
}
