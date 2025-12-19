"use client"

import { useState } from "react"
import Image from "next/image"
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
import { Brain, Users, Activity, Database, Eye, Layers, FileImage, BarChart3, MousePointer2, Sparkles } from "lucide-react"
import { OasisDataExplorer } from "@/components/OasisDataExplorer"

// Sample data structure showing what the dataset contains
const sampleSubjects = [
  { id: "OAS1_0001", age: 74, gender: "F", cdr: 0, mmse: 29, nwbv: 0.736, diagnosis: "Normal" },
  { id: "OAS1_0002", age: 55, gender: "F", cdr: 0, mmse: 29, nwbv: 0.783, diagnosis: "Normal" },
  { id: "OAS1_0015", age: 76, gender: "F", cdr: 0.5, mmse: 28, nwbv: 0.711, diagnosis: "Very Mild Dementia" },
  { id: "OAS1_0021", age: 80, gender: "M", cdr: 0.5, mmse: 23, nwbv: 0.698, diagnosis: "Very Mild Dementia" },
  { id: "OAS1_0028", age: 86, gender: "F", cdr: 1.0, mmse: 27, nwbv: 0.671, diagnosis: "Mild Dementia" },
  { id: "OAS1_0031", age: 88, gender: "M", cdr: 1.0, mmse: 26, nwbv: 0.645, diagnosis: "Mild Dementia" },
]

// MRI sample images for visualization
const mriSamples = {
  normal: [
    { src: "/mri-samples/normal_1.gif", label: "Subject 0001", age: 74, cdr: 0 },
    { src: "/mri-samples/normal_2.gif", label: "Subject 0002", age: 55, cdr: 0 },
    { src: "/mri-samples/normal_4.gif", label: "Subject 0010", age: 74, cdr: 0 },
  ],
  mci: [
    { src: "/mri-samples/mci_1.gif", label: "Subject 0015", age: 76, cdr: 0.5 },
    { src: "/mri-samples/mci_2.gif", label: "Subject 0021", age: 80, cdr: 0.5 },
    { src: "/mri-samples/normal_3.gif", label: "Subject 0003", age: 73, cdr: 0.5 },
  ],
  dementia: [
    { src: "/mri-samples/dementia_1.gif", label: "Subject 0028", age: 86, cdr: 1.0 },
    { src: "/mri-samples/dementia_2.gif", label: "Subject 0031", age: 88, cdr: 1.0 },
  ],
}

// Feature descriptions
const features = [
  { name: "CDR", description: "Clinical Dementia Rating (0, 0.5, 1, 2)", type: "Label" },
  { name: "MMSE", description: "Mini-Mental State Examination (0-30)", type: "Clinical" },
  { name: "nWBV", description: "Normalized Whole Brain Volume", type: "Anatomical" },
  { name: "eTIV", description: "Estimated Total Intracranial Volume", type: "Anatomical" },
  { name: "ASF", description: "Atlas Scaling Factor", type: "Anatomical" },
  { name: "Age", description: "Subject age at scan (18-96 years)", type: "Demographic" },
  { name: "Education", description: "Years of education (1-5 scale)", type: "Demographic" },
  { name: "SES", description: "Socioeconomic Status (1-5 scale)", type: "Demographic" },
]

export default function DatasetPage() {
  const [selectedView, setSelectedView] = useState("explorer")

  return (
    <div className="flex w-full flex-col gap-8">
      {/* Header */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              OASIS-1 Dataset Explorer
            </h2>
            <p className="text-sm text-muted-foreground">
              Open Access Series of Imaging Studies - Cross-sectional MRI Data
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge className="bg-blue-500/10 text-blue-600 border-blue-500/20">436 Subjects</Badge>
          <Badge className="bg-green-500/10 text-green-600 border-green-500/20">205 Labeled</Badge>
          <Badge className="bg-purple-500/10 text-purple-600 border-purple-500/20">T1-Weighted MRI</Badge>
          <Badge className="bg-orange-500/10 text-orange-600 border-orange-500/20">Cross-sectional</Badge>
        </div>
      </section>

      {/* Stats Cards */}
      <section className="grid gap-4 grid-cols-2 md:grid-cols-4">
        <Card className="border-l-4 border-l-blue-500">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <Users className="h-8 w-8 text-blue-500" />
              <div>
                <div className="text-2xl font-bold">436</div>
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
                <div className="text-2xl font-bold">205</div>
                <div className="text-xs text-muted-foreground">CDR Labeled</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-purple-500">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <Layers className="h-8 w-8 text-purple-500" />
              <div>
                <div className="text-2xl font-bold">512</div>
                <div className="text-xs text-muted-foreground">CNN Features</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-l-4 border-l-orange-500">
          <CardContent className="pt-4">
            <div className="flex items-center gap-3">
              <Database className="h-8 w-8 text-orange-500" />
              <div>
                <div className="text-2xl font-bold">6</div>
                <div className="text-xs text-muted-foreground">Clinical Features</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Main Tabs */}
      <Tabs value={selectedView} onValueChange={setSelectedView} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="explorer" className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            <span className="hidden sm:inline">Explorer</span>
          </TabsTrigger>
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            <span className="hidden sm:inline">Overview</span>
          </TabsTrigger>
          <TabsTrigger value="mri" className="flex items-center gap-2">
            <FileImage className="h-4 w-4" />
            <span className="hidden sm:inline">MRI Samples</span>
          </TabsTrigger>
          <TabsTrigger value="features" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Features</span>
          </TabsTrigger>
          <TabsTrigger value="subjects" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            <span className="hidden sm:inline">Subjects</span>
          </TabsTrigger>
        </TabsList>

        {/* Interactive Explorer Tab - Immersive Full-Screen Mode */}
        <TabsContent value="explorer" className="mt-6 -mx-8 -mb-8">
          <OasisDataExplorer />
        </TabsContent>

        {/* Overview Tab */}
        <TabsContent value="overview" className="mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Brain className="h-5 w-5 text-blue-500" />
                  About OASIS-1
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm text-muted-foreground">
                <p>
                  The <strong className="text-foreground">Open Access Series of Imaging Studies (OASIS)</strong> is a 
                  project aimed at making neuroimaging datasets freely available to the scientific community.
                </p>
                <p>
                  OASIS-1 is a cross-sectional collection of <strong className="text-foreground">436 subjects</strong> aged 
                  18 to 96, including individuals with and without dementia.
                </p>
                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="p-3 rounded-lg bg-muted/50">
                    <div className="text-lg font-bold text-foreground">18-96</div>
                    <div className="text-xs">Age Range (years)</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <div className="text-lg font-bold text-foreground">1.5T</div>
                    <div className="text-xs">MRI Field Strength</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <div className="text-lg font-bold text-foreground">3-4</div>
                    <div className="text-xs">Scans per Subject</div>
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
                  <Activity className="h-5 w-5 text-green-500" />
                  CDR Distribution
                </CardTitle>
                <CardDescription>Clinical Dementia Rating breakdown</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>CDR 0 (Normal)</span>
                      <span className="font-medium">316 subjects</span>
                    </div>
                    <div className="h-3 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 rounded-full" style={{ width: "72%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>CDR 0.5 (Very Mild)</span>
                      <span className="font-medium">70 subjects</span>
                    </div>
                    <div className="h-3 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500 rounded-full" style={{ width: "16%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>CDR 1 (Mild)</span>
                      <span className="font-medium">28 subjects</span>
                    </div>
                    <div className="h-3 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-orange-500 rounded-full" style={{ width: "6%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>CDR 2 (Moderate)</span>
                      <span className="font-medium">2 subjects</span>
                    </div>
                    <div className="h-3 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-red-500 rounded-full" style={{ width: "0.5%" }} />
                    </div>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-4">
                  * 20 subjects without CDR labels excluded from classification
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* MRI Samples Tab */}
        <TabsContent value="mri" className="mt-6">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <FileImage className="h-5 w-5 text-blue-500" />
                  Real MRI Scan Samples
                </CardTitle>
                <CardDescription>
                  Sagittal T1-weighted MRI slices from actual OASIS-1 subjects (anonymized)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  {/* Normal Controls */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-green-500" />
                      Normal Controls (CDR = 0)
                    </h4>
                    <div className="grid grid-cols-3 gap-4">
                      {mriSamples.normal.map((sample, i) => (
                        <div key={i} className="group relative">
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border-2 border-green-500/30 hover:border-green-500 transition-colors">
                            <Image
                              src={sample.src}
                              alt={sample.label}
                              width={200}
                              height={200}
                              className="w-full h-full object-contain"
                            />
                          </div>
                          <div className="mt-2 text-center">
                            <div className="text-xs font-medium">{sample.label}</div>
                            <div className="text-xs text-muted-foreground">Age {sample.age} • CDR {sample.cdr}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Very Mild Dementia */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-yellow-500" />
                      Very Mild Dementia (CDR = 0.5)
                    </h4>
                    <div className="grid grid-cols-3 gap-4">
                      {mriSamples.mci.map((sample, i) => (
                        <div key={i} className="group relative">
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border-2 border-yellow-500/30 hover:border-yellow-500 transition-colors">
                            <Image
                              src={sample.src}
                              alt={sample.label}
                              width={200}
                              height={200}
                              className="w-full h-full object-contain"
                            />
                          </div>
                          <div className="mt-2 text-center">
                            <div className="text-xs font-medium">{sample.label}</div>
                            <div className="text-xs text-muted-foreground">Age {sample.age} • CDR {sample.cdr}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Mild Dementia */}
                  <div>
                    <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-orange-500" />
                      Mild Dementia (CDR = 1.0)
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {mriSamples.dementia.map((sample, i) => (
                        <div key={i} className="group relative">
                          <div className="aspect-square bg-black rounded-lg overflow-hidden border-2 border-orange-500/30 hover:border-orange-500 transition-colors">
                            <Image
                              src={sample.src}
                              alt={sample.label}
                              width={200}
                              height={200}
                              className="w-full h-full object-contain"
                            />
                          </div>
                          <div className="mt-2 text-center">
                            <div className="text-xs font-medium">{sample.label}</div>
                            <div className="text-xs text-muted-foreground">Age {sample.age} • CDR {sample.cdr}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <Alert className="mt-6 text-xs">
                  <strong>Note:</strong> Visual differences between groups may not be apparent to untrained observers. 
                  Deep learning models detect subtle patterns in tissue density, ventricle size, and cortical thickness 
                  that correlate with cognitive decline.
                </Alert>
              </CardContent>
            </Card>

            {/* MRI Processing Info */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">MRI Preprocessing Pipeline</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="p-3 rounded-lg bg-muted/50 text-center">
                    <div className="text-2xl mb-1">🧲</div>
                    <div className="font-medium">1.5T Scanner</div>
                    <div className="text-xs text-muted-foreground">Siemens Vision</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 text-center">
                    <div className="text-2xl mb-1">📐</div>
                    <div className="font-medium">256×256×128</div>
                    <div className="text-xs text-muted-foreground">Resolution</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 text-center">
                    <div className="text-2xl mb-1">🎯</div>
                    <div className="font-medium">1×1×1.25mm</div>
                    <div className="text-xs text-muted-foreground">Voxel Size</div>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50 text-center">
                    <div className="text-2xl mb-1">⚡</div>
                    <div className="font-medium">MPRAGE</div>
                    <div className="text-xs text-muted-foreground">Sequence</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Features Tab */}
        <TabsContent value="features" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-500" />
                Available Features
              </CardTitle>
              <CardDescription>
                Clinical, demographic, and anatomical features extracted from each subject
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Description</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {features.map((f) => (
                    <TableRow key={f.name}>
                      <TableCell className="font-mono font-medium">{f.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={
                          f.type === "Label" ? "border-red-500 text-red-600" :
                          f.type === "Clinical" ? "border-blue-500 text-blue-600" :
                          f.type === "Anatomical" ? "border-green-500 text-green-600" :
                          "border-orange-500 text-orange-600"
                        }>
                          {f.type}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">{f.description}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-blue-500/10 to-purple-500/10 border">
                <h4 className="font-semibold mb-2">🧠 CNN Features (ResNet18)</h4>
                <p className="text-sm text-muted-foreground">
                  In addition to tabular features, we extract <strong className="text-foreground">512-dimensional 
                  deep features</strong> from each MRI scan using a pretrained ResNet18 CNN. These capture complex 
                  spatial patterns not visible in handcrafted features.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Subjects Tab */}
        <TabsContent value="subjects" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Users className="h-5 w-5 text-orange-500" />
                Sample Subjects
              </CardTitle>
              <CardDescription>
                Representative examples from each diagnostic category
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Subject ID</TableHead>
                    <TableHead>Age</TableHead>
                    <TableHead>Gender</TableHead>
                    <TableHead>CDR</TableHead>
                    <TableHead>MMSE</TableHead>
                    <TableHead>nWBV</TableHead>
                    <TableHead>Diagnosis</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sampleSubjects.map((s) => (
                    <TableRow key={s.id}>
                      <TableCell className="font-mono text-xs">{s.id}</TableCell>
                      <TableCell>{s.age}</TableCell>
                      <TableCell>{s.gender}</TableCell>
                      <TableCell>
                        <Badge variant={s.cdr === 0 ? "default" : s.cdr === 0.5 ? "secondary" : "destructive"}>
                          {s.cdr}
                        </Badge>
                      </TableCell>
                      <TableCell>{s.mmse}</TableCell>
                      <TableCell>{s.nwbv.toFixed(3)}</TableCell>
                      <TableCell className="text-sm">{s.diagnosis}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg border bg-green-500/5 border-green-500/20">
                  <div className="text-sm font-medium text-green-600 mb-1">Normal (CDR 0)</div>
                  <div className="text-2xl font-bold">135</div>
                  <div className="text-xs text-muted-foreground">Used for training</div>
                </div>
                <div className="p-4 rounded-lg border bg-yellow-500/5 border-yellow-500/20">
                  <div className="text-sm font-medium text-yellow-600 mb-1">Very Mild (CDR 0.5)</div>
                  <div className="text-2xl font-bold">70</div>
                  <div className="text-xs text-muted-foreground">Target detection class</div>
                </div>
                <div className="p-4 rounded-lg border bg-orange-500/5 border-orange-500/20">
                  <div className="text-sm font-medium text-orange-600 mb-1">Mild+ (CDR ≥1)</div>
                  <div className="text-2xl font-bold">30</div>
                  <div className="text-xs text-muted-foreground">Excluded (too few)</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Alert className="text-xs">
        This portal summarizes the OASIS-1 research dataset. All images and data are from the publicly 
        available OASIS project. Results are provided for transparency and methodological evaluation 
        only and are not intended for clinical use.
      </Alert>
    </div>
  )
}


