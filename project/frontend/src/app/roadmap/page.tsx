import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export default function RoadmapPage() {
  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            Roadmap &amp; Future Work
          </h2>
          <Badge variant="outline">Planned work</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          The NeuroScope research project has evolved into a comprehensive multi-dataset study.
          We have successfully integrated OASIS-1 and ADNI-1 datasets, enabling robust
          cross-dataset validation of our multimodal fusion architectures across 1,065 subjects.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card className="border-green-500/20 bg-green-500/5">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              Completed: Data Engineering
            </CardTitle>
            <CardDescription>Multi-dataset foundation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>• OASIS-1 & ADNI-1 full feature extraction (1,065 subjects)</p>
            <p>• ResNet18-based MRI embedding pipeline for all scans</p>
            <p>• Clinical feature harmonization (Age, Gender, MMSE, CDR)</p>
            <p>• Unified preprocessing for cross-center compatibility</p>
          </CardContent>
        </Card>

        <Card className="border-green-500/20 bg-green-500/5">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              Completed: Model Validation
            </CardTitle>
            <CardDescription>Robustness & Generalization</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>• Multi-modal fusion training (Late & Attention Fusion)</p>
            <p>• Cross-dataset transfer analysis (OASIS ↔ ADNI)</p>
            <p>• Identification of label definition shifts and impact on accuracy</p>
            <p>• Evaluation of modality-specific robustness in transfer settings</p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">In Progress: Advanced Fusion</CardTitle>
            <CardDescription>Beyond simple attention</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • Implementing gating mechanisms to better handle unreliable
              modalities during cross-center inference.
            </p>
            <p>
              • Developing specialized "Domain Adapters" to mitigate the
              label shift observed between OASIS and ADNI cohorts.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Future: Longitudinal Tracking</CardTitle>
            <CardDescription>Disease trajectory modeling</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • Extending the pipeline to handle longitudinal ADNI visits (M06, M12, M24).
            </p>
            <p>
              • Predicting time-to-conversion from Mild Cognitive Impairment (MCI)
              to Alzheimer's Disease (AD).
            </p>
            <p>
              • Integrating fluid biomarkers (CSF Aβ/Tau) with MRI signatures.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}


