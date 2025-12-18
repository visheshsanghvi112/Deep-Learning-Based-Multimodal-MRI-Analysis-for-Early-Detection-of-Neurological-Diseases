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
          This portal currently presents only OASIS-1 results. Future work
          focuses on completing CNN embeddings for OASIS-1, integrating ADNI,
          and performing cross-dataset validation of the multimodal pipeline.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Completed</CardTitle>
            <CardDescription>Locked scope</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>• OASIS-1 feature extraction (anatomical + clinical)</p>
            <p>• OASIS-1 Baseline model (Anatomical + Clinical, no CNN)</p>
            <p>• Prototype multimodal architecture with partial CNN embeddings</p>
            <p>• Initial interpretability analyses and reporting</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">In progress</CardTitle>
            <CardDescription>OASIS-1 multimodal completion</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • Fixing the local PyTorch environment (c10.dll) to enable robust
              extraction of CNN embeddings for the majority of OASIS-1 subjects.
            </p>
            <p>
              • Running a single, full multimodal retrain and evaluation once
              embeddings are available, using the same architecture shown here.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Planned: ADNI integration</CardTitle>
            <CardDescription>Future extension, not yet included</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • Apply the same preprocessing and feature extraction pipeline to
              ADNI structural MRI and clinical variables.
            </p>
            <p>
              • Harmonize OASIS-1 and ADNI features and labels for joint
              modeling.
            </p>
            <p>
              No ADNI results are currently shown in this portal; ADNI appears
              here only as future work.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">
              Planned: Cross-dataset validation
            </CardTitle>
            <CardDescription>Generalization &amp; robustness</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              • Evaluate transfer performance OASIS→ADNI and ADNI→OASIS once
              both datasets are fully processed.
            </p>
            <p>
              • Perform age-stratified metrics and confound analysis on the
              final multimodal model to assess robustness across cohorts and
              acquisition protocols.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}


