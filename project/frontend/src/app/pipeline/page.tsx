import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export default function PipelinePage() {
  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            Pipeline: From MRI to Latent Signatures
          </h2>
          <Badge variant="outline">OASIS-1 only</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          The pipeline converts structural T1-weighted MRI and clinical
          variables into subject-level representations of neurodegenerative
          change, using a combination of explicit anatomical features and
          learned embeddings.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">1. MRI preprocessing</CardTitle>
            <CardDescription>Image-level operations</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• Bias field correction (intensity non-uniformity)</p>
            <p>• Skull stripping (brain extraction)</p>
            <p>• Spatial normalization to a common template</p>
            <p>• Tissue segmentation (GM / WM / CSF)</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">2. Feature extraction</CardTitle>
            <CardDescription>214+ anatomical features</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• Global atrophy and brain volume measures</p>
            <p>• Regional volumes (e.g., hippocampus, ventricles)</p>
            <p>• Asymmetry indices and intensity statistics</p>
            <p>• Clinical covariates (age, gender, education)</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">3. Multimodal model</CardTitle>
            <CardDescription>
              Hybrid architecture (baseline + prototype)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• Anatomical encoder (MLP)</p>
            <p>• Clinical encoder (MLP, age modeled explicitly)</p>
            <p>• MRI encoder (CNN embeddings, prototype only)</p>
            <p>• Attention-based fusion and multi-task heads</p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Baseline vs prototype</CardTitle>
            <CardDescription>
              Same architecture, different inputs
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              <span className="font-medium">
                OASIS-1 Baseline (Anatomical + Clinical, no CNN):
              </span>{" "}
              uses only explicit anatomical features and clinical covariates.
              This is the primary, publication-safe configuration.
            </p>
            <p>
              <span className="font-medium">
                Prototype Multimodal (Partial CNN – Exploratory):
              </span>{" "}
              augments the same architecture with early CNN-based MRI
              embeddings. These embeddings are currently available for a very
              small subset of subjects and are therefore treated as exploratory.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Why age is explicit</CardTitle>
            <CardDescription>Controlling a key confounder</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              Age strongly correlates with both atrophy and cognitive scores.
              In this pipeline, age is modeled explicitly as a covariate rather
              than left for the model to exploit implicitly.
            </p>
            <p>
              This makes it easier to interpret learned representations and to
              evaluate whether the model is relying on age alone or capturing
              additional neurodegenerative signal beyond healthy aging.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}


