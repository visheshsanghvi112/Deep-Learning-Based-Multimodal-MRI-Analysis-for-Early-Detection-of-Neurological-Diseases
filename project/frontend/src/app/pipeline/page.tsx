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
          <Badge variant="outline">Multi-Dataset Robustness</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          The NeuroScope pipeline converts structural T1-weighted MRI and clinical
          variables into subject-level representations of neurodegenerative
          change. Our methodology is validated across OASIS-1 and ADNI-1
          using standardized feature extraction to ensure cross-dataset compatibility.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">1. Harmonization</CardTitle>
            <CardDescription>Preprocessing & site-alignment</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• Bias field correction & intensity normalization</p>
            <p>• Skull stripping & template-space (MNI) alignment</p>
            <p>• Resolution matching across multi-site datasets</p>
            <p>• Handling label definition shifts (CDR vs. DX_bl)</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">2. Feature Extraction</CardTitle>
            <CardDescription>ResNet18 + Anatomical</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• 512D MRI embeddings via pre-trained ResNet18</p>
            <p>• Regional brain volumes (Hippocampus, Ventricles)</p>
            <p>• Global atrophy & Intracranial Volume (eTIV)</p>
            <p>• Clinical covariates (Age, Sex, Education)</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">3. Multimodal Fusion</CardTitle>
            <CardDescription>Late & Attention Fusion</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm text-muted-foreground">
            <p>• Modality-specific MLP encoders</p>
            <p>• Late fusion (logit averaging/concatenation)</p>
            <p>• Cross-modal attention mechanisms</p>
            <p>• Explicit age modeling as a confounder</p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Experimental Consistency</CardTitle>
            <CardDescription>Ensuring publication-grade results</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              To ensure results are truly robust, we utilize the exact same
              architectures and hyperparameters for both in-dataset training
              and cross-dataset evaluation.
            </p>
            <p>
              MRI-Only models serve as the baseline, with Multimodal (Late/Attention)
              fusion architectures evaluating the additive value of clinical signal
              in predicting early cognitive decline.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Robustness philosophy</CardTitle>
            <CardDescription>Beyond internal validation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              Deep learning models often overfit to center-specific acquisition
              protocols. By testing our OASIS-trained models on ADNI (and vice-versa),
              we move beyond simple AUC reporting toward verifying the
              clinical stability of the learned brain signatures.
            </p>
            <p>
              This rigorous validation approach is critical for assessing
              the feasibility of deploying these models in diverse real-world
              clinical settings.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}


