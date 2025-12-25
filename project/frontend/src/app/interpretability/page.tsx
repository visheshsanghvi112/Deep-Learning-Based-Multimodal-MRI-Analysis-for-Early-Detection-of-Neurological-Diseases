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
import { Alert } from "@/components/ui/alert"

function InterpretabilityImage({
  src,
  alt,
  description
}: {
  src: string
  alt: string
  description: string
}) {
  const [imageError, setImageError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  if (imageError) {
    return (
      <div className="relative h-64 w-full overflow-hidden rounded-md border bg-muted/50 flex items-center justify-center">
        <div className="text-center space-y-2 p-4">
          <p className="text-sm text-muted-foreground">Image not available</p>
          <p className="text-xs text-muted-foreground/70">
            {description}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative h-64 w-full overflow-hidden rounded-md border bg-muted">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
          <div className="text-sm text-muted-foreground">Loading image...</div>
        </div>
      )}
      <Image
        src={src}
        alt={alt}
        fill
        style={{ objectFit: "contain" }}
        onError={() => setImageError(true)}
        onLoad={() => setIsLoading(false)}
        unoptimized
      />
    </div>
  )
}

export default function InterpretabilityPage() {
  const attentionUrl = "/static/attention_weights.png"
  const tsneUrl = "/static/embeddings_tsne.png"

  return (
    <div className="flex w-full flex-col gap-8">
      <section className="space-y-2">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold tracking-tight">
            Interpretability: Corroborated Insights
          </h2>
          <Badge variant="outline">Multi-Dataset analysis</Badge>
        </div>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Interpretability analyses summarize how anatomical, clinical, and
          ResNet18 MRI features contribute to predictions. Findings are
          corroborated across OASIS-1 and ADNI-1 datasets to identify
          universal signatures of early neurodegeneration.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Attention across modalities</CardTitle>
            <CardDescription>Example attention map</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <InterpretabilityImage
              src={attentionUrl}
              alt="Average attention across modalities"
              description="Attention weights visualization showing how the model weights different modalities (anatomical, clinical, MRI embeddings)."
            />
            <p>
              Average attention weights show how strongly the model relies on
              anatomical features, clinical covariates, and (prototype) MRI
              embeddings.
            </p>
            <p>
              In the current baseline configuration, attention is concentrated
              on age and global atrophy measures; in prototype settings,
              attention shifts partially toward MRI embeddings.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">t-SNE embeddings</CardTitle>
            <CardDescription>Latent representation space</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <InterpretabilityImage
              src={tsneUrl}
              alt="t-SNE visualization of subject embeddings"
              description="t-SNE visualization showing how subjects are organized in the learned embedding space, with gradients along age and CDR."
            />
            <p>
              t-SNE visualizations of subject-level embeddings reveal smooth
              gradients with respect to age and CDR, suggesting that the model
              captures both healthy aging and early neurodegenerative change.
            </p>
            <p>
              Clusters are not used for clinical diagnosis but help validate
              that the learned representations align with known disease
              progression patterns.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Feature importance</CardTitle>
            <CardDescription>
              Anatomical and clinical drivers (baseline)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              Top-ranked features include normalized whole brain volume (NWBV),
              estimated total intracranial volume (eTIV), hippocampal volumes,
              and age.
            </p>
            <p>
              Many of these features are partially age-driven, which is
              expected; follow-up analyses explicitly quantify age
              confounding to separate healthy aging from pathological
              degeneration.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Interpretability philosophy</CardTitle>
            <CardDescription>Anchoring in neurobiology</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              The goal is not to produce "black box" predictions, but to learn
              MRI-derived signatures that can be related back to known
              anatomical substrates of early cognitive decline.
            </p>
            <p>
              Explicit anatomical features and attention mechanisms provide
              handles for linking model behavior to brain regions and clinical
              anchors, supporting a publication-grade analysis.
            </p>
          </CardContent>
        </Card>
      </section>

      <Alert className="text-xs">
        Interpretability visualizations show attention weights and learned embeddings
        from the fusion models. These visualizations are for research purposes only and
        demonstrate model behavior on OASIS-1 and ADNI-1 datasets.
      </Alert>
    </div>
  )
}
