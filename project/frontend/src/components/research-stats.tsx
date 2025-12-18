"use client"

import { memo } from "react"
import { MetricsCard } from "./metrics-card"
import { Brain, Users, Target, Layers } from "lucide-react"

export const ResearchStats = memo(function ResearchStats() {
  return (
    <div className="grid gap-4 md:grid-cols-4">
      <MetricsCard
        title="Subjects"
        description="OASIS-1 cohort (all processed)"
        value="436"
        icon={<Users className="h-4 w-4" />}
        trend="neutral"
        detailText="Cross-sectional structural MRI with CNN features extracted for all subjects."
      />
      <MetricsCard
        title="MRI-Only"
        description="ResNet18 CNN embeddings (512-dim)"
        value="0.78"
        icon={<Target className="h-4 w-4" />}
        trend="neutral"
        detailText="Pure imaging biomarker for early dementia detection without clinical data."
      />
      <MetricsCard
        title="Late Fusion"
        description="MRI + Clinical (no MMSE)"
        value="0.80"
        icon={<Layers className="h-4 w-4" />}
        trend="up"
        detailText="Best multimodal fusion combining MRI and demographic features."
      />
      <MetricsCard
        title="Attention"
        description="Gated attention fusion"
        value="0.79"
        icon={<Brain className="h-4 w-4" />}
        trend="up"
        detailText="Attention-based fusion with learned modality weighting."
      />
    </div>
  )
})

