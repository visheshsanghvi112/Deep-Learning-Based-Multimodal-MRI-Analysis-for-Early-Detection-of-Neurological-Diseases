"use client"

import { memo } from "react"
import { MetricsCard } from "./metrics-card"
import { Brain, Users, Target, Layers, Database, ArrowRightLeft } from "lucide-react"

export const ResearchStats = memo(function ResearchStats() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <MetricsCard
        title="OASIS-1"
        description="Single-site, high-quality"
        value="436"
        icon={<Users className="h-4 w-4" />}
        trend="neutral"
        detailText="CDR 0 vs 0.5. Late Fusion AUC 0.80."
      />
      <MetricsCard
        title="ADNI Level-1"
        description="Honest baseline (Age/Sex)"
        value="0.60"
        icon={<Database className="h-4 w-4" />}
        trend="neutral"
        detailText="Level-1: MRI + Age/Sex only."
      />
      <MetricsCard
        title="🎯 Level-MAX"
        description="Biomarker breakthrough"
        value="0.81"
        icon={<Target className="h-4 w-4" />}
        trend="up"
        detailText="+16.5% with CSF, APOE4, Volumetrics!"
      />
      <MetricsCard
        title="Cross-Dataset"
        description="Robustness validated"
        value="0.62"
        icon={<ArrowRightLeft className="h-4 w-4" />}
        trend="up"
        detailText="OASIS↔ADNI transfer shows MRI stability."
      />
      <MetricsCard
        title="Key Finding"
        description="Direction-dependent"
        value="✓"
        icon={<Brain className="h-4 w-4" />}
        trend="up"
        detailText="MRI robust; Multimodal varies by domain."
      />
    </div>
  )
})
