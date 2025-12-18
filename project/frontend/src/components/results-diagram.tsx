"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

const MRI_ONLY_AUC = 0.78
const LATE_FUSION_AUC = 0.80
const ATTENTION_AUC = 0.79

export function ResultsDiagram() {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setStep((s) => (s + 1) % 3)
    }, 2200)
    return () => clearInterval(id)
  }, [])

  return (
    <Card className="border bg-card/70 backdrop-blur">
      <CardHeader>
        <CardTitle className="text-sm">Multimodal Fusion Performance</CardTitle>
        <CardDescription>
          Comparison of fusion strategies for early dementia detection (without MMSE)
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 md:flex-row md:items-end">
        <div className="flex flex-1 items-end gap-4">
          <MetricBar
            label="MRI Only"
            value={MRI_ONLY_AUC}
            highlight={step === 0}
          />
          <MetricBar
            label="Late Fusion"
            value={LATE_FUSION_AUC}
            highlight={step === 1}
          />
          <MetricBar
            label="Attention"
            value={ATTENTION_AUC}
            highlight={step === 2}
          />
        </div>
        <div className="flex-1 space-y-2 text-xs text-muted-foreground">
          <p className={step === 0 ? "opacity-100" : "opacity-50"}>
            • MRI-only (ResNet18 512-dim embeddings) achieves AUC of{" "}
            <span className="font-semibold">{MRI_ONLY_AUC.toFixed(2)}</span> for
            CDR=0 vs CDR=0.5 classification.
          </p>
          <p className={step === 1 ? "opacity-100" : "opacity-50"}>
            • Late Fusion (MRI + Clinical) reaches AUC of{" "}
            <span className="font-semibold">{LATE_FUSION_AUC.toFixed(2)}</span>{" "}
            combining imaging and demographic features.
          </p>
          <p className={step === 2 ? "opacity-100" : "opacity-50"}>
            • Attention Fusion achieves AUC of{" "}
            <span className="font-semibold">{ATTENTION_AUC.toFixed(2)}</span>{" "}
            with learned modality weighting.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

interface MetricBarProps {
  label: string
  value: number
  highlight?: boolean
  exploratory?: boolean
}

function MetricBar({ label, value, highlight, exploratory }: MetricBarProps) {
  const height = value * 100

  return (
    <div className="flex min-w-[80px] flex-col items-center gap-2">
      <div className="relative h-32 w-8 overflow-hidden rounded-full border bg-muted/60">
        <motion.div
          className="absolute bottom-0 left-0 right-0 rounded-t-full bg-primary"
          initial={{ height: 0 }}
          animate={{ height: `${height}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 16 }}
        />
      </div>
      <div className="flex flex-col items-center gap-0.5 text-[10px] text-muted-foreground">
        <span className={highlight ? "font-semibold text-foreground" : ""}>
          {label}
        </span>
        <span>{value.toFixed(2)} AUC</span>
        {exploratory && (
          <span className="rounded-full border px-1 py-0.5 text-[9px]">
            Exploratory
          </span>
        )}
      </div>
    </div>
  )
}


