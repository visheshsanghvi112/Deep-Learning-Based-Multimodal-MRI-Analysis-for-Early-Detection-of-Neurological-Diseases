"use client"

import { motion } from "framer-motion"
import { BrainVisualizationEnhanced } from "./brain-visualization-enhanced"

export function Hero3D() {
  return (
    <section className="relative overflow-hidden rounded-2xl border bg-card/50 backdrop-blur-sm p-6 md:p-8">
      <div className="relative grid gap-8 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] items-center">
        <div className="space-y-4 md:space-y-5">
          <motion.h1
            className="text-2xl md:text-4xl font-semibold tracking-tight"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            Latent Neurodegenerative Signatures
            <br />
            <span className="text-xl md:text-2xl font-medium text-muted-foreground">
              from Structural MRI
            </span>
          </motion.h1>
          
          <motion.p
            className="max-w-xl text-sm md:text-base text-muted-foreground leading-relaxed"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            A publication-grade, OASIS-1–only pipeline combining anatomical features
            and clinical anchors to learn stable MRI representations of early
            neurodegenerative change. Results are presented for research purposes
            only.
          </motion.p>
          
          <motion.div
            className="inline-flex items-center gap-2 rounded-full border bg-card/80 px-3 py-1 text-xs text-muted-foreground backdrop-blur"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            OASIS-1 Baseline (Anatomical + Clinical, no CNN) locked as primary result
          </motion.div>
        </div>

        <div className="relative h-56 md:h-72 lg:h-80">
          <div className="relative h-full w-full rounded-2xl overflow-hidden border border-border/50 bg-muted/30">
            <BrainVisualizationEnhanced />
          </div>
        </div>
      </div>
    </section>
  )
}


