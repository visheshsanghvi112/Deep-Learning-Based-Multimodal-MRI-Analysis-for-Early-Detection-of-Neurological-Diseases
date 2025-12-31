"use client"

import { motion } from "framer-motion"
import { BrainVisualizationEnhanced } from "./brain-visualization-enhanced"
import Image from "next/image"

export function Hero3D() {
  return (
    <section className="relative overflow-hidden rounded-2xl border bg-card/10 backdrop-blur-sm border-white/10 shadow-lg">
      <div className="absolute inset-0 bg-background/20" /> {/* Subtle tint */}

      <div className="relative p-4 md:p-10 grid gap-6 md:gap-8 grid-cols-1 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] items-center">
        <div className="space-y-3 md:space-y-6 order-2 md:order-1">
          <motion.h1
            className="text-xl sm:text-2xl md:text-5xl font-bold tracking-tight text-foreground drop-shadow-md leading-tight"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            Latent Neurodegenerative Signatures
            <br />
            <span className="text-base sm:text-xl md:text-3xl font-medium text-muted-foreground/90">
              from Structural MRI
            </span>
          </motion.h1>

          <motion.p
            className="max-w-xl text-xs sm:text-sm md:text-lg text-foreground/80 leading-relaxed font-medium"
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
            className="inline-flex items-center gap-2 rounded-full border border-border/50 bg-background/50 px-3 md:px-4 py-1 md:py-1.5 text-[10px] md:text-xs font-medium backdrop-blur-md shadow-sm"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <span className="h-1.5 w-1.5 md:h-2 md:w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
            <span className="hidden sm:inline">OASIS-1 Baseline (Anatomical + Clinical) locked as primary result</span>
            <span className="sm:hidden">OASIS-1 Primary Result</span>
          </motion.div>
        </div>

        {/* 3D Brain Viz - Floating in foreground - ALWAYS VISIBLE */}
        <div className="relative h-64 sm:h-72 md:h-80 lg:h-96 transform transition-transform hover:scale-105 duration-500 order-1 md:order-2">
          <div className="relative h-full w-full rounded-2xl overflow-hidden border border-white/10 shadow-2xl bg-black/20 backdrop-blur-sm">
            <BrainVisualizationEnhanced />
          </div>
        </div>
      </div>
    </section>
  )
}
