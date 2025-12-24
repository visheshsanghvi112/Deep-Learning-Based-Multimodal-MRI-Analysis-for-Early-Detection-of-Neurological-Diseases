"use client"

import { memo } from "react"
import Link from "next/link"
import { motion, Variants } from "framer-motion"
import { SpotlightCard } from "@/components/ui/spotlight-card"
import {
  Database,
  Workflow,
  BarChart3,
  Brain,
  ArrowRightLeft,
  ArrowUpRight
} from "lucide-react"

interface Feature {
  icon: React.ReactNode
  title: string
  description: string
  href: string
  items: string[]
  className?: string
  spotlight?: string
}

const features: Feature[] = [
  {
    icon: <Database className="h-6 w-6 text-blue-500" />,
    title: "OASIS-1 Analysis",
    description: "Single-site MRI + clinical anchors. The gold standard baseline.",
    href: "/dataset",
    items: ["436 Subjects", "CDR 0 vs 0.5", "512-dim ResNet18"],
    className: "md:col-span-2 md:row-span-2",
    spotlight: "rgba(59, 130, 246, 0.2)" // blue
  },
  {
    icon: <Database className="h-6 w-6 text-teal-500" />,
    title: "ADNI-1 Validation",
    description: "Multi-site robustness testing against real-world variance.",
    href: "/adni",
    items: ["629 Subjects", "MCI/AD Spectrum", "Universal Pipeline"],
    className: "md:col-span-1 md:row-span-2",
    spotlight: "rgba(20, 184, 166, 0.2)" // teal
  },
  {
    icon: <Workflow className="h-5 w-5" />,
    title: "Pipeline",
    description: "End-to-end processing",
    href: "/pipeline",
    items: ["ResNet18 Feature Extraction", "Multimodal Fusion"],
    className: "md:col-span-1",
    spotlight: "rgba(120, 120, 120, 0.15)"
  },
  {
    icon: <ArrowRightLeft className="h-5 w-5 text-purple-500" />,
    title: "Cross-Dataset",
    description: "Zero-shot transfer",
    href: "/results",
    items: ["OASIS→ADNI", "Label Shift"],
    className: "md:col-span-1",
    spotlight: "rgba(168, 85, 247, 0.2)" // purple
  },
  {
    icon: <Brain className="h-5 w-5" />,
    title: "Interpretability",
    description: "Latent space viz",
    href: "/interpretability",
    items: ["Attention Maps", "t-SNE Clusters"],
    className: "md:col-span-1",
    spotlight: "rgba(120, 120, 120, 0.15)"
  }
]

const container: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const item: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 50
    }
  }
}

const BentoCard = memo(function BentoCard({ feature }: { feature: Feature }) {
  return (
    <motion.div variants={item} className={feature.className}>
      <Link href={feature.href} className="block h-full cursor-pointer group/card">
        <SpotlightCard
          className="h-full border-border/40 bg-card/50 transition-colors hover:bg-card/80"
          spotlightColor={feature.spotlight}
        >
          <div className="flex h-full flex-col justify-between p-6">
            <div className="mb-4">
              <div className="mb-4 flex items-center justify-between">
                <div className="rounded-lg bg-background/80 p-2 shadow-sm backdrop-blur-sm ring-1 ring-border/50">
                  {feature.icon}
                </div>
                <ArrowUpRight className="h-5 w-5 text-muted-foreground opacity-0 transition-all duration-300 group-hover/card:opacity-100 group-hover/card:translate-x-1 group-hover/card:-translate-y-1" />
              </div>
              <h3 className="mb-1 text-lg font-semibold tracking-tight">
                {feature.title}
              </h3>
              <p className="text-sm text-muted-foreground">
                {feature.description}
              </p>
            </div>

            <div className="space-y-2">
              {feature.items.map((item, i) => (
                <div key={i} className="flex items-center text-xs text-muted-foreground/80">
                  <span className="mr-2 h-1 w-1 rounded-full bg-foreground/30" />
                  {item}
                </div>
              ))}
            </div>
          </div>
        </SpotlightCard>
      </Link>
    </motion.div>
  )
})

export function FeatureGrid() {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-100px" }}
      className="grid grid-cols-1 gap-4 md:grid-cols-3 md:auto-rows-[minmax(180px,auto)]"
    >
      {features.map((feature) => (
        <BentoCard key={feature.title} feature={feature} />
      ))}
    </motion.div>
  )
}
