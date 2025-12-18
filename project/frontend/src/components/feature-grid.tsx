"use client"

import { memo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { 
  Database, 
  Workflow, 
  BarChart3, 
  Brain, 
  FileText, 
  Route 
} from "lucide-react"

interface Feature {
  icon: React.ReactNode
  title: string
  description: string
  items: string[]
}

const features: Feature[] = [
  {
    icon: <Database className="h-5 w-5" />,
    title: "Dataset Characteristics",
    description: "OASIS-1 MRI + clinical anchors",
    items: [
      "436 cross-sectional structural MRI scans",
      "Clinical anchors (CDR, MMSE, demographics)",
      "214+ anatomical features extracted",
      "CNN embeddings for multimodal analysis"
    ]
  },
  {
    icon: <Workflow className="h-5 w-5" />,
    title: "Processing Pipeline",
    description: "End-to-end research workflow",
    items: [
      "Preprocessing & normalization",
      "Feature extraction & engineering",
      "Multimodal fusion architecture",
      "Multi-task learning framework"
    ]
  },
  {
    icon: <BarChart3 className="h-5 w-5" />,
    title: "Research Results",
    description: "Baseline and exploratory metrics",
    items: [
      "OASIS-1 baseline (publication-safe)",
      "Prototype multimodal results",
      "CDR prediction performance",
      "Binary classification metrics"
    ]
  },
  {
    icon: <Brain className="h-5 w-5" />,
    title: "Interpretability",
    description: "Model transparency and analysis",
    items: [
      "Attention weight visualizations",
      "Feature importance analysis",
      "Embedding space exploration",
      "Regional contribution mapping"
    ]
  },
  {
    icon: <FileText className="h-5 w-5" />,
    title: "Documentation",
    description: "Comprehensive research documentation",
    items: [
      "Methodology and architecture",
      "Training procedures",
      "Evaluation protocols",
      "Research limitations"
    ]
  },
  {
    icon: <Route className="h-5 w-5" />,
    title: "Roadmap",
    description: "Future research directions",
    items: [
      "ADNI dataset integration",
      "Cross-dataset validation",
      "Enhanced feature extraction",
      "Clinical validation studies"
    ]
  }
]

const FeatureCard = memo(function FeatureCard({ 
  feature, index 
}: { 
  feature: Feature
  index: number 
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
    >
      <Card className="h-full hover:shadow-lg transition-all duration-300 border-border/50">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-muted">
              {feature.icon}
            </div>
            <CardTitle className="text-base">{feature.title}</CardTitle>
          </div>
          <CardDescription>{feature.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            {feature.items.map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-muted-foreground/60 mt-1.5">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </motion.div>
  )
})

export function FeatureGrid() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {features.map((feature, index) => (
        <FeatureCard key={feature.title} feature={feature} index={index} />
      ))}
    </div>
  )
}

