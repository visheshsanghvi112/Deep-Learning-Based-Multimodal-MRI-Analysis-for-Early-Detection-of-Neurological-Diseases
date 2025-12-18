"use client"

import { memo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { motion } from "framer-motion"
import { TrendingUp, Users, Target, Activity } from "lucide-react"

interface QuickStat {
  label: string
  value: string
  icon: React.ReactNode
  color: string
}

const stats: QuickStat[] = [
  {
    label: "Subjects",
    value: "205",
    icon: <Users className="h-4 w-4" />,
    color: "text-blue-500"
  },
  {
    label: "MRI-Only",
    value: "0.78",
    icon: <Target className="h-4 w-4" />,
    color: "text-emerald-500"
  },
  {
    label: "Late Fusion",
    value: "0.80",
    icon: <TrendingUp className="h-4 w-4" />,
    color: "text-purple-500"
  },
  {
    label: "Attention",
    value: "0.79",
    icon: <Activity className="h-4 w-4" />,
    color: "text-orange-500"
  }
]

const QuickStatCard = memo(function QuickStatCard({ 
  stat, index 
}: { 
  stat: QuickStat
  index: number 
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      whileHover={{ scale: 1.05 }}
    >
      <Card className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-colors">
        <CardContent className="flex items-center gap-4 p-4">
          <div className={`p-2 rounded-lg bg-muted/50 ${stat.color}`}>
            {stat.icon}
          </div>
          <div className="flex-1">
            <div className="text-2xl font-bold">{stat.value}</div>
            <div className="text-xs text-muted-foreground">{stat.label}</div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
})

export function QuickStats() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {stats.map((stat, index) => (
        <QuickStatCard key={stat.label} stat={stat} index={index} />
      ))}
    </div>
  )
}

