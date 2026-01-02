"use client"

import { memo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { TrendingUp, Users, Target, Activity, ArrowRightLeft, Database } from "lucide-react"

interface QuickStat {
  label: string
  value: string
  subtext?: string
  icon: React.ReactNode
  color: string
}

const stats: QuickStat[] = [
  {
    label: "Total Subjects",
    value: "1,065",
    subtext: "OASIS + ADNI",
    icon: <Users className="h-4 w-4" />,
    color: "text-blue-500"
  },
  {
    label: "OASIS AUC",
    value: "0.80",
    subtext: "Late Fusion",
    icon: <Target className="h-4 w-4" />,
    color: "text-emerald-500"
  },
  {
    label: "🎯 Level-MAX",
    value: "0.81",
    subtext: "Bio-Profile",
    icon: <TrendingUp className="h-4 w-4" />,
    color: "text-emerald-500"
  },
  {
    label: "ADNI L1",
    value: "0.60",
    subtext: "Honest Baseline",
    icon: <Database className="h-4 w-4" />,
    color: "text-purple-500"
  },
  {
    label: "Cross-Dataset",
    value: "0.62",
    subtext: "Transfer AUC",
    icon: <ArrowRightLeft className="h-4 w-4" />,
    color: "text-orange-500"
  }
]

const QuickStatCard = memo(function QuickStatCard({
  stat
}: {
  stat: QuickStat
}) {
  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-colors">
      <CardContent className="flex items-center gap-4 p-4">
        <div className={`p-2 rounded-lg bg-muted/50 ${stat.color}`}>
          {stat.icon}
        </div>
        <div className="flex-1">
          <div className="text-2xl font-bold">{stat.value}</div>
          <div className="text-xs text-muted-foreground">{stat.label}</div>
          {stat.subtext && (
            <div className="text-[10px] text-muted-foreground/70">{stat.subtext}</div>
          )}
        </div>
      </CardContent>
    </Card>
  )
})

export function QuickStats() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {stats.map((stat) => (
        <QuickStatCard key={stat.label} stat={stat} />
      ))}
    </div>
  )
}
