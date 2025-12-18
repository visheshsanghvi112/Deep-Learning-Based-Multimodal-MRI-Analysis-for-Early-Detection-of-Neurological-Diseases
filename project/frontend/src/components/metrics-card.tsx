"use client"

import { memo } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"

interface MetricsCardProps {
  title: string
  description: string
  value: string | number
  badge?: {
    label: string
    variant?: "default" | "secondary" | "destructive" | "outline"
  }
  trend?: "up" | "down" | "neutral"
  icon?: React.ReactNode
  detailText?: string
}

export const MetricsCard = memo(function MetricsCard({ 
  title, 
  description, 
  value, 
  badge,
  trend,
  icon,
  detailText
}: MetricsCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -2 }}
    >
      <Card className="bg-card/80 backdrop-blur hover:shadow-lg transition-all duration-300 border-border/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium">{title}</CardTitle>
            {icon && <div className="text-muted-foreground">{icon}</div>}
          </div>
          <CardDescription className="text-xs">{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-semibold">{value}</div>
            {badge && (
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                badge.variant === "secondary" 
                  ? "bg-muted text-muted-foreground"
                  : "bg-primary/10 text-primary"
              }`}>
                {badge.label}
              </span>
            )}
          </div>
          {trend && (
            <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
              {trend === "up" && <TrendingUp className="h-3 w-3 text-emerald-500" />}
              {trend === "down" && <TrendingDown className="h-3 w-3 text-red-500" />}
              {trend === "neutral" && <Minus className="h-3 w-3" />}
            </div>
          )}
          {detailText && (
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {detailText}
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
})

