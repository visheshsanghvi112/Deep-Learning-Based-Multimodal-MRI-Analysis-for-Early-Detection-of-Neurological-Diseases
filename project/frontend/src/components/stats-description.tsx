"use client"

interface StatsDescriptionProps {
  value: string | number
  description?: string
}

export function StatsDescription({ value, description }: StatsDescriptionProps) {
  if (!description) return <p className="mt-1 text-xs text-muted-foreground">{value}</p>
  
  return (
    <p className="mt-1 text-xs text-muted-foreground">
      {description}
    </p>
  )
}

