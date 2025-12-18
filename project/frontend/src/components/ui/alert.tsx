import * as React from "react"

import { cn } from "@/lib/utils"

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "destructive"
}

export function Alert({
  className,
  variant = "default",
  ...props
}: AlertProps) {
  return (
    <div
      role="alert"
      className={cn(
        "relative w-full rounded-lg border px-4 py-3 text-sm",
        variant === "default" &&
          "border-border bg-muted text-foreground",
        variant === "destructive" &&
          "border-destructive/50 bg-destructive/10 text-destructive",
        className,
      )}
      {...props}
    />
  )
}


