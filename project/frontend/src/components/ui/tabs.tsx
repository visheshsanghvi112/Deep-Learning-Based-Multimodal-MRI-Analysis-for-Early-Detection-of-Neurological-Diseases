import * as React from "react"

import { cn } from "@/lib/utils"

export interface TabsProps {
  value: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
  className?: string
}

export function Tabs({ value, onValueChange, className, children }: TabsProps) {
  return (
    <div className={cn("w-full", className)} data-value={value}>
      {children}
    </div>
  )
}

export interface TabsListProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export function TabsList({ className, ...props }: TabsListProps) {
  return (
    <div
      className={cn(
        "inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground",
        className,
      )}
      {...props}
    />
  )
}

export interface TabsTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isActive?: boolean
}

export function TabsTrigger({
  className,
  isActive,
  ...props
}: TabsTriggerProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        isActive
          ? "bg-background text-foreground shadow"
          : "text-muted-foreground hover:bg-background/80",
        className,
      )}
      {...props}
    />
  )
}

export interface TabsContentProps
  extends React.HTMLAttributes<HTMLDivElement> {}

export function TabsContent({ className, ...props }: TabsContentProps) {
  return (
    <div
      className={cn("mt-3 border-t pt-4", className)}
      {...props}
    />
  )
}


