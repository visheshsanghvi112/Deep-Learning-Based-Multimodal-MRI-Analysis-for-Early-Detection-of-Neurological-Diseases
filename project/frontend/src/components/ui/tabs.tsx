"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

const TabsContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
} | null>(null)

export interface TabsProps {
  value: string
  onValueChange?: (value: string) => void
  defaultValue?: string
  children: React.ReactNode
  className?: string
}

export function Tabs({ value, onValueChange, defaultValue, className, children }: TabsProps) {
  // Support both controlled and uncontrolled (though we primarily need controlled for now)
  const [internalValue, setInternalValue] = React.useState(defaultValue || value)

  const handleValueChange = React.useCallback((newValue: string) => {
    setInternalValue(newValue)
    onValueChange?.(newValue)
  }, [onValueChange])

  // Sync prop value if controlled
  React.useEffect(() => {
    if (value !== undefined) {
      setInternalValue(value)
    }
  }, [value])

  return (
    <TabsContext.Provider value={{ value: internalValue, onValueChange: handleValueChange }}>
      <div className={cn("w-full", className)} data-value={internalValue}>
        {children}
      </div>
    </TabsContext.Provider>
  )
}

export interface TabsListProps
  extends React.HTMLAttributes<HTMLDivElement> { }

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
  value: string
}

export function TabsTrigger({
  className,
  value,
  isActive,
  ...props
}: TabsTriggerProps & { isActive?: boolean }) {
  const context = React.useContext(TabsContext)
  if (!context) throw new Error("TabsTrigger must be used within Tabs")

  return (
    <button
      onClick={() => context.onValueChange(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        context.value === value
          ? "bg-background text-foreground shadow"
          : "text-muted-foreground hover:bg-background/80",
        className,
      )}
      {...props}
    />
  )
}

export interface TabsContentProps
  extends React.HTMLAttributes<HTMLDivElement> {
  value?: string // Make it compatible with standard TabsContent props
}

export function TabsContent({ className, value, ...props }: TabsContentProps) {
  /* Note: In standard Radix this conditionally renders. 
     In the previous version it didn't take a value. 
     We should probably check context if value is provided. 
  */
  const context = React.useContext(TabsContext)
  if (value && context && context.value !== value) return null

  return (
    <div
      className={cn("mt-3 border-t pt-4", className)}
      {...props}
    />
  )
}
