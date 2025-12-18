"use client"

import { useEffect, useState } from "react"
import { Moon, Sun } from "lucide-react"
import { cn } from "@/lib/utils"

const STORAGE_KEY = "neuroscope-theme"

export function ThemeToggle() {
  const [mounted, setMounted] = useState(false)
  const [dark, setDark] = useState(false)

  useEffect(() => {
    setMounted(true)
    if (typeof window === "undefined") return
    const stored = window.localStorage.getItem(STORAGE_KEY)
    const prefersDark = window.matchMedia?.(
      "(prefers-color-scheme: dark)",
    ).matches
    const isDark = stored === "dark" || (!stored && prefersDark)
    setDark(isDark)
    document.documentElement.classList.toggle("dark", isDark)
  }, [])

  if (!mounted) {
    return (
      <button
        className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border bg-card text-muted-foreground"
        aria-label="Toggle theme"
      />
    )
  }

  const toggle = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
    window.localStorage.setItem(STORAGE_KEY, next ? "dark" : "light")
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded-full border border-border bg-card text-muted-foreground transition-colors hover:text-foreground",
      )}
      aria-label="Toggle dark mode"
    >
      {dark ? (
        <Moon className="h-4 w-4" />
      ) : (
        <Sun className="h-4 w-4" />
      )}
    </button>
  )
}


