"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

const links = [
  { href: "/roadmap", label: "Journey" },
  { href: "/dataset", label: "OASIS" },
  { href: "/adni", label: "ADNI" },
  { href: "/pipeline", label: "Pipeline" },
  { href: "/results", label: "Results" },
  { href: "/interpretability", label: "Visualizations" },
  { href: "/documentation", label: "Docs" },
]

export function MainNav() {
  const pathname = usePathname()

  return (
    <nav className="flex items-center gap-2 text-sm">
      {links.map((link) => {
        const active =
          pathname === link.href ||
          (link.href === "/dataset" && pathname === "/")
        return (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
              "rounded-md px-3 py-1.5 text-sm transition-colors text-muted-foreground hover:text-foreground hover:bg-muted",
              active && "bg-muted text-foreground"
            )}
          >
            {link.label}
          </Link>
        )
      })}
    </nav>
  )
}

