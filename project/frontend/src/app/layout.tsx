import type { Metadata } from "next"
import Link from "next/link"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Badge } from "@/components/ui/badge"
import { MainNav } from "@/components/main-nav"
import { ThemeToggle } from "@/components/theme-toggle"
import { OptimizedBackground } from "@/components/optimized-background"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "NeuroScope – OASIS-1 MRI Research Portal",
  description:
    "Research-grade visualization portal for OASIS-1 structural MRI and clinical features. Not for clinical use.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground relative`}
      >
        <OptimizedBackground />
        <div className="flex min-h-screen flex-col relative z-0">
          <header className="border-b bg-card/80 backdrop-blur">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
              <div className="flex items-center gap-3">
                <Link href="/" className="font-semibold tracking-tight text-lg hover:opacity-80 transition-opacity">
                  NeuroScope
                </Link>
                <Badge variant="secondary" className="text-xs">
                  Research
                </Badge>
              </div>
              <div className="flex items-center gap-4">
                <MainNav />
                <ThemeToggle />
              </div>
            </div>
            <div className="border-t bg-muted/40">
              <div className="mx-auto max-w-6xl px-6 py-2 text-xs text-muted-foreground">
                This portal presents research models trained on the OASIS-1
                dataset only. Results are{" "}
                <span className="font-semibold">
                  not intended for clinical use.
                </span>
              </div>
            </div>
          </header>
          <main className="mx-auto flex w-full max-w-6xl flex-1 px-6 py-8">
            {children}
          </main>
          <footer className="border-t bg-card">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4 text-xs text-muted-foreground">
              <span>
                OASIS-1 baseline (Anatomical + Clinical, no CNN) is treated as
                the primary, publication-safe result. Prototype multimodal
                results are exploratory.
              </span>
              <span className="text-[10px]">
                © {new Date().getFullYear()} NeuroScope · Research use only
              </span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}
