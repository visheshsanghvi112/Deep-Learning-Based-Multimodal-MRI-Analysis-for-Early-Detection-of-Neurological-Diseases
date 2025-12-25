import type { Metadata, Viewport } from "next"
import Link from "next/link"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Badge } from "@/components/ui/badge"
import { MainNav } from "@/components/main-nav"
import { ThemeToggle } from "@/components/theme-toggle"
import { OptimizedBackground } from "@/components/optimized-background"
import { MobileNav } from "@/components/mobile-nav"
import "@/lib/suppress-rsc-errors"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

// SEO Metadata
export const metadata: Metadata = {
  metadataBase: new URL("https://neuroscope.vercel.app"),
  title: {
    default: "NeuroScope – Multi-Dataset Dementia Detection Research Portal",
    template: "%s | NeuroScope"
  },
  description: "Research-grade deep learning portal for early dementia detection using structural MRI. Cross-validated on OASIS-1 & ADNI-1 datasets with robustness analysis across 1,065 subjects. ResNet18 features, multimodal fusion, and interpretability tools.",
  keywords: [
    "dementia detection",
    "Alzheimer's disease",
    "MRI analysis",
    "deep learning",
    "OASIS dataset",
    "ADNI dataset",
    "neuroimaging",
    "brain imaging",
    "multimodal fusion",
    "machine learning healthcare",
    "cognitive impairment",
    "ResNet18",
    "medical AI research"
  ],
  authors: [{ name: "Vishesh Sanghvi" }],
  creator: "Vishesh Sanghvi",
  publisher: "NeuroScope Research",

  // Open Graph for social sharing
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://neuroscope.vercel.app",
    siteName: "NeuroScope",
    title: "NeuroScope – Multi-Dataset Dementia Detection",
    description: "Research-grade deep learning portal for early dementia detection. Cross-validated on OASIS-1 & ADNI-1 with 1,065 subjects.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NeuroScope - Dementia Detection Research Portal",
      }
    ],
  },

  // Twitter Card
  twitter: {
    card: "summary_large_image",
    title: "NeuroScope – Dementia Detection Research",
    description: "Deep learning portal for MRI-based dementia detection. OASIS & ADNI datasets validated.",
    images: ["/og-image.png"],
    creator: "@visheshsanghvi",
  },

  // Robots
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },

  // Icons
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-touch-icon.png",
  },

  // Manifest
  manifest: "/manifest.json",

  // Verification (add your own codes when you have them)
  // verification: {
  //   google: "your-google-verification-code",
  // },

  // Category
  category: "Medical Research",
}

// Viewport configuration
export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#030308" },
  ],
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Preconnect to external resources */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />

        {/* Canonical URL */}
        <link rel="canonical" href="https://neuroscope.vercel.app" />

        {/* Structured Data - Organization */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "ResearchProject",
              "name": "NeuroScope",
              "description": "Multi-dataset dementia detection research using deep learning on structural MRI",
              "url": "https://neuroscope.vercel.app",
              "author": {
                "@type": "Person",
                "name": "Vishesh Sanghvi"
              },
              "about": {
                "@type": "MedicalCondition",
                "name": "Dementia"
              },
              "keywords": "dementia, Alzheimer's, MRI, deep learning, OASIS, ADNI"
            })
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground relative`}
        suppressHydrationWarning
      >
        <OptimizedBackground />
        <div className="flex min-h-screen flex-col relative z-0">
          <header className="border-b bg-card/80 backdrop-blur sticky top-0 z-40">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 md:px-6 py-4">
              <div className="flex items-center gap-3">
                <Link href="/" className="font-semibold tracking-tight text-lg hover:opacity-80 transition-opacity">
                  NeuroScope
                </Link>
                <Badge variant="secondary" className="text-xs hidden sm:inline-flex">
                  Research
                </Badge>
              </div>
              <div className="flex items-center gap-4">
                <div className="hidden md:block">
                  <MainNav />
                </div>
                <div className="flex items-center gap-2">
                  <ThemeToggle />
                  <MobileNav />
                </div>
              </div>
            </div>
            <div className="border-t bg-muted/40 hidden md:block">
              <div className="mx-auto max-w-6xl px-6 py-2 text-xs text-muted-foreground">
                This portal presents research models validated on <span className="font-medium">OASIS-1</span> and{" "}
                <span className="font-medium">ADNI-1</span> datasets with cross-dataset robustness analysis. Results are{" "}
                <span className="font-semibold">
                  not intended for clinical use.
                </span>
              </div>
            </div>
          </header>
          <main className="mx-auto flex w-full max-w-6xl flex-1 px-4 md:px-6 py-8">
            {children}
          </main>
          <footer className="border-t bg-card">
            <div className="mx-auto flex flex-col md:flex-row max-w-6xl items-center justify-between gap-4 px-4 md:px-6 py-6 text-xs text-muted-foreground text-center md:text-left">
              <span>
                Cross-dataset robustness validated: OASIS ↔ ADNI. MRI features show
                strong transfer stability. Multimodal fusion exhibits direction-dependent robustness.
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
