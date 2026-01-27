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
    default: "NeuroScope | Advanced AI Dementia Detection Research by Vishesh Sanghvi",
    template: "%s | NeuroScope Research"
  },
  description: "Pioneering deep learning research for early dementia detection by Vishesh Sanghvi. Featuring the novel 'Level-MAX' fusion architecture (0.808 AUC), this portal demonstrates robust biological biomarker integration and cross-dataset validation on 1,000+ subjects from OASIS & ADNI cohorts.",
  keywords: [
    "Vishesh Sanghvi",
    "Dementia AI",
    "Alzheimer's Detection",
    "Deep Learning Healthcare",
    "Multimodal Fusion",
    "Medical Imaging AI",
    "NeuroScope",
    "MRI Analysis",
    "Biomarker Fusion",
    "Cross-Dataset Generalization",
    "OASIS Dataset",
    "ADNI Dataset",
    "ResNet18",
    "Attention Mechanisms",
    "Medical AI Research"
  ],
  authors: [{ name: "Vishesh Sanghvi", url: "https://www.visheshsanghvi.me/" }],
  creator: "Vishesh Sanghvi",
  publisher: "Vishesh Sanghvi Research",
  applicationName: "NeuroScope",

  // Open Graph for social sharing
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://neuroscope.vercel.app",
    siteName: "NeuroScope by Vishesh Sanghvi",
    title: "NeuroScope – Advanced AI for Early Dementia Detection",
    description: "Explore the 0.808 AUC 'Level-MAX' model by Vishesh Sanghvi. A breakthrough in robust, biomarker-informed deep learning for Alzheimer's detection.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NeuroScope AI Research - Vishesh Sanghvi",
      }
    ],
  },

  // Twitter Card
  twitter: {
    card: "summary_large_image",
    title: "NeuroScope AI Research | Vishesh Sanghvi",
    description: "Deep learning breakdown: How 'Level-MAX' fusion achieves 0.808 AUC in early dementia detection. Robust cross-dataset validation by Vishesh Sanghvi.",
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

  // Icons - Using SVG icon from public folder
  icons: {
    icon: [
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-touch-icon.png",
  },

  // Manifest
  manifest: "/manifest.json",

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
        {/* CRITICAL: Prevent white flash on load - apply theme immediately */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const theme = localStorage.getItem('theme') || 
                    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
                  document.documentElement.classList.add(theme);
                  document.documentElement.style.colorScheme = theme;
                } catch (e) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />

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
              "name": "NeuroScope: Advanced Dementia AI",
              "alternateName": "Level-MAX Fusion Model",
              "description": "A pioneering deep learning research initiative by Vishesh Sanghvi, demonstrating how 'Level-MAX' biomarker fusion achieves 0.808 AUC in honest early dementia detection using MRI and biological profiles.",
              "url": "https://neuroscope.vercel.app",
              "author": {
                "@type": "Person",
                "name": "Vishesh Sanghvi",
                "url": "https://www.visheshsanghvi.me/",
                "sameAs": [
                  "https://linkedin.com/in/vishesh-sanghvi-96b16a237/",
                  "https://github.com/visheshsanghvi112"
                ]
              },
              "about": {
                "@type": "MedicalCondition",
                "name": "Alzheimer's Disease"
              },
              "keywords": "Vishesh Sanghvi, Dementia AI, Level-MAX, Alzheimer's, MRI, Deep Learning, OASIS, ADNI"
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
            <div className="mx-auto max-w-6xl px-4 md:px-6 py-6">
              <div className="flex flex-col gap-4">
                {/* Author Section */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-medium">Built by Vishesh Sanghvi</span>
                    <div className="flex items-center gap-2">
                      <a
                        href="https://www.visheshsanghvi.me/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-primary transition-colors"
                        aria-label="Portfolio"
                      >
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                        </svg>
                      </a>
                      <a
                        href="https://linkedin.com/in/vishesh-sanghvi-96b16a237/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-[#0A66C2] transition-colors"
                        aria-label="LinkedIn"
                      >
                        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                        </svg>
                      </a>
                      <a
                        href="https://github.com/visheshsanghvi112"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-foreground transition-colors"
                        aria-label="GitHub"
                      >
                        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                        </svg>
                      </a>
                    </div>
                  </div>
                  <span className="text-[10px] text-muted-foreground">
                    © {new Date().getFullYear()} NeuroScope · Research use only
                  </span>
                </div>
                {/* Research Info */}
                <div className="text-xs text-muted-foreground text-center sm:text-left border-t pt-4">
                  Cross-dataset robustness validated: OASIS ↔ ADNI. MRI features show strong transfer stability.
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}
