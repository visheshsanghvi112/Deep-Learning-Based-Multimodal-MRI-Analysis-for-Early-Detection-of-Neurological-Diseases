"use client"

import dynamic from "next/dynamic"

// Dynamic import to avoid SSR issues with Three.js
const FullBrainExplorer = dynamic(
  () => import("@/components/full-brain-explorer").then((mod) => mod.FullBrainExplorer),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-screen bg-[#020208] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-2 border-cyan-500/30 rounded-full animate-ping mx-auto mb-4" />
          <div className="text-cyan-400 text-sm font-mono">INITIALIZING NEURAL MAP</div>
        </div>
      </div>
    ),
  }
)

export default function BrainExplorerPage() {
  return <FullBrainExplorer />
}
