"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import Image from "next/image"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { Alert } from "@/components/ui/alert"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import {
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Move,
  Download,
  ChevronLeft,
  ChevronRight,
  Eye,
  Layers,
  Grid3X3,
  Sparkles
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

// Visualization data organized by category
const visualizations = {
  oasis: [
    {
      id: "A1",
      src: "/figures/A1_oasis_model_comparison.png",
      title: "OASIS Model Comparison",
      description: "Comparison of different model architectures on OASIS-1 dataset showing MRI-only, Late Fusion, and Attention Fusion performance.",
      insight: "MRI-only achieves 0.78 AUC, with minimal gains from fusion approaches."
    },
    {
      id: "A2",
      src: "/figures/A2_oasis_class_distribution.png",
      title: "OASIS Class Distribution",
      description: "Distribution of CDR 0 (healthy) vs CDR 0.5+ (very mild dementia) in OASIS-1 dataset.",
      insight: "Well-balanced dataset with 206 CDR=0 and 230 CDR≥0.5 subjects."
    },
  ],
  adni: [
    {
      id: "B1",
      src: "/figures/B1_adni_level1_honest.png",
      title: "ADNI Level-1 Honest Baseline",
      description: "Performance on ADNI without cognitive scores (MMSE, CDR-SB) - the realistic early detection scenario.",
      insight: "Level-1 achieves 0.60 AUC - the honest baseline for early detection."
    },
    {
      id: "B2",
      src: "/figures/B2_level1_vs_level2_circularity.png",
      title: "Level-1 vs Level-2 Circularity",
      description: "Dramatic performance gap when including vs excluding cognitive scores reveals circularity problem.",
      insight: "Level-2 (0.99 AUC) uses cognitive scores that ARE the diagnosis - not early detection!"
    },
    {
      id: "B3",
      src: "/figures/B3_adni_class_distribution.png",
      title: "ADNI Class Distribution",
      description: "Distribution of CN (cognitively normal) vs MCI/AD in ADNI-1 dataset.",
      insight: "629 unique subjects after de-duplication: 195 CN, 298 MCI, 136 AD."
    },
  ],
  transfer: [
    {
      id: "C1",
      src: "/figures/C1_in_vs_cross_dataset_collapse.png",
      title: "In-Dataset vs Cross-Dataset Collapse",
      description: "Performance comparison showing how models perform in-dataset vs cross-dataset transfer.",
      insight: "All models show 15-25% AUC drop during cross-dataset transfer."
    },
    {
      id: "C2",
      src: "/figures/C2_transfer_robustness_heatmap.png",
      title: "Transfer Robustness Heatmap",
      description: "Heatmap showing transfer performance in both directions: OASIS→ADNI and ADNI→OASIS.",
      insight: "MRI-only is most robust for OASIS→ADNI; Late Fusion best for ADNI→OASIS."
    },
    {
      id: "C3",
      src: "/figures/C3_auc_drop_robustness.png",
      title: "AUC Drop Analysis",
      description: "Quantification of performance degradation during cross-dataset transfer.",
      insight: "Attention Fusion shows highest variance and poorest robustness."
    },
  ],
  preprocessing: [
    {
      id: "D1",
      src: "/figures/D1_preprocessing_pipeline.png",
      title: "Preprocessing Pipeline",
      description: "Complete data preprocessing workflow from raw MRI to extracted features.",
      insight: "7 major cleaning steps ensure zero data leakage and proper validation."
    },
    {
      id: "D2",
      src: "/figures/D2_sample_size_reduction.png",
      title: "Sample Size Reduction",
      description: "How subject-level de-duplication reduces ADNI from 1,825 scans to 629 unique subjects.",
      insight: "Baseline-only selection prevents temporal leakage in cross-sectional analysis."
    },
    {
      id: "D3",
      src: "/figures/D3_age_distribution.png",
      title: "Age Distribution",
      description: "Age distribution across datasets and diagnostic groups.",
      insight: "ADNI subjects generally older (avg 75) than OASIS (avg 70)."
    },
    {
      id: "D4",
      src: "/figures/D4_sex_distribution.png",
      title: "Sex Distribution",
      description: "Sex distribution across OASIS and ADNI datasets.",
      insight: "Both datasets show slight female predominance (~55%)."
    },
    {
      id: "D5",
      src: "/figures/D5_feature_dimensions.png",
      title: "Feature Dimensions",
      description: "Breakdown of feature dimensions: 512 MRI + 2 clinical vs ideal biomarker setup.",
      insight: "512 MRI features vs 2 clinical creates dimension imbalance in fusion."
    },
  ],
  embeddings: [
    {
      id: "E1",
      src: "/static/attention_weights.png",
      title: "Attention Weights",
      description: "Attention distribution across modalities in the fusion model.",
      insight: "Attention concentrates on age and global atrophy measures."
    },
    {
      id: "E2",
      src: "/static/embeddings_tsne.png",
      title: "t-SNE Embeddings",
      description: "t-SNE visualization of learned subject embeddings colored by diagnosis.",
      insight: "Smooth gradients with respect to age and CDR show meaningful representations."
    },
  ],
  longitudinal: [
    {
      id: "L1",
      src: "/figures/L1_phase1_resnet_results.png",
      title: "Phase 1: ResNet Results",
      description: "Initial experiment with ResNet18 features showing near-chance performance across all models.",
      insight: "All models (0.51-0.52 AUC) failed — prompted deep investigation."
    },
    {
      id: "L2",
      src: "/figures/L2_biomarker_power.png",
      title: "Individual Biomarker Power",
      description: "Predictive power of each structural biomarker for MCI→Dementia progression.",
      insight: "Hippocampus is the BEST single predictor at 0.725 AUC."
    },
    {
      id: "L3",
      src: "/figures/L3_feature_combinations.png",
      title: "Feature Combinations",
      description: "Performance comparison of different feature combinations from ResNet to biomarkers.",
      insight: "Biomarkers + Longitudinal achieves 0.848 AUC (+31 pts vs ResNet)."
    },
    {
      id: "L4",
      src: "/figures/L4_apoe4_risk.png",
      title: "APOE4 Genetic Risk",
      description: "Conversion rates by APOE4 allele count showing genetic risk stratification.",
      insight: "APOE4 carriers have 2x higher conversion risk (44% vs 23%)."
    },
    {
      id: "L5",
      src: "/figures/L5_longitudinal_improvement.png",
      title: "Longitudinal Improvement",
      description: "Direct comparison showing the benefit of adding temporal change information.",
      insight: "Adding atrophy rate improves AUC by +11.2% (0.74 → 0.848)."
    },
    {
      id: "L6",
      src: "/figures/L6_research_journey.png",
      title: "Complete Research Journey",
      description: "Visual summary of the three-phase journey from failure to breakthrough.",
      insight: "Right features (biomarkers) matter more than complex models (LSTM)."
    },
  ],
  level_max: [
    {
      id: "E1",
      src: "/figures/E1_level_max_auc_comparison.png",
      title: "Level-MAX AUC Comparison",
      description: "Comparison of MRI-Only vs. Level-MAX Fusion models showing significant AUC improvement.",
      insight: "+16.5% improvement (0.643 → 0.808) with 14D biological profile."
    },
    {
      id: "E2",
      src: "/figures/E2_level_max_accuracy_comparison.png",
      title: "Level-MAX Accuracy",
      description: "Classification accuracy comparison between baseline and enhanced fusion models.",
      insight: "Attention Fusion reaches 76.2% accuracy, significantly beating MRI-only baseline."
    },
    {
      id: "E3",
      src: "/figures/E3_level_max_summary.png",
      title: "Level-MAX Summary",
      description: "Comprehensive summary of AUC and Accuracy across all Level-MAX experiments.",
      insight: "Fusion models consistently outperform single-modality baselines when using proper biomarkers."
    },
  ],
}

type VisualizationType = { id: string; src: string; title: string; description: string; insight: string }

// Enhanced Image Viewer with Pan & Zoom
function EnhancedImageViewer({
  viz,
  onClose,
  onNext,
  onPrev,
  hasNext,
  hasPrev,
}: {
  viz: VisualizationType
  onClose: () => void
  onNext?: () => void
  onPrev?: () => void
  hasNext?: boolean
  hasPrev?: boolean
}) {
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [isLoading, setIsLoading] = useState(true)
  const [showControls, setShowControls] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const lastTouchDistance = useRef<number | null>(null)

  const MIN_SCALE = 0.5
  const MAX_SCALE = 5
  const ZOOM_STEP = 0.25

  const handleZoomIn = useCallback(() => {
    setScale(prev => Math.min(prev + ZOOM_STEP, MAX_SCALE))
  }, [])

  const handleZoomOut = useCallback(() => {
    setScale(prev => Math.max(prev - ZOOM_STEP, MIN_SCALE))
  }, [])

  const handleReset = useCallback(() => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }, [])

  const handleFitToScreen = useCallback(() => {
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }, [])

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
    setScale(prev => Math.min(Math.max(prev + delta, MIN_SCALE), MAX_SCALE))
  }, [])

  // Mouse drag for panning
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (scale > 1) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y })
    }
  }, [scale, position])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && scale > 1) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      })
    }
  }, [isDragging, scale, dragStart])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  // Touch handling for pinch-to-zoom
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const distance = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      )
      lastTouchDistance.current = distance
    } else if (e.touches.length === 1 && scale > 1) {
      setIsDragging(true)
      setDragStart({
        x: e.touches[0].clientX - position.x,
        y: e.touches[0].clientY - position.y,
      })
    }
  }, [scale, position])

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (e.touches.length === 2 && lastTouchDistance.current !== null) {
      e.preventDefault()
      const distance = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      )
      const delta = (distance - lastTouchDistance.current) * 0.01
      setScale(prev => Math.min(Math.max(prev + delta, MIN_SCALE), MAX_SCALE))
      lastTouchDistance.current = distance
    } else if (e.touches.length === 1 && isDragging && scale > 1) {
      setPosition({
        x: e.touches[0].clientX - dragStart.x,
        y: e.touches[0].clientY - dragStart.y,
      })
    }
  }, [isDragging, scale, dragStart])

  const handleTouchEnd = useCallback(() => {
    setIsDragging(false)
    lastTouchDistance.current = null
  }, [])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case '+':
        case '=':
          handleZoomIn()
          break
        case '-':
          handleZoomOut()
          break
        case '0':
          handleReset()
          break
        case 'ArrowLeft':
          if (hasPrev && onPrev) onPrev()
          break
        case 'ArrowRight':
          if (hasNext && onNext) onNext()
          break
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose, handleZoomIn, handleZoomOut, handleReset, hasNext, hasPrev, onNext, onPrev])

  // Auto-hide controls
  useEffect(() => {
    let timeout: NodeJS.Timeout
    const handleActivity = () => {
      setShowControls(true)
      clearTimeout(timeout)
      timeout = setTimeout(() => setShowControls(false), 3000)
    }
    handleActivity()
    window.addEventListener('mousemove', handleActivity)
    window.addEventListener('touchstart', handleActivity)
    return () => {
      clearTimeout(timeout)
      window.removeEventListener('mousemove', handleActivity)
      window.removeEventListener('touchstart', handleActivity)
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col"
    >
      {/* Top Controls Bar */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-0 left-0 right-0 z-20 bg-gradient-to-b from-black/80 to-transparent p-3 sm:p-4"
          >
            <div className="flex items-center justify-between max-w-7xl mx-auto">
              {/* Title */}
              <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                <Badge className="bg-emerald-600/90 text-white text-[10px] sm:text-xs shrink-0">
                  {viz.id}
                </Badge>
                <h3 className="text-white font-semibold text-sm sm:text-base truncate">
                  {viz.title}
                </h3>
              </div>

              {/* Close Button */}
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="text-white/70 hover:text-white hover:bg-white/10 shrink-0"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Image Container */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Loading State */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center z-10"
            >
              <div className="flex flex-col items-center gap-3">
                <div className="relative w-12 h-12">
                  <div className="absolute inset-0 rounded-full border-2 border-emerald-500/30" />
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-emerald-500 animate-spin" />
                </div>
                <span className="text-white/50 text-sm">Loading visualization...</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Image with Transform */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center p-4 sm:p-8"
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transition: isDragging ? 'none' : 'transform 0.2s ease-out',
          }}
        >
          <div className="relative w-full h-full max-w-6xl max-h-[80vh]">
            <Image
              src={viz.src}
              alt={viz.title}
              fill
              style={{ objectFit: "contain" }}
              className="select-none"
              onLoad={() => setIsLoading(false)}
              onError={() => setIsLoading(false)}
              unoptimized
              priority
            />
          </div>
        </motion.div>

        {/* Navigation Arrows */}
        <AnimatePresence>
          {showControls && (
            <>
              {hasPrev && onPrev && (
                <motion.button
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  onClick={onPrev}
                  className="absolute left-2 sm:left-4 top-1/2 -translate-y-1/2 z-20 p-2 sm:p-3 rounded-full bg-white/10 backdrop-blur-sm hover:bg-white/20 text-white transition-all hover:scale-110"
                >
                  <ChevronLeft className="h-5 w-5 sm:h-6 sm:w-6" />
                </motion.button>
              )}
              {hasNext && onNext && (
                <motion.button
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  onClick={onNext}
                  className="absolute right-2 sm:right-4 top-1/2 -translate-y-1/2 z-20 p-2 sm:p-3 rounded-full bg-white/10 backdrop-blur-sm hover:bg-white/20 text-white transition-all hover:scale-110"
                >
                  <ChevronRight className="h-5 w-5 sm:h-6 sm:w-6" />
                </motion.button>
              )}
            </>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Controls Bar */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/90 to-transparent p-3 sm:p-4"
          >
            <div className="max-w-4xl mx-auto space-y-3">
              {/* Zoom Controls */}
              <div className="flex items-center justify-center gap-1 sm:gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleZoomOut}
                  disabled={scale <= MIN_SCALE}
                  className="text-white/70 hover:text-white hover:bg-white/10 h-8 w-8 sm:h-9 sm:w-9 p-0"
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>

                {/* Zoom Level Indicator */}
                <div className="px-3 py-1.5 rounded-full bg-white/10 backdrop-blur-sm min-w-[80px] text-center">
                  <span className="text-white text-xs sm:text-sm font-mono">
                    {Math.round(scale * 100)}%
                  </span>
                </div>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleZoomIn}
                  disabled={scale >= MAX_SCALE}
                  className="text-white/70 hover:text-white hover:bg-white/10 h-8 w-8 sm:h-9 sm:w-9 p-0"
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>

                <div className="w-px h-5 bg-white/20 mx-1 sm:mx-2" />

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleReset}
                  className="text-white/70 hover:text-white hover:bg-white/10 h-8 w-8 sm:h-9 sm:w-9 p-0"
                  title="Reset View (0)"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleFitToScreen}
                  className="text-white/70 hover:text-white hover:bg-white/10 h-8 w-8 sm:h-9 sm:w-9 p-0"
                  title="Fit to Screen"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>

              {/* Description & Insight */}
              <div className="text-center space-y-1.5 px-4">
                <p className="text-white/60 text-xs sm:text-sm line-clamp-2">
                  {viz.description}
                </p>
                <p className="text-emerald-400 text-xs sm:text-sm font-medium">
                  💡 {viz.insight}
                </p>
              </div>

              {/* Keyboard Shortcuts Hint */}
              <div className="hidden sm:flex items-center justify-center gap-4 text-[10px] text-white/30">
                <span>Scroll to zoom</span>
                <span>•</span>
                <span>Drag to pan</span>
                <span>•</span>
                <span>← → Navigate</span>
                <span>•</span>
                <span>ESC Close</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// Enhanced Visualization Card
function VisualizationCard({
  viz,
  onOpen,
  index,
}: {
  viz: VisualizationType
  onOpen: () => void
  index: number
}) {
  const [imageError, setImageError] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
    >
      <Card
        className="group overflow-hidden cursor-pointer border-border/50 hover:border-emerald-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-emerald-500/5 bg-gradient-to-br from-background to-muted/30"
        onClick={onOpen}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Image Container */}
        <div className="relative h-40 sm:h-48 lg:h-52 w-full overflow-hidden bg-muted/30">
          {/* Loading Skeleton */}
          <AnimatePresence>
            {isLoading && !imageError && (
              <motion.div
                initial={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-gradient-to-br from-muted/50 to-muted/30"
              >
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-8 h-8 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
                </div>
                {/* Shimmer Effect */}
                <div className="absolute inset-0 -translate-x-full animate-[shimmer_1.5s_infinite] bg-gradient-to-r from-transparent via-white/5 to-transparent" />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error State */}
          {imageError ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-muted-foreground">
              <Layers className="h-8 w-8 opacity-50" />
              <span className="text-xs">Image unavailable</span>
            </div>
          ) : (
            <>
              {/* Image */}
              <Image
                src={viz.src}
                alt={viz.title}
                fill
                style={{ objectFit: "contain" }}
                className="p-3 transition-transform duration-500 group-hover:scale-105"
                onError={() => setImageError(true)}
                onLoad={() => setIsLoading(false)}
                unoptimized
              />

              {/* Hover Overlay */}
              <motion.div
                initial={false}
                animate={{ opacity: isHovered ? 1 : 0 }}
                className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent flex items-center justify-center"
              >
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: isHovered ? 1 : 0.8, opacity: isHovered ? 1 : 0 }}
                  transition={{ duration: 0.2 }}
                  className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/20 backdrop-blur-sm text-white"
                >
                  <ZoomIn className="h-4 w-4" />
                  <span className="text-sm font-medium">Click to zoom</span>
                </motion.div>
              </motion.div>

              {/* Badge Overlay */}
              <div className="absolute top-2 left-2">
                <Badge
                  variant="secondary"
                  className="bg-black/60 backdrop-blur-sm text-white border-0 text-[10px] font-mono"
                >
                  {viz.id}
                </Badge>
              </div>
            </>
          )}
        </div>

        {/* Content */}
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm font-semibold line-clamp-1 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
            {viz.title}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 pb-4">
          <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
            {viz.description}
          </p>
          <div className="flex items-start gap-1.5 pt-1">
            <Sparkles className="h-3 w-3 text-emerald-500 shrink-0 mt-0.5" />
            <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 line-clamp-2">
              {viz.insight}
            </p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

// Category Header Component
function CategoryHeader({ category, count }: { category: string; count: number }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
      <div className="flex items-center gap-2">
        <Grid3X3 className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          {category}
        </span>
        <Badge variant="secondary" className="text-[10px]">
          {count}
        </Badge>
      </div>
      <div className="h-px flex-1 bg-gradient-to-r from-border via-transparent to-transparent" />
    </div>
  )
}

export default function InterpretabilityPage() {
  const [tab, setTab] = useState("oasis")
  const [selectedViz, setSelectedViz] = useState<VisualizationType | null>(null)
  const [selectedIndex, setSelectedIndex] = useState<number>(0)

  // Get all visualizations in current tab as flat array
  const currentVisualizations = visualizations[tab as keyof typeof visualizations] || []

  const handleOpen = (viz: VisualizationType, index: number) => {
    setSelectedViz(viz)
    setSelectedIndex(index)
  }

  const handleClose = () => {
    setSelectedViz(null)
  }

  const handleNext = () => {
    if (selectedIndex < currentVisualizations.length - 1) {
      setSelectedIndex(selectedIndex + 1)
      setSelectedViz(currentVisualizations[selectedIndex + 1])
    }
  }

  const handlePrev = () => {
    if (selectedIndex > 0) {
      setSelectedIndex(selectedIndex - 1)
      setSelectedViz(currentVisualizations[selectedIndex - 1])
    }
  }

  return (
    <div className="flex w-full flex-col gap-6 px-2 sm:px-0">
      {/* Header */}
      <motion.section
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-3"
      >
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <Eye className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h2 className="text-lg sm:text-xl font-semibold tracking-tight">
              Research Visualizations
            </h2>
          </div>
          <Badge variant="outline" className="text-xs">22 Figures</Badge>
          <Badge className="bg-emerald-600 text-xs">Interactive Zoom</Badge>
        </div>
        <p className="text-sm text-muted-foreground max-w-2xl">
          Click any visualization for an immersive full-screen experience with pan & zoom.
          Use mouse wheel or pinch gestures to zoom, drag to pan when zoomed in.
        </p>
      </motion.section>

      {/* Tabs */}
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="flex flex-wrap h-auto gap-1 bg-muted/50 p-1 rounded-xl">
          <TabsTrigger
            value="oasis"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            OASIS ({visualizations.oasis.length})
          </TabsTrigger>
          <TabsTrigger
            value="adni"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            ADNI ({visualizations.adni.length})
          </TabsTrigger>
          <TabsTrigger
            value="transfer"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            Transfer ({visualizations.transfer.length})
          </TabsTrigger>
          <TabsTrigger
            value="preprocessing"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            Data ({visualizations.preprocessing.length})
          </TabsTrigger>
          <TabsTrigger
            value="embeddings"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            Embeddings ({visualizations.embeddings.length})
          </TabsTrigger>
          <TabsTrigger
            value="level_max"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            Level-MAX ({visualizations.level_max.length})
          </TabsTrigger>
          <TabsTrigger
            value="longitudinal"
            className="text-xs sm:text-sm data-[state=active]:bg-background data-[state=active]:shadow-sm rounded-lg"
          >
            🔬 Longitudinal ({visualizations.longitudinal.length})
          </TabsTrigger>
        </TabsList>

        {/* Tab Contents */}
        {Object.entries(visualizations).map(([key, vizList]) => (
          <TabsContent key={key} value={key} className="mt-6">
            <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {vizList.map((viz, index) => (
                <VisualizationCard
                  key={viz.id}
                  viz={viz}
                  onOpen={() => handleOpen(viz, index)}
                  index={index}
                />
              ))}
            </div>
          </TabsContent>
        ))}
      </Tabs>

      {/* Summary Stats */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="grid gap-3 sm:gap-4 grid-cols-2 lg:grid-cols-4"
      >
        <Card className="bg-gradient-to-br from-blue-500/5 to-transparent border-blue-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Layers className="h-4 w-4 text-blue-500" />
              Total Figures
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">22</div>
            <p className="text-xs text-muted-foreground">Publication-ready</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500/5 to-transparent border-purple-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Grid3X3 className="h-4 w-4 text-purple-500" />
              Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2</div>
            <p className="text-xs text-muted-foreground">OASIS-1 + ADNI-1</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-emerald-500/5 to-transparent border-emerald-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-emerald-500" />
              Key Finding
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold text-emerald-600">0.848 AUC</div>
            <p className="text-xs text-muted-foreground">Longitudinal biomarkers</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-500/5 to-transparent border-orange-500/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Eye className="h-4 w-4 text-orange-500" />
              Circularity Gap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-bold text-orange-600">+39%</div>
            <p className="text-xs text-muted-foreground">Level-1 vs Level-2</p>
          </CardContent>
        </Card>
      </motion.section>

      {/* Alert */}
      <Alert className="text-xs bg-muted/30 border-muted-foreground/20">
        <div className="flex items-start gap-2">
          <Eye className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          <div>
            All visualizations are generated from actual research data on OASIS-1 (436 subjects)
            and ADNI-1 (629 subjects). Click any figure to view full-size with zoom controls.
            <span className="hidden sm:inline text-muted-foreground/70">
              {" "}Use keyboard shortcuts: +/- to zoom, ← → to navigate, ESC to close.
            </span>
          </div>
        </div>
      </Alert>

      {/* Full-screen Image Viewer */}
      <AnimatePresence>
        {selectedViz && (
          <EnhancedImageViewer
            viz={selectedViz}
            onClose={handleClose}
            onNext={handleNext}
            onPrev={handlePrev}
            hasNext={selectedIndex < currentVisualizations.length - 1}
            hasPrev={selectedIndex > 0}
          />
        )}
      </AnimatePresence>

      {/* Custom Shimmer Animation */}
      <style jsx global>{`
        @keyframes shimmer {
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </div>
  )
}
