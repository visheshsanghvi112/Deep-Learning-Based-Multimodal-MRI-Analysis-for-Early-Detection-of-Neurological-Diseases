"use client"

import { useState, useRef, useCallback, useMemo } from "react"
import { Canvas, useFrame, extend } from "@react-three/fiber"
import { Points, shaderMaterial, Line, OrbitControls, PerspectiveCamera, Environment } from "@react-three/drei"
import * as THREE from "three"
import { EffectComposer, Bloom, Vignette, ChromaticAberration } from "@react-three/postprocessing"
import { BlendFunction } from "postprocessing"
import { motion, AnimatePresence } from "framer-motion"
import { Brain, Activity, Layers, ChevronRight, X, Zap, TrendingDown } from "lucide-react"

// =============================================================================
// BRAIN REGION DATA - Comprehensive dementia research data
// =============================================================================
const BRAIN_REGIONS = {
  frontal: {
    id: "frontal",
    name: "Frontal Lobe",
    color: "#00d4ff",
    glowColor: "#00a0cc",
    description: "Executive function, decision-making, personality, motor control",
    dementiaRole: "Early behavioral changes in frontotemporal dementia (FTD). Reduced activity affects planning, judgment, and impulse control in Alzheimer's disease.",
    atrophyLevel: 0.85,
    keyStructures: ["Prefrontal cortex", "Motor cortex", "Broca's area"],
    symptoms: ["Personality changes", "Poor judgment", "Difficulty planning"],
  },
  parietal: {
    id: "parietal",
    name: "Parietal Lobe", 
    color: "#00ff88",
    glowColor: "#00cc66",
    description: "Spatial awareness, sensory processing, navigation",
    dementiaRole: "Visuospatial deficits are common in AD. Patients may get lost in familiar places and have difficulty with object recognition.",
    atrophyLevel: 0.72,
    keyStructures: ["Somatosensory cortex", "Posterior parietal cortex"],
    symptoms: ["Getting lost", "Difficulty with objects", "Spatial confusion"],
  },
  temporal: {
    id: "temporal",
    name: "Temporal Lobe",
    color: "#ff00ff",
    glowColor: "#cc00cc",
    description: "Memory formation, language comprehension, emotion processing",
    dementiaRole: "Hippocampal atrophy is the hallmark of Alzheimer's disease. This is where memory loss begins, particularly affecting episodic memory.",
    atrophyLevel: 0.58,
    keyStructures: ["Hippocampus", "Amygdala", "Wernicke's area"],
    symptoms: ["Memory loss", "Language difficulties", "Emotional changes"],
  },
  occipital: {
    id: "occipital",
    name: "Occipital Lobe",
    color: "#ff8800",
    glowColor: "#cc6600",
    description: "Visual processing and interpretation",
    dementiaRole: "Visual hallucinations in Dementia with Lewy Bodies (DLB). Generally preserved until late stages in typical Alzheimer's disease.",
    atrophyLevel: 0.91,
    keyStructures: ["Primary visual cortex", "Visual association areas"],
    symptoms: ["Visual hallucinations", "Difficulty reading", "Color perception issues"],
  },
  hippocampus: {
    id: "hippocampus",
    name: "Hippocampus",
    color: "#ff3366",
    glowColor: "#cc2952",
    description: "Critical for forming new memories and spatial navigation",
    dementiaRole: "First structure affected in Alzheimer's disease. Volume loss of 15-25% in early AD compared to healthy aging.",
    atrophyLevel: 0.42,
    keyStructures: ["CA1-CA4 fields", "Dentate gyrus", "Subiculum"],
    symptoms: ["Short-term memory loss", "Disorientation", "Difficulty learning"],
  },
}

type RegionId = keyof typeof BRAIN_REGIONS

// =============================================================================
// SHADER MATERIAL
// =============================================================================
const BrainShaderMaterial = shaderMaterial(
  {
    uTime: 0,
    uHoveredRegion: -1.0,
    uSelectedRegion: -1.0,
    uIntensity: 1.0,
  },
  // Vertex
  `
    uniform float uTime;
    uniform float uHoveredRegion;
    uniform float uSelectedRegion;
    uniform float uIntensity;
    
    attribute vec3 aColor;
    attribute float aRegion;
    attribute vec3 aRandom;
    
    varying vec3 vColor;
    varying float vAlpha;

    void main() {
      vColor = aColor;
      
      vec3 pos = position;
      
      // Hover expansion
      bool isHovered = abs(aRegion - uHoveredRegion) < 0.5;
      bool isSelected = abs(aRegion - uSelectedRegion) < 0.5;
      
      if (isHovered) {
        pos *= 1.06;
        vColor *= 1.4;
      }
      if (isSelected) {
        pos *= 1.04;
        vColor *= 1.8;
      }
      
      // Pulse
      float pulse = sin(uTime * 2.0 + pos.y * 3.0) * 0.02;
      pos *= 1.0 + pulse;
      
      // Glitch
      float glitch = step(0.99, sin(uTime * 3.0 + pos.x * 20.0));
      pos.x += glitch * sin(uTime * 80.0) * 0.02;
      
      // Size
      float size = 0.03 * uIntensity;
      float twinkle = sin(uTime * aRandom.x + aRandom.y) * 0.5 + 0.5;
      size *= (0.8 + 0.4 * twinkle);
      
      if (isHovered) size *= 1.4;
      if (isSelected) size *= 1.3;

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = size * (250.0 / -mvPosition.z);

      vAlpha = 0.75 + 0.25 * twinkle;
      if (uSelectedRegion >= 0.0 && !isSelected) vAlpha *= 0.25;
    }
  `,
  // Fragment
  `
    varying vec3 vColor;
    varying float vAlpha;

    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;

      float glow = pow(1.0 - dist * 2.0, 1.5);
      gl_FragColor = vec4(vColor * 1.3, vAlpha * glow);
    }
  `
)

extend({ BrainShaderMaterial })

declare module "@react-three/fiber" {
  interface ThreeElements {
    brainShaderMaterial: any
  }
}

// =============================================================================
// BRAIN PARTICLES
// =============================================================================
function BrainParticles({
  hoveredRegion,
  selectedRegion,
}: {
  hoveredRegion: RegionId | null
  selectedRegion: RegionId | null
}) {
  const pointsRef = useRef<THREE.Points>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)
  const count = 7000

  const { positions, colors, regions, randoms } = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    const regions = new Float32Array(count)
    const randoms = new Float32Array(count * 3)

    const regionColors: Record<number, THREE.Color> = {
      0: new THREE.Color("#00d4ff"),
      1: new THREE.Color("#00ff88"),
      2: new THREE.Color("#ff00ff"),
      3: new THREE.Color("#ff8800"),
      4: new THREE.Color("#ff3366"),
    }

    for (let i = 0; i < count; i++) {
      let x, y, z, region
      let found = false

      while (!found) {
        x = (Math.random() - 0.5) * 2.8
        y = (Math.random() - 0.5) * 2.2
        z = (Math.random() - 0.5) * 3.2

        // Brain ellipsoid
        const r = (x * x) / 1.2 + (y * y) / 0.9 + (z * z) / 1.8
        const inner = (x * x) / 0.15 + (y * y) / 0.15 + (z * z) / 0.25

        if (r < 1 && inner > 0.6) {
          // Cortex folds
          if (Math.sin(x * 6) * Math.cos(z * 6) > -0.5) {
            found = true
          }
        }
      }

      // Region assignment
      if (Math.abs(x!) < 0.3 && y! < -0.2 && Math.abs(z!) < 0.4) {
        region = 4 // Hippocampus (deep, medial)
      } else if (z! < -0.6) {
        region = 0 // Frontal
      } else if (z! > 0.9) {
        region = 3 // Occipital
      } else if (y! < -0.4) {
        region = 2 // Temporal
      } else {
        region = 1 // Parietal
      }

      positions[i * 3] = x!
      positions[i * 3 + 1] = y!
      positions[i * 3 + 2] = z!

      const color = regionColors[region]
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b

      regions[i] = region

      randoms[i * 3] = Math.random() * 5 + 1
      randoms[i * 3 + 1] = Math.random() * 10
      randoms[i * 3 + 2] = Math.random()
    }

    return { positions, colors, regions, randoms }
  }, [])

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime

      const regionMap: Record<string, number> = {
        frontal: 0,
        parietal: 1,
        temporal: 2,
        occipital: 3,
        hippocampus: 4,
      }
      materialRef.current.uniforms.uHoveredRegion.value = hoveredRegion
        ? regionMap[hoveredRegion]
        : -1
      materialRef.current.uniforms.uSelectedRegion.value = selectedRegion
        ? regionMap[selectedRegion]
        : -1
    }
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.08
    }
  })

  return (
    <Points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} args={[positions, 3]} />
        <bufferAttribute attach="attributes-aColor" count={count} args={[colors, 3]} />
        <bufferAttribute attach="attributes-aRegion" count={count} args={[regions, 1]} />
        <bufferAttribute attach="attributes-aRandom" count={count} args={[randoms, 3]} />
      </bufferGeometry>
      {/* @ts-ignore */}
      <brainShaderMaterial
        ref={materialRef}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  )
}

// =============================================================================
// NEURAL CONNECTIONS
// =============================================================================
function NeuralNetwork({ positions }: { positions: Float32Array }) {
  const groupRef = useRef<THREE.Group>(null)

  const connections = useMemo(() => {
    const lines: Array<{ start: THREE.Vector3; end: THREE.Vector3 }> = []
    const numParticles = positions.length / 3

    for (let i = 0; i < 100; i++) {
      const idx1 = Math.floor(Math.random() * numParticles)
      const idx2 = Math.floor(Math.random() * numParticles)

      const start = new THREE.Vector3(
        positions[idx1 * 3],
        positions[idx1 * 3 + 1],
        positions[idx1 * 3 + 2]
      )
      const end = new THREE.Vector3(
        positions[idx2 * 3],
        positions[idx2 * 3 + 1],
        positions[idx2 * 3 + 2]
      )

      if (start.distanceTo(end) < 0.6 && start.distanceTo(end) > 0.1) {
        lines.push({ start, end })
      }
    }
    return lines
  }, [positions])

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.08
    }
  })

  return (
    <group ref={groupRef}>
      {connections.map((conn, i) => (
        <Line
          key={i}
          points={[conn.start, conn.end]}
          color="#00d4ff"
          lineWidth={0.3}
          opacity={0.15}
          transparent
        />
      ))}
    </group>
  )
}

// =============================================================================
// HITBOXES FOR CLICK DETECTION
// =============================================================================
function RegionHitboxes({
  onHover,
  onClick,
}: {
  onHover: (r: RegionId | null) => void
  onClick: (r: RegionId) => void
}) {
  const boxes = [
    { region: "frontal" as RegionId, pos: [0, 0.2, -0.9] as const, scale: 0.6 },
    { region: "parietal" as RegionId, pos: [0, 0.5, 0.1] as const, scale: 0.7 },
    { region: "temporal" as RegionId, pos: [0.6, -0.4, 0.1] as const, scale: 0.5 },
    { region: "occipital" as RegionId, pos: [0, 0.1, 1.1] as const, scale: 0.4 },
    { region: "hippocampus" as RegionId, pos: [0, -0.3, 0] as const, scale: 0.35 },
  ]

  return (
    <group>
      {boxes.map(({ region, pos, scale }) => (
        <mesh
          key={region}
          position={pos}
          scale={scale}
          onPointerOver={(e) => {
            e.stopPropagation()
            onHover(region)
            document.body.style.cursor = "pointer"
          }}
          onPointerOut={() => {
            onHover(null)
            document.body.style.cursor = "auto"
          }}
          onClick={(e) => {
            e.stopPropagation()
            onClick(region)
          }}
        >
          <sphereGeometry args={[1, 8, 8]} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>
      ))}
    </group>
  )
}

// =============================================================================
// SCENE
// =============================================================================
function Scene({
  hoveredRegion,
  selectedRegion,
  onHover,
  onClick,
}: {
  hoveredRegion: RegionId | null
  selectedRegion: RegionId | null
  onHover: (r: RegionId | null) => void
  onClick: (r: RegionId) => void
}) {
  const positions = useMemo(() => {
    const count = 7000
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      let x, y, z
      let found = false
      while (!found) {
        x = (Math.random() - 0.5) * 2.8
        y = (Math.random() - 0.5) * 2.2
        z = (Math.random() - 0.5) * 3.2
        const r = (x * x) / 1.2 + (y * y) / 0.9 + (z * z) / 1.8
        const inner = (x * x) / 0.15 + (y * y) / 0.15 + (z * z) / 0.25
        if (r < 1 && inner > 0.6 && Math.sin(x * 6) * Math.cos(z * 6) > -0.5) found = true
      }
      arr[i * 3] = x!
      arr[i * 3 + 1] = y!
      arr[i * 3 + 2] = z!
    }
    return arr
  }, [])

  return (
    <>
      <color attach="background" args={["#020208"]} />
      <fog attach="fog" args={["#020208", 5, 12]} />

      <ambientLight intensity={0.2} />
      <pointLight position={[3, 3, 3]} intensity={0.4} color="#00d4ff" />
      <pointLight position={[-3, -2, 3]} intensity={0.3} color="#ff00ff" />
      <pointLight position={[0, 0, -3]} intensity={0.2} color="#ff8800" />

      <BrainParticles hoveredRegion={hoveredRegion} selectedRegion={selectedRegion} />
      <NeuralNetwork positions={positions} />
      <RegionHitboxes onHover={onHover} onClick={onClick} />

      <OrbitControls
        enableZoom
        enablePan={false}
        autoRotate={!selectedRegion}
        autoRotateSpeed={0.3}
        minDistance={2.5}
        maxDistance={8}
        maxPolarAngle={Math.PI * 0.8}
        minPolarAngle={Math.PI * 0.2}
      />

      <EffectComposer>
        <Bloom luminanceThreshold={0.3} mipmapBlur intensity={0.8} radius={0.6} />
        <ChromaticAberration
          offset={new THREE.Vector2(0.0003, 0.0003)}
          radialModulation={false}
          modulationOffset={0}
          blendFunction={BlendFunction.NORMAL}
        />
        <Vignette eskil={false} offset={0.1} darkness={0.7} />
      </EffectComposer>
    </>
  )
}

// =============================================================================
// INFO PANEL
// =============================================================================
function RegionPanel({ region, onClose }: { region: RegionId; onClose: () => void }) {
  const info = BRAIN_REGIONS[region]
  const health = Math.round(info.atrophyLevel * 100)

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 30 }}
      className="absolute right-4 top-4 bottom-4 w-80 bg-black/90 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden flex flex-col"
      style={{ borderLeftColor: info.color, borderLeftWidth: 4 }}
    >
      {/* Header */}
      <div className="p-5 border-b border-white/10">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 p-1 rounded-lg hover:bg-white/10 transition-colors"
        >
          <X size={18} className="text-white/50" />
        </button>
        <div className="flex items-center gap-3">
          <div
            className="w-4 h-4 rounded-full animate-pulse"
            style={{ backgroundColor: info.color, boxShadow: `0 0 20px ${info.color}` }}
          />
          <h2 className="text-lg font-semibold" style={{ color: info.color }}>
            {info.name}
          </h2>
        </div>
        <p className="text-xs text-white/60 mt-2 leading-relaxed">{info.description}</p>
      </div>

      {/* Volume Health */}
      <div className="p-5 border-b border-white/10">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-white/50 flex items-center gap-1.5">
            <Activity size={12} />
            Volume Preservation
          </span>
          <span
            className="text-sm font-mono font-bold"
            style={{ color: health > 70 ? "#00ff88" : health > 50 ? "#ffaa00" : "#ff4444" }}
          >
            {health}%
          </span>
        </div>
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${health}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="h-full rounded-full"
            style={{
              background: `linear-gradient(90deg, ${
                health > 70 ? "#00ff88" : health > 50 ? "#ffaa00" : "#ff4444"
              }, ${info.color})`,
            }}
          />
        </div>
        {health < 60 && (
          <div className="flex items-center gap-1.5 mt-2 text-[10px] text-red-400">
            <TrendingDown size={10} />
            Significant atrophy detected
          </div>
        )}
      </div>

      {/* Key Structures */}
      <div className="p-5 border-b border-white/10">
        <div className="text-[10px] text-white/40 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <Layers size={10} />
          Key Structures
        </div>
        <div className="flex flex-wrap gap-1.5">
          {info.keyStructures.map((s) => (
            <span
              key={s}
              className="px-2 py-1 rounded-full text-[10px] bg-white/5 border border-white/10"
              style={{ color: info.color }}
            >
              {s}
            </span>
          ))}
        </div>
      </div>

      {/* Dementia Role */}
      <div className="p-5 flex-1 overflow-auto">
        <div className="text-[10px] text-white/40 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <Brain size={10} />
          Role in Dementia
        </div>
        <p className="text-xs text-white/70 leading-relaxed">{info.dementiaRole}</p>

        <div className="mt-4">
          <div className="text-[10px] text-white/40 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Zap size={10} />
            Common Symptoms
          </div>
          <ul className="space-y-1">
            {info.symptoms.map((symptom) => (
              <li key={symptom} className="flex items-center gap-2 text-xs text-white/60">
                <ChevronRight size={10} style={{ color: info.color }} />
                {symptom}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-white/10 bg-white/5">
        <div className="text-[9px] text-white/30 text-center">
          Data based on OASIS-1 cohort analysis
        </div>
      </div>
    </motion.div>
  )
}

// =============================================================================
// LEGEND
// =============================================================================
function Legend({ hoveredRegion }: { hoveredRegion: RegionId | null }) {
  return (
    <div className="absolute bottom-4 left-4 right-4 md:right-auto md:max-w-md">
      <div className="bg-black/70 backdrop-blur-xl border border-white/10 rounded-xl p-4">
        <div className="grid grid-cols-5 gap-2 mb-3">
          {Object.entries(BRAIN_REGIONS).map(([key, region]) => (
            <motion.div
              key={key}
              className="flex flex-col items-center gap-1 p-2 rounded-lg"
              animate={{
                backgroundColor:
                  hoveredRegion === key ? "rgba(255,255,255,0.1)" : "transparent",
                scale: hoveredRegion === key ? 1.05 : 1,
              }}
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{
                  backgroundColor: region.color,
                  boxShadow: hoveredRegion === key ? `0 0 10px ${region.color}` : "none",
                }}
              />
              <span
                className="text-[9px] font-medium text-center"
                style={{ color: hoveredRegion === key ? region.color : "rgba(255,255,255,0.5)" }}
              >
                {region.name.split(" ")[0]}
              </span>
            </motion.div>
          ))}
        </div>

        <div className="flex items-center justify-between text-[10px] text-white/40 border-t border-white/10 pt-3">
          <div className="flex items-center gap-3">
            <span>🖱️ Drag rotate</span>
            <span>⚙️ Scroll zoom</span>
            <span>👆 Click region</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-emerald-400">NEURAL MAP v2.1</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function FullBrainExplorer() {
  const [hoveredRegion, setHoveredRegion] = useState<RegionId | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<RegionId | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  const handleClick = useCallback((region: RegionId) => {
    setSelectedRegion((prev) => (prev === region ? null : region))
  }, [])

  return (
    <div className="relative w-full h-screen bg-[#020208]">
      {/* Loading */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 bg-[#020208] flex items-center justify-center"
          >
            <div className="text-center">
              <div className="w-16 h-16 border-2 border-cyan-500/30 rounded-full animate-ping mx-auto mb-4" />
              <div className="text-cyan-400 text-sm font-mono">LOADING NEURAL MAP</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        onCreated={() => setTimeout(() => setIsLoaded(true), 500)}
        gl={{ antialias: true }}
        dpr={[1, 2]}
      >
        <Scene
          hoveredRegion={hoveredRegion}
          selectedRegion={selectedRegion}
          onHover={setHoveredRegion}
          onClick={handleClick}
        />
      </Canvas>

      {/* Title */}
      <div className="absolute top-4 left-4">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <Brain className="text-cyan-400" size={24} />
          Neural Region Explorer
        </h1>
        <p className="text-xs text-white/50 mt-1">Interactive 3D Brain Mapping • OASIS-1 Research</p>
      </div>

      {/* Info Panel */}
      <AnimatePresence>
        {selectedRegion && <RegionPanel region={selectedRegion} onClose={() => setSelectedRegion(null)} />}
      </AnimatePresence>

      {/* Legend */}
      <Legend hoveredRegion={hoveredRegion} />
    </div>
  )
}
