"use client"

import { useRef, useMemo, useState, useCallback } from "react"
import { Canvas, useFrame, extend } from "@react-three/fiber"
import { Points, shaderMaterial, Line, OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import { EffectComposer, Bloom, Vignette, ChromaticAberration } from "@react-three/postprocessing"
import { BlendFunction } from "postprocessing"
import { motion, AnimatePresence } from "framer-motion"

// =============================================================================
// BRAIN REGION DATA - Links to dementia research
// =============================================================================
const BRAIN_REGIONS = {
  frontal: {
    id: "frontal",
    name: "Frontal Lobe",
    color: "#00d4ff",
    description: "Executive function, decision-making, personality",
    dementiaRole: "Early behavioral changes in FTD. Reduced activity in Alzheimer's affects planning and judgment.",
    atrophyLevel: 0.85, // 0-1, higher = more healthy
  },
  parietal: {
    id: "parietal",
    name: "Parietal Lobe",
    color: "#00ff88",
    description: "Spatial awareness, sensory processing",
    dementiaRole: "Visuospatial deficits common in AD. Difficulty with navigation and object recognition.",
    atrophyLevel: 0.72,
  },
  temporal: {
    id: "temporal",
    name: "Temporal Lobe",
    color: "#ff00ff",
    description: "Memory formation, language, emotion",
    dementiaRole: "Hippocampal atrophy is hallmark of AD. Memory loss begins here.",
    atrophyLevel: 0.58, // Most affected in AD
  },
  occipital: {
    id: "occipital",
    name: "Occipital Lobe",
    color: "#ff8800",
    description: "Visual processing",
    dementiaRole: "Visual hallucinations in DLB. Preserved until late stages in typical AD.",
    atrophyLevel: 0.91,
  },
}

type RegionId = keyof typeof BRAIN_REGIONS

// =============================================================================
// ENHANCED HOLOGRAPHIC SHADER - With hover/selection states
// =============================================================================
const EnhancedHologramMaterial = shaderMaterial(
  {
    uTime: 0,
    uColorFrontal: new THREE.Color("#00d4ff"),
    uColorParietal: new THREE.Color("#00ff88"),
    uColorTemporal: new THREE.Color("#ff00ff"),
    uColorOccipital: new THREE.Color("#ff8800"),
    uHoveredRegion: -1.0, // -1 = none, 0-3 = region index
    uSelectedRegion: -1.0,
    uAtrophyFrontal: 0.85,
    uAtrophyParietal: 0.72,
    uAtrophyTemporal: 0.58,
    uAtrophyOccipital: 0.91,
  },
  // Vertex Shader
  `
    uniform float uTime;
    uniform float uHoveredRegion;
    uniform float uSelectedRegion;
    uniform float uAtrophyFrontal;
    uniform float uAtrophyParietal;
    uniform float uAtrophyTemporal;
    uniform float uAtrophyOccipital;
    
    attribute vec3 aRandom;
    attribute float aRegion;
    
    varying vec3 vColor;
    varying float vAlpha;
    varying float vRegion;

    uniform vec3 uColorFrontal;
    uniform vec3 uColorParietal;
    uniform vec3 uColorTemporal;
    uniform vec3 uColorOccipital;

    void main() {
      vRegion = aRegion;
      
      // Get atrophy level for this region
      float atrophy = 1.0;
      if (aRegion < 0.5) atrophy = uAtrophyFrontal;
      else if (aRegion < 1.5) atrophy = uAtrophyParietal;
      else if (aRegion < 2.5) atrophy = uAtrophyTemporal;
      else atrophy = uAtrophyOccipital;
      
      // Base color by region - blend towards red for low atrophy (unhealthy)
      vec3 healthyColor;
      if (aRegion < 0.5) healthyColor = uColorFrontal;
      else if (aRegion < 1.5) healthyColor = uColorParietal;
      else if (aRegion < 2.5) healthyColor = uColorTemporal;
      else healthyColor = uColorOccipital;
      
      vec3 atrophyColor = vec3(1.0, 0.2, 0.1); // Red for atrophy
      vColor = mix(atrophyColor, healthyColor, atrophy);

      // Position with brain shrinkage based on atrophy
      vec3 pos = position * (0.85 + atrophy * 0.15);
      
      // Hover effect - expand hovered region
      bool isHovered = abs(aRegion - uHoveredRegion) < 0.5;
      bool isSelected = abs(aRegion - uSelectedRegion) < 0.5;
      
      if (isHovered) {
        pos *= 1.08;
        vColor *= 1.5; // Brighter
      }
      if (isSelected) {
        pos *= 1.05;
        vColor *= 2.0; // Even brighter
      }
      
      // Glitch effect
      float glitch = step(0.985, sin(uTime * 2.0 + pos.y * 10.0));
      pos.x += glitch * (sin(uTime * 50.0) * 0.03);

      // Scanning wave
      float scanWave = sin(pos.y * 4.0 - uTime * 1.5);
      float pulse = smoothstep(0.85, 1.0, scanWave) * 0.4;
      
      // Particle size
      float baseSize = 0.035;
      float size = baseSize * (1.0 + pulse * 1.5);
      
      // Twinkle
      float twinkle = sin(uTime * aRandom.x + aRandom.y) * 0.5 + 0.5;
      size *= (0.85 + 0.3 * twinkle);
      
      // Atrophy affects size - less healthy = smaller particles
      size *= (0.7 + atrophy * 0.3);
      
      // Hover makes bigger
      if (isHovered) size *= 1.3;
      if (isSelected) size *= 1.2;

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = size * (280.0 / -mvPosition.z);

      vAlpha = 0.7 + 0.3 * pulse + 0.15 * twinkle;
      
      // Dim non-selected regions when something is selected
      if (uSelectedRegion >= 0.0 && !isSelected) {
        vAlpha *= 0.3;
      }
    }
  `,
  // Fragment Shader
  `
    varying vec3 vColor;
    varying float vAlpha;
    varying float vRegion;

    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;

      float glow = 1.0 - (dist * 2.0);
      glow = pow(glow, 1.3);

      vec3 finalColor = vColor * 1.4;
      
      gl_FragColor = vec4(finalColor, vAlpha * glow);
    }
  `
)

extend({ EnhancedHologramMaterial })

declare module "@react-three/fiber" {
  interface ThreeElements {
    enhancedHologramMaterial: any
  }
}

// =============================================================================
// NEURAL CONNECTIONS - Lines between particles
// =============================================================================
function NeuralConnections({ positions, count = 150 }: { positions: Float32Array; count?: number }) {
  const lines = useMemo(() => {
    const connections: Array<{ start: THREE.Vector3; end: THREE.Vector3; color: string }> = []
    const numParticles = positions.length / 3

    for (let i = 0; i < count; i++) {
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

      // Only connect nearby particles
      if (start.distanceTo(end) < 0.8) {
        connections.push({
          start,
          end,
          color: `hsl(${180 + Math.random() * 60}, 100%, 60%)`, // Cyan-ish
        })
      }
    }
    return connections
  }, [positions, count])

  const groupRef = useRef<THREE.Group>(null)
  const [opacity, setOpacity] = useState(0.15)

  useFrame((state) => {
    // Pulsing opacity
    const pulse = Math.sin(state.clock.elapsedTime * 0.8) * 0.1 + 0.2
    setOpacity(pulse)

    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.15
    }
  })

  return (
    <group ref={groupRef}>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={[line.start, line.end]}
          color={line.color}
          lineWidth={0.5}
          opacity={opacity}
          transparent
        />
      ))}
    </group>
  )
}

// =============================================================================
// REGION HITBOXES - Invisible spheres for click detection
// =============================================================================
function RegionHitboxes({
  onHover,
  onClick,
}: {
  onHover: (region: RegionId | null) => void
  onClick: (region: RegionId) => void
}) {
  const hitboxes = [
    { region: "frontal" as RegionId, position: [0, 0.2, -0.8] as [number, number, number], scale: 0.7 },
    { region: "parietal" as RegionId, position: [0, 0.5, 0] as [number, number, number], scale: 0.8 },
    { region: "temporal" as RegionId, position: [0.7, -0.3, 0] as [number, number, number], scale: 0.6 },
    { region: "occipital" as RegionId, position: [0, 0, 1.0] as [number, number, number], scale: 0.5 },
  ]

  return (
    <group>
      {hitboxes.map(({ region, position, scale }) => (
        <mesh
          key={region}
          position={position}
          scale={scale}
          onPointerOver={() => onHover(region)}
          onPointerOut={() => onHover(null)}
          onClick={() => onClick(region)}
        >
          <sphereGeometry args={[1, 8, 8]} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>
      ))}
    </group>
  )
}

// =============================================================================
// ENHANCED BRAIN PARTICLES
// =============================================================================
function EnhancedBrainParticles({
  count = 5000,
  hoveredRegion,
  selectedRegion,
}: {
  count?: number
  hoveredRegion: RegionId | null
  selectedRegion: RegionId | null
}) {
  const pointsRef = useRef<THREE.Points>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)

  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const aRandom = new Float32Array(count * 3)
    const aRegion = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      let x, y, z
      let found = false
      while (!found) {
        x = (Math.random() - 0.5) * 2.5
        y = (Math.random() - 0.5) * 2.0
        z = (Math.random() - 0.5) * 3.0

        const r = x * x / 1.0 + y * y / 0.8 + z * z / 1.5
        const inner = x * x / 0.2 + y * y / 0.2 + z * z / 0.3

        if (r < 1 && inner > 0.5) {
          if (Math.sin(x * 5) * Math.cos(z * 5) > -0.6) {
            found = true
          }
        }
      }

      positions[i * 3] = x!
      positions[i * 3 + 1] = y!
      positions[i * 3 + 2] = z!

      aRandom[i * 3] = Math.random() * 5.0 + 1.0
      aRandom[i * 3 + 1] = Math.random() * 10.0
      aRandom[i * 3 + 2] = Math.random()

      let region = 1.0
      if (z! < -0.5) region = 0.0
      else if (z! > 0.8) region = 3.0
      else if (y! < -0.3) region = 2.0

      aRegion[i] = region
    }

    return { positions, aRandom, aRegion }
  }, [count])

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime

      // Update hover/selection uniforms
      const regionToIndex: Record<string, number> = { frontal: 0, parietal: 1, temporal: 2, occipital: 3 }
      materialRef.current.uniforms.uHoveredRegion.value = hoveredRegion ? regionToIndex[hoveredRegion] : -1
      materialRef.current.uniforms.uSelectedRegion.value = selectedRegion ? regionToIndex[selectedRegion] : -1
    }
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.12
      const s = 1 + Math.sin(state.clock.elapsedTime * 0.4) * 0.015
      pointsRef.current.scale.set(s, s, s)
    }
  })

  return (
    <>
      <Points ref={pointsRef} positions={particles.positions} stride={3}>
        <bufferGeometry attach="geometry">
          <bufferAttribute
            attach="attributes-position"
            count={particles.positions.length / 3}
            args={[particles.positions, 3]}
          />
          <bufferAttribute
            attach="attributes-aRandom"
            count={particles.aRandom.length / 3}
            args={[particles.aRandom, 3]}
          />
          <bufferAttribute
            attach="attributes-aRegion"
            count={particles.aRegion.length}
            args={[particles.aRegion, 1]}
          />
        </bufferGeometry>
        {/* @ts-ignore */}
        <enhancedHologramMaterial
          ref={materialRef}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </Points>
      <NeuralConnections positions={particles.positions} count={120} />
    </>
  )
}

// =============================================================================
// GLASS SKULL - Enhanced with interaction
// =============================================================================
function EnhancedGlassSkull({ opacity = 0.15 }: { opacity?: number }) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = -state.clock.elapsedTime * 0.04
    }
  })

  return (
    <mesh ref={meshRef} scale={1.25}>
      <sphereGeometry args={[1.5, 48, 48]} />
      <meshPhysicalMaterial
        transparent
        opacity={opacity}
        roughness={0.1}
        metalness={0.1}
        envMapIntensity={1}
        clearcoat={1}
        clearcoatRoughness={0.1}
        color="#88ccff"
      />
    </mesh>
  )
}

// =============================================================================
// SCENE - Main 3D scene with all elements
// =============================================================================
function Scene({
  hoveredRegion,
  selectedRegion,
  onHover,
  onClick,
}: {
  hoveredRegion: RegionId | null
  selectedRegion: RegionId | null
  onHover: (region: RegionId | null) => void
  onClick: (region: RegionId) => void
}) {
  return (
    <>
      <color attach="background" args={["#030308"]} />

      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={0.5} color="#00d4ff" />
      <pointLight position={[-5, -5, 5]} intensity={0.3} color="#ff00ff" />

      <EnhancedBrainParticles
        count={6000}
        hoveredRegion={hoveredRegion}
        selectedRegion={selectedRegion}
      />

      <EnhancedGlassSkull opacity={selectedRegion ? 0.05 : 0.12} />

      <RegionHitboxes onHover={onHover} onClick={onClick} />

      <OrbitControls
        enableZoom={true}
        enablePan={false}
        autoRotate={!selectedRegion}
        autoRotateSpeed={0.4}
        maxDistance={7}
        minDistance={2.5}
        maxPolarAngle={Math.PI * 0.75}
        minPolarAngle={Math.PI * 0.25}
      />

      <EffectComposer>
        <Bloom
          luminanceThreshold={0.4}
          mipmapBlur
          intensity={1.0}
          radius={0.7}
        />
        <ChromaticAberration
          offset={new THREE.Vector2(0.0005, 0.0005)}
          radialModulation={false}
          modulationOffset={0}
          blendFunction={BlendFunction.NORMAL}
        />
        <Vignette
          eskil={false}
          offset={0.15}
          darkness={0.8}
        />
      </EffectComposer>
    </>
  )
}

// =============================================================================
// REGION INFO PANEL - Shows when region is selected
// =============================================================================
function RegionInfoPanel({
  region,
  onClose,
}: {
  region: RegionId
  onClose: () => void
}) {
  const info = BRAIN_REGIONS[region]
  const healthPercent = Math.round(info.atrophyLevel * 100)

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="absolute right-4 top-4 w-72 bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl p-4 text-white z-10"
      style={{ borderLeftColor: info.color, borderLeftWidth: 3 }}
    >
      <button
        onClick={onClose}
        className="absolute right-3 top-3 text-white/50 hover:text-white text-lg"
      >
        ×
      </button>

      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-3 h-3 rounded-full animate-pulse"
          style={{ backgroundColor: info.color }}
        />
        <h3 className="font-semibold text-sm" style={{ color: info.color }}>
          {info.name}
        </h3>
      </div>

      <p className="text-xs text-white/70 mb-3">{info.description}</p>

      <div className="mb-3">
        <div className="flex justify-between text-[10px] mb-1">
          <span className="text-white/50">Volume Preservation</span>
          <span style={{ color: healthPercent > 75 ? "#00ff88" : healthPercent > 50 ? "#ffaa00" : "#ff4444" }}>
            {healthPercent}%
          </span>
        </div>
        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${healthPercent}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="h-full rounded-full"
            style={{
              backgroundColor: healthPercent > 75 ? "#00ff88" : healthPercent > 50 ? "#ffaa00" : "#ff4444",
            }}
          />
        </div>
      </div>

      <div className="bg-white/5 rounded-lg p-2.5">
        <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">
          Role in Dementia
        </div>
        <p className="text-[11px] text-white/80 leading-relaxed">
          {info.dementiaRole}
        </p>
      </div>
    </motion.div>
  )
}

// =============================================================================
// LEGEND OVERLAY
// =============================================================================
function LegendOverlay({ hoveredRegion }: { hoveredRegion: RegionId | null }) {
  return (
    <div className="absolute bottom-4 left-4 right-4 pointer-events-none">
      <div className="bg-black/60 backdrop-blur-md border border-white/10 px-4 py-3 rounded-xl">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-4 text-[10px] font-mono">
            {Object.entries(BRAIN_REGIONS).map(([key, region]) => (
              <motion.span
                key={key}
                className="flex items-center gap-1.5"
                animate={{
                  opacity: hoveredRegion === key ? 1 : hoveredRegion ? 0.3 : 0.8,
                  scale: hoveredRegion === key ? 1.1 : 1,
                }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: region.color }}
                />
                <span style={{ color: hoveredRegion === key ? region.color : "rgba(255,255,255,0.6)" }}>
                  {region.name.split(" ")[0].toUpperCase()}
                </span>
              </motion.span>
            ))}
          </div>

          <div className="text-[9px] text-white/40 flex items-center gap-2">
            <span>🖱️ Drag to rotate</span>
            <span>•</span>
            <span>Scroll to zoom</span>
            <span>•</span>
            <span>Click region for details</span>
          </div>
        </div>

        <div className="mt-2 pt-2 border-t border-white/5 flex items-center justify-between">
          <div className="text-[9px] text-cyan-400/70 font-mono">
            HOLOGRAPHIC NEURAL MAPPING v2.0
          </div>
          <div className="flex items-center gap-1 text-[9px]">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span className="text-green-400/70">LIVE</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function BrainVisualizationEnhanced() {
  const [hoveredRegion, setHoveredRegion] = useState<RegionId | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<RegionId | null>(null)
  const [isInView, setIsInView] = useState(true) // Start as true for immediate rendering
  const containerRef = useRef<HTMLDivElement>(null)

  const handleClick = useCallback((region: RegionId) => {
    setSelectedRegion((prev) => (prev === region ? null : region))
  }, [])

  return (
    <div ref={containerRef} className="relative h-full w-full">
      {isInView ? (
        <Canvas
          camera={{ position: [0, 0.5, 4.5], fov: 45 }}
          style={{ width: "100%", height: "100%" }}
          gl={{ antialias: true, alpha: false }}
          dpr={[1, 1.5]} // Limit pixel ratio for performance
        >
          <Scene
            hoveredRegion={hoveredRegion}
            selectedRegion={selectedRegion}
            onHover={setHoveredRegion}
            onClick={handleClick}
          />
        </Canvas>
      ) : (
        <div className="w-full h-full bg-[#030308] flex items-center justify-center">
          <div className="w-12 h-12 border-2 border-cyan-500/30 rounded-full animate-pulse" />
        </div>
      )}

      {/* Region Info Panel */}
      <AnimatePresence>
        {selectedRegion && (
          <RegionInfoPanel
            region={selectedRegion}
            onClose={() => setSelectedRegion(null)}
          />
        )}
      </AnimatePresence>

      {/* Legend */}
      <LegendOverlay hoveredRegion={hoveredRegion} />

      {/* Top corner accents */}
      <div className="absolute inset-x-0 top-0 p-4 flex justify-between pointer-events-none">
        <div className="h-1 w-20 bg-gradient-to-r from-cyan-500/50 to-transparent rounded-full" />
        <div className="h-1 w-20 bg-gradient-to-l from-cyan-500/50 to-transparent rounded-full" />
      </div>
    </div>
  )
}
