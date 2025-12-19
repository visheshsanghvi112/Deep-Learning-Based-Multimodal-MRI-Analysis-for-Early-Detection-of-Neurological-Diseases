"use client"

import { useRef, useMemo } from "react"
import { Canvas, useFrame, extend } from "@react-three/fiber"
import { Points, shaderMaterial } from "@react-three/drei"
import * as THREE from "three"
import { EffectComposer, Bloom, Scanline, Vignette, Noise } from "@react-three/postprocessing"
import { BlendFunction } from "postprocessing"
import { GlassSkull } from "./glass-skull"

// -----------------------------------------------------------------------------
// 1. Custom Holographic Shader Material
// -----------------------------------------------------------------------------
const HologramMaterial = shaderMaterial(
  {
    uTime: 0,
    uColorFrontal: new THREE.Color("#00d4ff"),
    uColorParietal: new THREE.Color("#00ff88"),
    uColorTemporal: new THREE.Color("#ff00ff"),
    uColorOccipital: new THREE.Color("#ff8800"),
  },
  // Vertex Shader
  `
    uniform float uTime;
    attribute vec3 aRandom; // [speed, offset, size_variation]
    attribute float aRegion; // 0: frontal, 1: parietal, 2: temporal, 3: occipital
    
    varying vec3 vColor;
    varying float vAlpha;
    varying vec2 vUv;

    uniform vec3 uColorFrontal;
    uniform vec3 uColorParietal;
    uniform vec3 uColorTemporal;
    uniform vec3 uColorOccipital;

    void main() {
      vUv = uv;
      
      // Determine base color based on region
      if (aRegion < 0.5) vColor = uColorFrontal;
      else if (aRegion < 1.5) vColor = uColorParietal;
      else if (aRegion < 2.5) vColor = uColorTemporal;
      else vColor = uColorOccipital;

      // Dynamic position logic (Holographic wobble)
      vec3 pos = position;
      
      // Glitch effect: random jitter on X axis occasionally
      float glitch = step(0.98, sin(uTime * 2.0 + pos.y * 10.0));
      pos.x += glitch * (sin(uTime * 50.0) * 0.05);

      // Vertical scanning wave
      float scanWave = sin(pos.y * 4.0 - uTime * 2.0);
      float pulse = smoothstep(0.8, 1.0, scanWave) * 0.5;
      
      // Particle Size logic
      float baseSize = 0.04;
      // Pulse size when scan wave hits
      float size = baseSize * (1.0 + pulse * 2.0);
      
      // Add random twinkling
      float twinkle = sin(uTime * aRandom.x + aRandom.y) * 0.5 + 0.5;
      size *= (0.8 + 0.4 * twinkle);

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      gl_PointSize = size * (300.0 / -mvPosition.z); // Perspective scaling

      // Vary alpha based on scan wave
      vAlpha = 0.6 + 0.4 * pulse + 0.2 * twinkle;
    }
  `,
  // Fragment Shader
  `
    varying vec3 vColor;
    varying float vAlpha;

    void main() {
      // Circular particle shape
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;

      // Soft edge glow
      float glow = 1.0 - (dist * 2.0);
      glow = pow(glow, 1.5);

      vec3 finalColor = vColor * 1.5; // Boost brightness for bloom
      
      gl_FragColor = vec4(finalColor, vAlpha * glow);
    }
  `
)

extend({ HologramMaterial })

// Add types for TS
declare module "@react-three/fiber" {
  interface ThreeElements {
    hologramMaterial: any
  }
}

// -----------------------------------------------------------------------------
// 2. Brain Component
// -----------------------------------------------------------------------------
function BrainParticles({ count = 4000 }) {
  const pointsRef = useRef<THREE.Points>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)

  const particles = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const aRandom = new Float32Array(count * 3)
    const aRegion = new Float32Array(count) // Float attribute for region

    for (let i = 0; i < count; i++) {
      // rejection sampling for brain shape
      let x, y, z
      let found = false
      while (!found) {
        x = (Math.random() - 0.5) * 2.5
        y = (Math.random() - 0.5) * 2.0
        z = (Math.random() - 0.5) * 3.0

        const r = x * x / (1.0) + y * y / (0.8) + z * z / (1.5)
        const inner = x * x / (0.2) + y * y / (0.2) + z * z / (0.3) // Ventricles

        if (r < 1 && inner > 0.5) {
          if (Math.sin(x * 5) * Math.cos(z * 5) > -0.6) { // Cortex folds
            found = true
          }
        }
      }

      positions[i * 3] = x!
      positions[i * 3 + 1] = y!
      positions[i * 3 + 2] = z!

      // Random attributes for animation
      aRandom[i * 3] = Math.random() * 5.0 + 1.0 // speed
      aRandom[i * 3 + 1] = Math.random() * 10.0      // offset
      aRandom[i * 3 + 2] = Math.random()             // variation

      // Region encoding
      let region = 1.0 // Parietal default
      if (z! < -0.5) region = 0.0 // Frontal
      else if (z! > 0.8) region = 3.0 // Occipital
      else if (y! < -0.3) region = 2.0 // Temporal

      aRegion[i] = region
    }

    return { positions, aRandom, aRegion }
  }, [count])

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime
    }
    if (pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.15
      const s = 1 + Math.sin(state.clock.elapsedTime * 0.5) * 0.02
      pointsRef.current.scale.set(s, s, s)
    }
  })

  return (
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
      <hologramMaterial
        ref={materialRef}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  )
}

export function BrainVisualization() {
  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [0, 1, 4.5], fov: 45 }}
        style={{ width: "100%", height: "100%" }}
        gl={{ antialias: false, alpha: false }} // Optimizations for post-processing
      >
        <color attach="background" args={['#050510']} />

        <ambientLight intensity={0.2} />

        <BrainParticles count={6000} />
        <GlassSkull />

        {/* Post-Processing Effects */}
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.5}
            mipmapBlur
            intensity={1.2}
            radius={0.6}
          />
          <Scanline
            density={2.5}
            opacity={0.10}
          />
          <Vignette
            eskil={false}
            offset={0.1}
            darkness={0.7}
          />
          <Noise
            opacity={0.05}
            blendFunction={BlendFunction.OVERLAY}
          />
        </EffectComposer>

      </Canvas>

      {/* Tech Overlay UI */}
      <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
        {/* Top corners */}
        <div className="flex justify-between">
          <div className="h-2 w-16 bg-[#00d4ff]/30 rounded-sm" />
          <div className="h-2 w-16 bg-[#00d4ff]/30 rounded-sm" />
        </div>

        {/* Bottom Legend */}
        <div className="bg-black/50 backdrop-blur-md border border-white/10 px-4 py-2 rounded-lg text-[10px] font-mono text-muted-foreground self-start border-l-2 border-l-[#00d4ff]">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#00d4ff] animate-pulse" /> FRONTAL</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#00ff88] animate-pulse" /> PARIETAL</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#ff00ff] animate-pulse" /> TEMPORAL</span>
          </div>
          <div className="mt-1 text-[#00d4ff]/70">
            SYSTEM STATUS: HOLOGRAPHIC PROJECTION ACTIVE
          </div>
        </div>
      </div>
    </div>
  )
}
