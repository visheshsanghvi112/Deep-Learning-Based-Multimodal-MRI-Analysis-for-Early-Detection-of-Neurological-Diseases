"use client"

import { useRef, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Points, PointMaterial } from "@react-three/drei"
import * as THREE from "three"

interface DNAHelixProps {
  position?: [number, number, number]
  scale?: number
}

function DNAHelix({ position = [0, 0, 0], scale = 1 }: DNAHelixProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const helixRef = useRef<THREE.Group>(null)
  
  // Create DNA helix geometry with double helix structure
  const dnaPoints = useMemo(() => {
    const strand1: number[] = []
    const strand2: number[] = []
    const connectors: number[] = []
    
    const radius = 0.8
    const turns = 3
    const pointsPerTurn = 50
    const totalPoints = turns * pointsPerTurn
    
    for (let i = 0; i < totalPoints; i++) {
      const t = (i / totalPoints) * Math.PI * 2 * turns
      const y = (i / totalPoints) * 4 - 2
      
      // First strand
      const x1 = Math.cos(t) * radius
      const z1 = Math.sin(t) * radius
      strand1.push(x1, y, z1)
      
      // Second strand (180° out of phase)
      const x2 = Math.cos(t + Math.PI) * radius
      const z2 = Math.sin(t + Math.PI) * radius
      strand2.push(x2, y, z2)
      
      // Base pairs (connectors)
      if (i % 5 === 0) {
        connectors.push(x1, y, z1)
        connectors.push(x2, y, z2)
      }
    }
    
    // Combine all points
    const allPoints = [...strand1, ...strand2, ...connectors]
    return new Float32Array(allPoints)
  }, [])
  
  useFrame((state) => {
    if (helixRef.current) {
      helixRef.current.rotation.y += 0.005
      helixRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.2
    }
  })
  
  return (
    <group ref={helixRef} position={position} scale={scale}>
      <Points ref={pointsRef} positions={dnaPoints} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="#00d4ff"
          size={0.05}
          sizeAttenuation={true}
          depthWrite={false}
          opacity={0.8}
        />
      </Points>
    </group>
  )
}

export function DNAHelixBackground() {
  return (
    <div className="absolute inset-0 -z-10 overflow-hidden opacity-30 dark:opacity-20">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 75 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#00d4ff" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ff00ff" />
        <DNAHelix />
        <DNAHelix position={[3, 1, -2]} scale={0.6} />
        <DNAHelix position={[-3, -1, -2]} scale={0.6} />
      </Canvas>
    </div>
  )
}

