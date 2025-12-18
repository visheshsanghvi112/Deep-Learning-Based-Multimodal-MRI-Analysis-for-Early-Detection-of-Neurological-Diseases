"use client"

import { useRef } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Sphere, MeshDistortMaterial, Float } from "@react-three/drei"
import * as THREE from "three"

function BrainLobe({ position, scale, color, delay = 0 }: { 
  position: [number, number, number]
  scale: number
  color: string
  delay?: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.elapsedTime + delay
      meshRef.current.position.y += Math.sin(time * 2) * 0.002
      meshRef.current.rotation.y = Math.sin(time * 0.5) * 0.2
    }
  })
  
  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.5}>
      <Sphere ref={meshRef} args={[1, 32, 32]} position={position} scale={scale}>
        <meshStandardMaterial
          color={color}
          roughness={0.7}
          metalness={0.1}
          transparent
          opacity={0.3}
        />
      </Sphere>
    </Float>
  )
}

function BrainScan() {
  const groupRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.1
    }
  })
  
  return (
    <group ref={groupRef}>
      {/* Frontal lobe */}
      <BrainLobe position={[0, 1.2, 0]} scale={0.8} color="#00d4ff" delay={0} />
      
      {/* Parietal lobes */}
      <BrainLobe position={[-0.8, 0.3, 0.5]} scale={0.6} color="#00ff88" delay={0.3} />
      <BrainLobe position={[0.8, 0.3, 0.5]} scale={0.6} color="#00ff88" delay={0.6} />
      
      {/* Temporal lobes */}
      <BrainLobe position={[-1, -0.5, 0]} scale={0.7} color="#ff00ff" delay={0.9} />
      <BrainLobe position={[1, -0.5, 0]} scale={0.7} color="#ff00ff" delay={1.2} />
      
      {/* Occipital lobe */}
      <BrainLobe position={[0, -1.2, -0.3]} scale={0.65} color="#ff8800" delay={1.5} />
      
      {/* Brain stem */}
      <BrainLobe position={[0, -1.8, 0]} scale={0.3} color="#ffff00" delay={1.8} />
    </group>
  )
}

export function BrainVisualization() {
  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [0, 0, 6], fov: 50 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 5, 5]} intensity={1} color="#00d4ff" />
        <pointLight position={[-5, -5, -5]} intensity={0.5} color="#ff00ff" />
        <directionalLight position={[0, 5, 0]} intensity={0.3} />
        <BrainScan />
      </Canvas>
    </div>
  )
}

