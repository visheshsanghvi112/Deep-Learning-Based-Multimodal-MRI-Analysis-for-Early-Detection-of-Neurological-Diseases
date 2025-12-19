"use client"

import React, { useRef, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Line } from "@react-three/drei"
import * as THREE from "three"

interface NeuronNode {
  position: [number, number, number]
  connections: number[]
}

function NeuralNetwork({ nodes }: { nodes: NeuronNode[] }) {
  const groupRef = useRef<THREE.Group>(null)
  const timeRef = useRef(0)

  useFrame((state) => {
    timeRef.current = state.clock.elapsedTime
    if (groupRef.current) {
      groupRef.current.rotation.z = Math.sin(timeRef.current * 0.2) * 0.1
    }
  })

  const lines = useMemo(() => {
    const lineElements: React.JSX.Element[] = []
    nodes.forEach((node, i) => {
      node.connections.forEach((targetIdx) => {
        const target = nodes[targetIdx]
        if (target) {
          const distance = Math.sqrt(
            Math.pow(node.position[0] - target.position[0], 2) +
            Math.pow(node.position[1] - target.position[1], 2) +
            Math.pow(node.position[2] - target.position[2], 2)
          )
          const opacity = Math.max(0.1, 1 - distance / 3)

          lineElements.push(
            <Line
              key={`${i}-${targetIdx}`}
              points={[node.position, target.position]}
              color="#00ff88"
              lineWidth={0.5 * opacity}
              transparent
              opacity={opacity}
            />
          )
        }
      })
    })
    return lineElements
  }, [nodes])

  return (
    <group ref={groupRef}>
      {lines}
      {nodes.map((node, i) => (
        <mesh key={i} position={node.position}>
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshStandardMaterial color="#00ff88" emissive="#00ff88" emissiveIntensity={0.5} />
        </mesh>
      ))}
    </group>
  )
}

function createNeuralNetworkNodes(count: number = 50): NeuronNode[] {
  const nodes: NeuronNode[] = []

  // Create layered structure like a neural network
  const layers = 4
  const nodesPerLayer = Math.floor(count / layers)

  for (let layer = 0; layer < layers; layer++) {
    const layerNodes = layer === layers - 1 ? count - (layers - 1) * nodesPerLayer : nodesPerLayer
    for (let i = 0; i < layerNodes; i++) {
      const x = (layer - layers / 2) * 2
      const y = (i - layerNodes / 2) * 0.5
      const z = (Math.random() - 0.5) * 0.5

      const connections: number[] = []
      if (layer < layers - 1) {
        // Connect to next layer
        const nextLayerStart = (layer + 1) * nodesPerLayer
        const nextLayerNodes = layer === layers - 2 ? count - (layers - 1) * nodesPerLayer : nodesPerLayer
        for (let j = 0; j < Math.min(3, nextLayerNodes); j++) {
          const targetIdx = nextLayerStart + Math.floor((i / layerNodes) * nextLayerNodes) + j
          if (targetIdx < count) connections.push(targetIdx)
        }
      }

      nodes.push({
        position: [x, y, z],
        connections
      })
    }
  }

  return nodes
}

export function NeuralNetworkBackground() {
  const nodes = useMemo(() => createNeuralNetworkNodes(60), [])

  return (
    <div className="absolute inset-0 -z-10 overflow-hidden opacity-20 dark:opacity-10">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 60 }}
        style={{ width: "100%", height: "100%" }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 5, 5]} intensity={0.5} color="#00ff88" />
        <NeuralNetwork nodes={nodes} />
      </Canvas>
    </div>
  )
}

