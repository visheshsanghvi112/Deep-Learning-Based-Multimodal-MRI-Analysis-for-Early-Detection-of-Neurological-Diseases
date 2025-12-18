"use client"

import { useRef, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import * as THREE from "three"

// Advanced shader-based MRI scan visualization
function MRIScanShader() {
  const meshRef = useRef<THREE.Mesh>(null)
  const shaderMaterialRef = useRef<THREE.ShaderMaterial>(null)
  
  useFrame((state) => {
    if (shaderMaterialRef.current) {
      shaderMaterialRef.current.uniforms.time.value = state.clock.elapsedTime
    }
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.05) * 0.1
    }
  })
  
  const vertexShader = `
    varying vec2 vUv;
    varying vec3 vPosition;
    void main() {
      vUv = uv;
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `
  
  const fragmentShader = `
    uniform float time;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    // 3D noise function for volumetric MRI-like patterns
    vec3 mod289(vec3 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    
    vec4 mod289(vec4 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    
    vec4 permute(vec4 x) {
      return mod289(((x*34.0)+1.0)*x);
    }
    
    vec4 taylorInvSqrt(vec4 r) {
      return 1.79284291400159 - 0.85373472095314 * r;
    }
    
    float snoise(vec3 v) {
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
      vec3 i = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      i = mod289(i);
      vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      float n_ = 0.142857142857;
      vec3 ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
      vec3 p0 = vec3(a0.xy,h.x);
      vec3 p1 = vec3(a0.zw,h.y);
      vec3 p2 = vec3(a1.xy,h.z);
      vec3 p3 = vec3(a1.zw,h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }
    
    void main() {
      vec3 pos = vPosition * 0.5 + time * 0.1;
      float n = snoise(pos);
      float n2 = snoise(pos * 2.0 + time * 0.05);
      float n3 = snoise(pos * 4.0 + time * 0.02);
      
      // Create MRI-like intensity mapping
      float intensity = (n + n2 * 0.5 + n3 * 0.25) / 1.75;
      intensity = smoothstep(0.3, 0.7, intensity);
      
      // Professional grayscale with clear visibility
      vec3 color = vec3(intensity * 0.4);
      color += vec3(0.02, 0.025, 0.03) * intensity;
      
      // Edge detection for anatomical structures
      float edge = abs(n - n2) * 3.0;
      edge = smoothstep(0.1, 0.4, edge);
      color += vec3(edge * 0.08);
      
      // Clearly visible but professional
      gl_FragColor = vec4(color, 0.5);
    }
  `
  
  return (
    <mesh ref={meshRef} position={[0, 0, -5]}>
      <planeGeometry args={[30, 30, 128, 128]} />
      <shaderMaterial
        ref={shaderMaterialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          time: { value: 0 }
        }}
        transparent
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

// Procedurally generated neural network structure
function ProceduralNeuralNetwork() {
  const groupRef = useRef<THREE.Group>(null)
  
  const { nodes, edges } = useMemo(() => {
    // Generate a more realistic neural network structure
    const nodeCount = 80
    const nodes: THREE.Vector3[] = []
    
    // Layered structure (input, hidden layers, output)
    const layers = 5
    const nodesPerLayer = Math.ceil(nodeCount / layers)
    
    for (let layer = 0; layer < layers; layer++) {
      const nodesInLayer = layer === layers - 1 ? nodeCount - (layers - 1) * nodesPerLayer : nodesPerLayer
      for (let i = 0; i < nodesInLayer; i++) {
        const x = (layer / layers - 0.5) * 8
        const y = ((i / nodesInLayer) - 0.5) * 6
        const z = (Math.random() - 0.5) * 0.5
        nodes.push(new THREE.Vector3(x, y, z))
      }
    }
    
    // Create edges (connections)
    const edges: [number, number][] = []
    let nodeIndex = 0
    for (let layer = 0; layer < layers - 1; layer++) {
      const nodesInLayer = layer === layers - 1 ? nodeCount - (layers - 1) * nodesPerLayer : nodesPerLayer
      const nextLayerStart = nodeIndex + nodesInLayer
      const nextLayerNodes = layer === layers - 2 ? nodeCount - (layers - 1) * nodesPerLayer : nodesPerLayer
      
      for (let i = 0; i < nodesInLayer; i++) {
        for (let j = 0; j < Math.min(3, nextLayerNodes); j++) {
          const targetIdx = nextLayerStart + Math.floor((i / nodesInLayer) * nextLayerNodes) + (j % nextLayerNodes)
          if (targetIdx < nodeCount) {
            edges.push([nodeIndex + i, targetIdx])
          }
        }
      }
      nodeIndex += nodesInLayer
    }
    
    return { nodes, edges }
  }, [])
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.02) * 0.05
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.1
    }
  })
  
  const lineGeometry = useMemo(() => {
    const positions = new Float32Array(edges.length * 6)
    edges.forEach(([start, end], i) => {
      const startPos = nodes[start]
      const endPos = nodes[end]
      positions[i * 6] = startPos.x
      positions[i * 6 + 1] = startPos.y
      positions[i * 6 + 2] = startPos.z
      positions[i * 6 + 3] = endPos.x
      positions[i * 6 + 4] = endPos.y
      positions[i * 6 + 5] = endPos.z
    })
    return new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(positions, 3))
  }, [nodes, edges])
  
  const pointGeometry = useMemo(() => {
    const positions = new Float32Array(nodes.length * 3)
    nodes.forEach((node, i) => {
      positions[i * 3] = node.x
      positions[i * 3 + 1] = node.y
      positions[i * 3 + 2] = node.z
    })
    return new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(positions, 3))
  }, [nodes])
  
  return (
    <group ref={groupRef}>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial color="#444444" opacity={0.35} transparent />
      </lineSegments>
      <points geometry={pointGeometry}>
        <pointsMaterial size={1.5} color="#666666" sizeAttenuation={false} />
      </points>
    </group>
  )
}

function ProfessionalBackground3D() {
  return (
    <div className="fixed inset-0 -z-10 pointer-events-none">
      <Canvas
        camera={{ position: [0, 0, 12], fov: 50 }}
        style={{ width: "100%", height: "100%" }}
        gl={{ alpha: true, antialias: true, powerPreference: "high-performance" }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={0.5} />
        <directionalLight position={[-5, -5, -5]} intensity={0.4} />
        
        {/* MRI-like volumetric shader - professional but visible */}
        <MRIScanShader />
        
        {/* Neural network structure - professional but visible */}
        <ProceduralNeuralNetwork />
      </Canvas>
    </div>
  )
}

export function ProfessionalScientificBackground() {
  return (
    <>
      <ProfessionalBackground3D />
      {/* Minimal overlay to maintain readability */}
      <div className="fixed inset-0 -z-10 bg-gradient-to-b from-background/50 via-background/60 to-background/70 pointer-events-none" />
    </>
  )
}

