"use client"

import { useRef, useMemo, memo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import * as THREE from "three"

// MRI slice visualization - representing structural brain imaging
const MRISlicePattern = memo(function MRISlicePattern() {
  const meshRef = useRef<THREE.Mesh>(null)
  const shaderMaterialRef = useRef<THREE.ShaderMaterial>(null)

  useFrame((state) => {
    if (shaderMaterialRef.current) {
      shaderMaterialRef.current.uniforms.time.value = state.clock.elapsedTime
    }
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.02) * 0.05
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
    
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
    
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
      p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }
    
    void main() {
      vec2 uv = vUv;
      vec3 pos1 = vec3(uv * 3.0, time * 0.1);
      float n1 = snoise(pos1);
      float intensity1 = smoothstep(0.3, 0.7, n1);
      vec3 pos2 = vec3(uv * 4.0 + 0.5, time * 0.08);
      float n2 = snoise(pos2);
      float intensity2 = smoothstep(0.4, 0.8, n2) * 0.6;
      vec3 pos3 = vec3(uv * 5.0 - 0.3, time * 0.06);
      float n3 = snoise(pos3);
      float intensity3 = smoothstep(0.2, 0.6, abs(n3)) * 0.4;
      float totalIntensity = (intensity1 + intensity2 + intensity3) / 2.0;
      vec3 color = vec3(totalIntensity * 0.5);
      float edge = abs(n1 - n2) * 2.0;
      edge = smoothstep(0.15, 0.35, edge);
      color += vec3(edge * 0.1);
      color += vec3(0.01, 0.012, 0.015) * totalIntensity;
      gl_FragColor = vec4(color, 0.45);
    }
  `

  return (
    <mesh ref={meshRef} position={[0, 0, -5]}>
      <planeGeometry args={[40, 40, 128, 128]} />
      <shaderMaterial
        ref={shaderMaterialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{ time: { value: 0 } }}
        transparent
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
})

// Brain connectivity network
const BrainConnectivityNetwork = memo(function BrainConnectivityNetwork() {
  const groupRef = useRef<THREE.Group>(null)

  const { nodes, connections } = useMemo(() => {
    const brainRegions = [
      { pos: [-2.5, 1.5, 0] }, { pos: [-2, 0.5, 0.5] }, { pos: [0, 2.5, 0] },
      { pos: [2, 2, 0] }, { pos: [2.5, 0, 0] }, { pos: [0, -2, -0.5] },
      { pos: [0, 0, 0.8] }, { pos: [-1.5, -1, 0.3] }, { pos: [2.5, 1.5, 0] },
      { pos: [2, 0.5, 0.5] },
    ]
    const nodes: THREE.Vector3[] = brainRegions.map(r => new THREE.Vector3(r.pos[0], r.pos[1], r.pos[2]))
    const connections: [number, number][] = [
      [0, 1], [8, 9], [0, 8], [1, 9], [0, 2], [0, 4], [1, 4], [2, 3], [4, 3], [0, 6], [6, 2], [4, 5], [0, 7],
    ]
    return { nodes, connections }
  }, [])

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.01) * 0.02
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.05) * 0.05
    }
  })

  const lineGeometry = useMemo(() => {
    const positions = new Float32Array(connections.length * 6)
    connections.forEach(([start, end], i) => {
      const s = nodes[start], e = nodes[end]
      positions[i * 6] = s.x; positions[i * 6 + 1] = s.y; positions[i * 6 + 2] = s.z
      positions[i * 6 + 3] = e.x; positions[i * 6 + 4] = e.y; positions[i * 6 + 5] = e.z
    })
    return new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(positions, 3))
  }, [nodes, connections])

  const pointGeometry = useMemo(() => {
    const positions = new Float32Array(nodes.length * 3)
    nodes.forEach((node, i) => {
      positions[i * 3] = node.x; positions[i * 3 + 1] = node.y; positions[i * 3 + 2] = node.z
    })
    return new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(positions, 3))
  }, [nodes])

  return (
    <group ref={groupRef}>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial color="#666666" opacity={0.4} transparent linewidth={2} />
      </lineSegments>
      <points geometry={pointGeometry}>
        <pointsMaterial size={2.0} color="#555555" sizeAttenuation={false} transparent opacity={0.7} />
      </points>
    </group>
  )
})

function NeuroscienceBackground3D() {
  return (
    <div className="fixed inset-0 -z-10 pointer-events-none hidden dark:block">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 55 }}
        style={{ width: "100%", height: "100%" }}
        gl={{ alpha: true, antialias: false, powerPreference: "high-performance" }}
        frameloop="always"
      >
        <ambientLight intensity={0.9} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} />
        <MRISlicePattern />
        <BrainConnectivityNetwork />
      </Canvas>
    </div>
  )
}

export const NeuroscienceBackground = memo(function NeuroscienceBackground() {
  return (
    <>
      <div className="fixed inset-0 -z-20 bg-background pointer-events-none" />
      <NeuroscienceBackground3D />
    </>
  )
})
