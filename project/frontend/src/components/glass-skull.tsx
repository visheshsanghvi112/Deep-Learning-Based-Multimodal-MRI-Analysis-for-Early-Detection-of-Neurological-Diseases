"use client"

import { useRef } from "react"
import { useFrame } from "@react-three/fiber"
import { Sphere, MeshTransmissionMaterial } from "@react-three/drei"
import * as THREE from "three"

export function GlassSkull({ scale = 1.2 }: { scale?: number }) {
    const meshRef = useRef<THREE.Mesh>(null)

    useFrame((state) => {
        if (meshRef.current) {
            // Slowly rotate opposite to brain for dynamic effect
            meshRef.current.rotation.y = -state.clock.elapsedTime * 0.05
        }
    })

    return (
        <Sphere ref={meshRef} args={[1.5, 64, 64]} scale={scale}>
            <MeshTransmissionMaterial
                backside
                samples={4}
                thickness={0.5}
                chromaticAberration={0.05}
                anisotropy={0.1}
                distortion={0.1}
                distortionScale={0.1}
                temporalDistortion={0.1}
                iridescence={1}
                iridescenceIOR={1}
                iridescenceThicknessRange={[0, 1400]}
                roughness={0.1}
                background={new THREE.Color("#000000")}
            />
        </Sphere>
    )
}
