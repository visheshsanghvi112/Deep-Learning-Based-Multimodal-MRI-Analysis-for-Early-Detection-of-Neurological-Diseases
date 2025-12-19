"use client"

import { useRef, useState, useMemo, useEffect } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Plane, Box } from "@react-three/drei"
import * as THREE from "three"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Brain, Scan, Activity, AlertCircle, Maximize2, Layers } from "lucide-react"
import { generateBrainVolume } from "@/lib/volume-generator"

// -----------------------------------------------------------------------------
// Volumetric Raymarching Shader
// -----------------------------------------------------------------------------
const VolumetricShaderMaterial = {
    uniforms: {
        uVolume: { value: null }, // Data3DTexture
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2() },
        uCameraPos: { value: new THREE.Vector3() },
        uSliceIndex: { value: 0.5 },    // For MPR slice
        uAxis: { value: 0 },            // 0: Axial, 1: Coronal, 2: Sagittal
        uMode: { value: 0 },            // 0: Slice (MPR), 1: MIP (Glass Brain), 2: ISO (Surface)
        uThreshold: { value: 0.15 },    // Isosurface threshold
        uWindowLevel: { value: 0.5 },   // Brightness
        uWindowWidth: { value: 0.8 },   // Contrast
        uShowHeatmap: { value: 0.0 },   // AI Overlay
        uSteps: { value: 128 },         // Raymarching steps
    },
    vertexShader: `
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
    fragmentShader: `
    precision highp float;
    precision highp sampler3D;

    uniform float uTime;
    uniform sampler3D uVolume;
    uniform float uSliceIndex;
    uniform int uAxis;
    uniform int uMode; // 0=Slice, 1=MIP, 2=ISO
    
    uniform float uThreshold;
    uniform float uWindowLevel;
    uniform float uWindowWidth;
    uniform float uShowHeatmap;
    
    varying vec2 vUv;
    varying vec3 vPosition;

    // AABB Intersection (Ray vs Box)
    // Box is -0.5 to 0.5
    vec2 hitBox(vec3 orig, vec3 dir) {
        vec3 boxMin = vec3(-0.5);
        vec3 boxMax = vec3(0.5);
        vec3 invDir = 1.0 / dir;
        vec3 tmin = (boxMin - orig) * invDir;
        vec3 tmax = (boxMax - orig) * invDir;
        vec3 t1 = min(tmin, tmax);
        vec3 t2 = max(tmin, tmax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        return vec2(tNear, tFar);
    }

    // Apply Window/Level to raw density
    float applyWindowLevel(float v) {
        float minVal = uWindowLevel - uWindowWidth * 0.5;
        float maxVal = uWindowLevel + uWindowWidth * 0.5;
        return clamp((v - minVal) / (maxVal - minVal), 0.0, 1.0);
    }

    // Get Heatmap color for position
    vec4 getHeatmap(vec3 pos) {
        if (uShowHeatmap < 0.5) return vec4(0.0);
        
        // Temporal Lobe Hotspots (approximate in -0.5 to 0.5 space)
        vec3 rightTemp = vec3(0.25, -0.1, 0.0);
        vec3 leftTemp = vec3(-0.25, -0.1, 0.0);
        
        float d1 = length(pos - rightTemp);
        float d2 = length(pos - leftTemp);
        
        float pulse = 0.5 + 0.5 * sin(uTime * 3.0);
        float intensity = 0.0;
        
        intensity += smoothstep(0.15, 0.0, d1) * pulse;
        intensity += smoothstep(0.15, 0.0, d2) * (1.0 - pulse * 0.5);
        
        vec3 color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), intensity);
        return vec4(color, intensity * 0.8);
    }

    void main() {
        // MPR (2D Slice) Mode
        if (uMode == 0) {
            vec3 pos;
            float slice = clamp(uSliceIndex, 0.01, 0.99); // Avoid edge artifacts
            
            if (uAxis == 0) { // Axial (Z-slice in texture coords usually, Y-slice in world)
                 // Map UV to XY plane, fixed Z
                 pos = vec3(vUv.x, vUv.y, slice);
            } else if (uAxis == 1) { // Coronal
                 pos = vec3(vUv.x, slice, vUv.y);
            } else { // Sagittal
                 pos = vec3(slice, vUv.y, vUv.x);
            }
            
            float val = texture(uVolume, pos).r;
            float displayVal = applyWindowLevel(val);
            
            vec3 color = vec3(displayVal);
            
            // Heatmap overlay
            // Convert texture space [0,1] to World Space [-0.5, 0.5] for hotspot check
            vec3 worldPos = pos - 0.5;
            vec4 heat = getHeatmap(worldPos);
            color += heat.rgb * heat.a;

            // Scanlines
            float scan = sin(vUv.y * 150.0 + uTime * 2.0) * 0.02;
            color += scan;

            gl_FragColor = vec4(color, 1.0);
            return;
        }

        // Raymarching Initialization for MIP/ISO
        vec3 rayOrigin = cameraPosition;
        vec3 rayDir = normalize(vPosition - cameraPosition);

        // Intersect volume box
        vec2 t = hitBox(rayOrigin, rayDir);
        
        if (t.x > t.y) discard; // Missed box
        
        t.x = max(t.x, 0.0); // Clamp near plane
        
        vec3 p = rayOrigin + rayDir * t.x;
        vec3 step = rayDir * ((t.y - t.x) / 128.0); // 128 steps
        
        float maxVal = 0.0;
        vec4 accColor = vec4(0.0);
        
        // Raymarch Loop
        for(int i=0; i<128; i++) {
            // Map world pos [-0.5, 0.5] to Texture UV [0, 1]
            vec3 texCoord = p + 0.5;
            
            if(texCoord.x < 0.0 || texCoord.x > 1.0 ||
               texCoord.y < 0.0 || texCoord.y > 1.0 ||
               texCoord.z < 0.0 || texCoord.z > 1.0) break;
               
            float val = texture(uVolume, texCoord).r;
            
            // Mode 1: MIP (Maximum Intensity Projection)
            if (uMode == 1) {
                if (val > maxVal) maxVal = val;
                
                // Add heatmap integration in 3D
                vec4 heat = getHeatmap(p);
                if (heat.a > 0.1) maxVal += heat.a * 0.05; // Make hot areas glow in MIP
            } 
            // Mode 2: Accumulated Density (Volumetric Fog / Glassy)
            else if (uMode == 2) {
                float opacity = val * 0.1; // Density factor
                if (val > uThreshold) {
                    accColor.rgb += (1.0 - accColor.a) * vec3(val) * opacity;
                    accColor.a += (1.0 - accColor.a) * opacity;
                    
                    vec4 heat = getHeatmap(p);
                    accColor.rgb += heat.rgb * heat.a * 0.2;
                }
            }
            
            p += step;
            if(accColor.a >= 0.95) break; 
        }
        
        vec3 finalColor;
        
        if (uMode == 1) {
            // MIP Result
            float displayVal = applyWindowLevel(maxVal);
            finalColor = vec3(displayVal * 0.8 + 0.1); // Blue tint base
            finalColor += vec3(0.0, 0.1, 0.2); // Cool tone
        } else {
            // Volumetric Result
            finalColor = accColor.rgb;
        }

        gl_FragColor = vec4(finalColor, 1.0);
    }
  `
}

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

function VolumetricRenderer({
    volume,
    slice,
    axis,
    mode,
    showHeatmap,
    windowLevel,
    windowWidth
}: {
    volume: THREE.Data3DTexture | null,
    slice: number,
    axis: number,
    mode: number,
    showHeatmap: boolean,
    windowLevel: number,
    windowWidth: number
}) {
    const materialRef = useRef<THREE.ShaderMaterial>(null)
    const { camera } = useThree()

    useFrame((state) => {
        if (materialRef.current && volume) {
            materialRef.current.uniforms.uVolume.value = volume
            materialRef.current.uniforms.uTime.value = state.clock.elapsedTime
            materialRef.current.uniforms.uSliceIndex.value = slice
            materialRef.current.uniforms.uAxis.value = axis
            materialRef.current.uniforms.uMode.value = mode
            materialRef.current.uniforms.uShowHeatmap.value = showHeatmap ? 1.0 : 0.0

            // Window/Level Controls
            materialRef.current.uniforms.uWindowLevel.value = windowLevel
            materialRef.current.uniforms.uWindowWidth.value = windowWidth

            // Pass simple camera uniform even though three.js does it, just for clarity/safety in custom shader
            materialRef.current.uniforms.uCameraPos.value.copy(camera.position)
        }
    })

    return (
        // A box acting as the bounds for the volume raymarching
        <Box args={[1, 1, 1]}>
            {/* @ts-ignore */}
            <shaderMaterial
                ref={materialRef}
                args={[VolumetricShaderMaterial]}
                transparent
                side={THREE.DoubleSide}
            />
        </Box>
    )
}

export function MRISliceViewer() {
    // 1. Data State
    const [volume, setVolume] = useState<THREE.Data3DTexture | null>(null)
    const [loading, setLoading] = useState(true)

    // 2. View State
    const [slice, setSlice] = useState(0.5)
    const [axis, setAxis] = useState("axial")
    const [mode, setMode] = useState("slice") // slice, mip, iso
    const [heatmap, setHeatmap] = useState(false)

    // 3. Clinical Controls
    const [windowLevel, setWindowLevel] = useState(0.6) // Initial brightness
    const [windowWidth, setWindowWidth] = useState(0.8) // Initial contrast

    const axisMap: Record<string, number> = { "axial": 0, "coronal": 1, "sagittal": 2 }
    const modeMap: Record<string, number> = { "slice": 0, "mip": 1, "iso": 2 }

    useEffect(() => {
        // Generate Volume Async
        const size = 128;
        const data = generateBrainVolume(size);
        const texture = new THREE.Data3DTexture(data, size, size, size);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.unpackAlignment = 1;
        texture.needsUpdate = true;

        setVolume(texture);
        setLoading(false);

        return () => {
            texture.dispose();
        }
    }, [])

    return (
        <Card className="w-full overflow-hidden border-2 border-border/50 relative bg-black/20">
            {/* Header / Top Controls */}
            <div className="absolute top-0 right-0 p-2 z-10 flex gap-2">
                <div className="flex items-center space-x-2 bg-background/80 backdrop-blur-sm p-1.5 rounded-full border border-border">
                    <Switch
                        id="heatmap-mode"
                        checked={heatmap}
                        onCheckedChange={setHeatmap}
                        className="data-[state=checked]:bg-red-500"
                    />
                    <Label htmlFor="heatmap-mode" className="text-[10px] font-bold pr-2 cursor-pointer select-none flex items-center gap-1">
                        <Activity className="w-3 h-3 text-red-500" />
                        AI HEATMAP
                    </Label>
                </div>
            </div>

            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Scan className="w-5 h-5 text-blue-400" />
                    Volumetric Neural Analysis
                </CardTitle>
                <CardDescription className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full animate-pulse ${loading ? 'bg-yellow-500' : 'bg-green-500'}`} />
                    {loading ? "GENERATING 3D VOLUME..." : "DICOM/NIfTI EMULATION ACTIVE"}
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid md:grid-cols-[1fr_300px] gap-6">
                    {/* Viewport */}
                    <div className="relative aspect-square bg-black rounded-lg overflow-hidden border border-white/10 shadow-2xl group cursor-move">

                        {/* Holographic Overlays */}
                        <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-blue-500/50 rounded-tl-lg pointer-events-none" />
                        <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-blue-500/50 rounded-tr-lg pointer-events-none" />
                        <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-blue-500/50 rounded-bl-lg pointer-events-none" />
                        <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-blue-500/50 rounded-br-lg pointer-events-none" />

                        {/* Info Text */}
                        <div className="absolute top-4 left-6 z-30 font-mono text-[10px] text-blue-400/80 space-y-1 pointer-events-none">
                            <div>MODE: {mode.toUpperCase()}</div>
                            <div>AXIS: {axis.toUpperCase()}</div>
                            <div>WL: {windowLevel.toFixed(2)} / WW: {windowWidth.toFixed(2)}</div>
                        </div>

                        {/* 3D Scene */}
                        <Canvas camera={{ position: [0, 0, 2], fov: 50 }}>
                            <ambientLight intensity={0.5} />

                            {!loading && (
                                <VolumetricRenderer
                                    volume={volume}
                                    slice={slice}
                                    axis={axisMap[axis]}
                                    mode={modeMap[mode]}
                                    showHeatmap={heatmap}
                                    windowLevel={windowLevel}
                                    windowWidth={windowWidth}
                                />
                            )}
                        </Canvas>
                    </div>

                    {/* Controls */}
                    <div className="space-y-6 flex flex-col justify-center">

                        {/* Rendering Mode */}
                        <div className="space-y-4">
                            <Label className="text-xs text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <Layers className="w-3 h-3" /> Rendering Mode
                            </Label>
                            <Tabs value={mode} onValueChange={(v) => setMode(v)} className="w-full">
                                <TabsList className="grid w-full grid-cols-3 bg-muted/50">
                                    <TabsTrigger value="slice" className="text-xs">MPR</TabsTrigger>
                                    <TabsTrigger value="mip" className="text-xs">MIP</TabsTrigger>
                                    <TabsTrigger value="iso" className="text-xs">3D</TabsTrigger>
                                </TabsList>
                            </Tabs>
                        </div>

                        {/* Axis Selection (Only relevant for Slice Mode) */}
                        <div className={`space-y-4 transition-opacity duration-300 ${mode !== 'slice' ? 'opacity-50 pointer-events-none' : ''}`}>
                            <Label className="text-xs text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <Maximize2 className="w-3 h-3" /> Plane
                            </Label>
                            <Tabs value={axis} onValueChange={(v) => setAxis(v)} className="w-full">
                                <TabsList className="grid w-full grid-cols-3 bg-muted/50">
                                    <TabsTrigger value="axial" className="text-xs">AXIAL</TabsTrigger>
                                    <TabsTrigger value="coronal" className="text-xs">COR</TabsTrigger>
                                    <TabsTrigger value="sagittal" className="text-xs">SAG</TabsTrigger>
                                </TabsList>
                            </Tabs>
                        </div>

                        {/* Slice Slider */}
                        <div className={`space-y-4 bg-muted/30 p-4 rounded-xl border border-border/50 ${mode !== 'slice' ? 'opacity-50' : ''}`}>
                            <div className="flex items-center justify-between">
                                <Label className="text-xs uppercase tracking-wider">Slice Depth</Label>
                                <span className="font-mono text-xs text-blue-400">{(slice * 100).toFixed(0)}%</span>
                            </div>
                            <Slider
                                disabled={mode !== 'slice'}
                                value={[slice]}
                                min={0}
                                max={1}
                                step={0.005}
                                onValueChange={(vals) => setSlice(vals[0])}
                                className="py-2"
                            />
                        </div>

                        {/* Window/Level Controls (Radio-style) */}
                        <div className="space-y-2 pt-2">
                            <Label className="text-xs uppercase tracking-wider text-muted-foreground">Radiology Controls</Label>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <Label className="text-[10px]">Brightness (Level)</Label>
                                    <Slider value={[windowLevel]} min={0} max={1} step={0.01} onValueChange={v => setWindowLevel(v[0])} />
                                </div>
                                <div>
                                    <Label className="text-[10px]">Contrast (Width)</Label>
                                    <Slider value={[windowWidth]} min={0.1} max={2} step={0.01} onValueChange={v => setWindowWidth(v[0])} />
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
